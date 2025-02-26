from vmramar.models.factory import load_model, RegisterModel, get_model_by_name
import torch
import torch.nn as nn

@RegisterModel("vmra_mar")
class VMRAMaR(nn.Module):
    def __init__(self, args):
        super(VMRAMaR, self).__init__()
        self.args = args
        
        # Initialize image encoder (e.g., using Mirai weights)
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)
        
        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # Store image representation dimension
        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        
        # Initialize the VMRNN module for temporal encoding
        if args.vmrnn_snapshot is not None:
            self.vmrnn = load_model(args.vmrnn_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.vmrnn = get_model_by_name('vmrnn', False, args)
        
        # Optionally load asymmetry modules if enabled in args
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad = get_model_by_name('sad', False, args)  # Spatial Asymmetry Detector
            self.lat = get_model_by_name('lat', False, args)  # Longitudinal Asymmetry Tracker
        
        # Initialize Additive Hazard Layer (AHL) for risk prediction
        self.ahl = get_model_by_name('ahl', False, args)

    def forward(self, x, risk_factors=None, batch=None):
        """
        Expected input:
          x: a tensor of shape (B, T, V, C, H, W) where
             B = batch size,
             T = number of time steps (screenings),
             V = number of views per screening (e.g., left/right or four standard views),
             C, H, W = image channels, height, and width.
        """
        B, T, V, C, H, W = x.size()
        # Reshape to process all views from all time steps at once
        x = x.view(B * T * V, C, H, W)
        
        # Optionally, expand risk factors per image if provided
        if risk_factors is not None:
            risk_factors_per_img = [
                factor.expand([V, *factor.size()]).contiguous().view(-1, factor.size(-1))
                for factor in risk_factors
            ]
        else:
            risk_factors_per_img = None
        
        # Extract image features using the encoder
        _, img_feats, _ = self.image_encoder(x, risk_factors_per_img, batch)
        
        # Reshape features to (B, T, V, D)
        img_feats = img_feats.view(B, T, V, -1)
        
        # Fuse multi-view features for each time step 
        fused_feats = img_feats.mean(dim=2)  # shape: (B, T, D)
        
        # Temporal encoding using the VMRNN block
        temporal_output, hidden_states = self.vmrnn(fused_feats, risk_factors, batch)
        
        # Incorporate asymmetry features if enabled
        if self.use_asymmetry:
            # For simplicity, assume view 0 corresponds to the left and view 1 to the right.
            left_feats = img_feats[:, :, 0, :]   # shape: (B, T, D)
            right_feats = img_feats[:, :, 1, :]   # shape: (B, T, D)
            
            # Use the spatial asymmetry detector (SAD) to align right features
            aligned_right_feats = self.sad(right_feats)  # module specifics defined elsewhere
            
            # Compute the per-time-step asymmetry as the absolute difference
            asym_feats = torch.abs(left_feats - aligned_right_feats)  # (B, T, D)
            
            # Aggregate longitudinal asymmetry using the LAT module
            asym_feature = self.lat(asym_feats)  # returns a tensor of shape (B, D)
            
            # Reduce temporal output over time 
            temporal_feature = temporal_output.mean(dim=1)  # shape: (B, D)
            
            # Concatenate temporal and asymmetry features
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)  # shape: (B, 2*D)
        else:
            combined_feats = temporal_output.mean(dim=1)  # shape: (B, D)
        
        # Risk prediction via the Additive Hazard Layer (AHL)
        risk_pred = self.ahl(combined_feats)
        
        return risk_pred, hidden_states
