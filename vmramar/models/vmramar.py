from vmramar.models.factory import load_model, RegisterModel, get_model_by_name
import torch
import torch.nn as nn

@RegisterModel("vmra_mar")
class VMRAMaR(nn.Module):
    def __init__(self, args):
        super(VMRAMaR, self).__init__()
        self.args = args
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)
        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        if args.vmrnn_snapshot is not None:
            self.vmrnn = load_model(args.vmrnn_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.vmrnn = get_model_by_name('vmrnn', False, args)
        self.use_asymmetry = getattr(args, "use_asymmetry", False)
        if self.use_asymmetry:
            self.sad = get_model_by_name('sad', False, args)
            self.lat = get_model_by_name('lat', False, args)
        self.ahl = get_model_by_name('ahl', False, args)

    def forward(self, x, risk_factors=None, batch=None):
        B, T, V, C, H, W = x.size()
        x = x.view(B * T * V, C, H, W)
        if risk_factors is not None:
            risk_factors_per_img = [
                factor.expand([V, *factor.size()]).contiguous().view(-1, factor.size(-1))
                for factor in risk_factors
            ]
        else:
            risk_factors_per_img = None
        _, img_feats, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_feats = img_feats.view(B, T, V, -1)
        fused_feats = img_feats.mean(dim=2)
        temporal_output, hidden_states = self.vmrnn(fused_feats, risk_factors, batch)
        if self.use_asymmetry:
            left_feats = img_feats[:, :, 0, :]
            right_feats = img_feats[:, :, 1, :]
            aligned_right_feats = self.sad(right_feats)
            asym_feats = torch.abs(left_feats - aligned_right_feats)
            asym_feature = self.lat(asym_feats)
            temporal_feature = temporal_output.mean(dim=1)
            combined_feats = torch.cat([temporal_feature, asym_feature], dim=1)
        else:
            combined_feats = temporal_output.mean(dim=1)
        risk_pred = self.ahl(combined_feats)
        return risk_pred, hidden_states
