import torch
import torch.nn as nn
import pdb

@RegisterModel("ahl")
class AdditiveHazardLayer(nn.Module):
    def __init__(self, num_features, args, max_followup):
        super(AdditiveHazardLayer, self).__init__()
        self.args = args
        self.max_followup = max_followup
        
        # Linear layer to compute time-specific marginal hazards
        self.hazard_fc = nn.Linear(num_features, max_followup)
        # Linear layer to compute a baseline risk
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        
        # Create an upper-triangular mask (transposed lower-triangular)
        mask = torch.ones((max_followup, max_followup))
        mask = torch.tril(mask, diagonal=0)
        # Transpose so that for each time t we sum hazards from times 0 to t
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triangular_mask', mask)

    def hazards(self, x):
        # Compute raw hazards and apply ReLU to ensure non-negativity
        raw_hazards = self.hazard_fc(x)  # Shape: (B, max_followup)
        pos_hazards = self.relu(raw_hazards)
        return pos_hazards

    def forward(self, x):
        """
        x: Combined feature vector of shape (B, num_features)
        
        Returns:
            cum_hazard: Cumulative risk predictions for each follow-up time
                        (Shape: (B, max_followup))
        """
        # Option to make probabilities independent if specified in args
        if self.args.make_probs_indep:
            return self.hazards(x)
        
        hazards = self.hazards(x)  # (B, max_followup)
        B, T = hazards.size()  # Here, T equals max_followup
        
        # Expand hazards to compute cumulative risk across time intervals
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # (B, T, T)
        masked_hazards = expanded_hazards * self.upper_triangular_mask  # (B, T, T)
        
        # Sum along the time dimension and add a baseline risk term
        cum_hazard = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_hazard


