
import torch
import torch.nn as nn

class DynamicActivationPredictor(nn.Module):
    def __init__(self, input_dim=1152, hidden_dim=192, output_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.constant_(self.net[-1].weight, 0)
        torch.nn.init.constant_(self.net[-1].bias, 0)
        with torch.no_grad():
            self.net[-1].bias[0] = 1.0 

    def forward(self, t_emb):
        return self.net(t_emb)
