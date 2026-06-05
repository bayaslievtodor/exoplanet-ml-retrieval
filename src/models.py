import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


class ExoplanetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.aux_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(3328 + 32, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 15) # [q1, q2, q3] vector blocks grouped per gas
        )

    def forward(self, spectrum, aux):
        x_spec = self.conv(spectrum)
        x_aux = self.aux_net(aux)    
        
        x = torch.cat([x_spec, x_aux], dim=1)
        return self.dense(x)


def build_random_forest(n_estimators=100, max_depth=15, random_state=123, n_jobs=-1):
    return MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
    )