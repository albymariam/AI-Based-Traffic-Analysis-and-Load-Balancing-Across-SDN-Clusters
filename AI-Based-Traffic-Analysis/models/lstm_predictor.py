"""
PyTorch LSTM Link Utilization Predictor
Academic Research Project RM-T11 (Phase P2)

Implements Equation (3): U_hat_{l, t+1:t+H} = f_LSTM(X_{l,t}; Theta)
Forecasts short-term link utilization ratio over future horizon H.
"""

import torch
import torch.nn as nn

class LSTMTrafficPredictor(nn.Module):
    """
    Long Short-Term Memory (LSTM) Recurrent Neural Network for Temporal Link Utilization Forecasting.
    
    Architecture:
      1. LSTM Layer: Captures temporal dynamics across window W.
      2. Dropout Layer: Prevents overfitting on bursty telemetry.
      3. Fully Connected (FC) Layers: Maps hidden representation to forecast horizon H.
      4. Sigmoid Output: Bounds predicted utilization ratio in [0.0, 1.0].
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=5, dropout=0.1):
        super(LSTMTrafficPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Recurrent LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dense Prediction Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
            nn.Sigmoid()  # Guarantees output is a valid utilization ratio [0, 1]
        )

    def forward(self, x):
        """
        Forward Pass:
          x: Input tensor of shape (batch_size, sequence_length W=10, feature_dim=7)
          Returns: Predicted utilization tensor of shape (batch_size, forecast_horizon H=5)
        """
        lstm_out, (hn, cn) = self.lstm(x)
        last_timestep_out = lstm_out[:, -1, :]  # Extract last temporal state
        predictions = self.fc(last_timestep_out)
        return predictions

if __name__ == "__main__":
    print("[Phase P2] Testing LSTM Predictor Architecture (Equation 3)...")
    model = LSTMTrafficPredictor(input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=5)
    dummy_input = torch.randn(8, 10, 7)  # Batch=8, Window W=10, Features=7
    predicted_util = model(dummy_input)
    print(f"  Input Tensor Shape:  {dummy_input.shape}")
    print(f"  Output Tensor Shape: {predicted_util.shape} (Multi-step forecasts over H=5 horizon)")
