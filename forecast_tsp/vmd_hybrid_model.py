"""VMD-LSTM Hybrid model for 24h multi-step wind power forecasting.

Architecture:
  Input  → (batch, seq_len, 4 IMF channels)
  ├─ Path A (Trend-LSTM):      IMF1+IMF2 → LSTM(hidden=100, layers=2) → FC(50) → FC(24)
  ├─ Path B (Fluctuation-LSTM): IMF3+IMF4 → LSTM(hidden=128, layers=2) → FC(50) → FC(24)
  └─ Sum → clamp[0, capacity] → final 24h power prediction
"""

import torch
import torch.nn as nn


class VMDLSTMHybrid(nn.Module):
    """VMD-guided dual-path LSTM for multi-step wind power forecasting.

    Parameters
    ----------
    n_imfs        : number of IMF channels in input (default 4)
    output_dim    : forecast horizon (default 24)
    trend_hidden  : hidden size for Trend-LSTM path (default 100)
    fluct_hidden  : hidden size for Fluctuation-LSTM path (default 128)
    n_layers      : LSTM layers (default 2)
    dropout       : dropout rate (default 0.3)
    capacity      : max power (kW) for output clamping (default 2000.0)
    """

    def __init__(self, n_imfs=4, output_dim=24, trend_hidden=100, fluct_hidden=128,
                 n_layers=2, dropout=0.3, capacity=2000.0):
        super().__init__()
        self.output_dim = output_dim
        self.capacity = capacity

        # ── Path A: Trend-LSTM (IMF 1-2) ──
        self.trend_lstm = nn.LSTM(
            input_size=2, hidden_size=trend_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.trend_fc1 = nn.Linear(trend_hidden, 50)
        self.trend_drop = nn.Dropout(dropout)
        self.trend_fc2 = nn.Linear(50, output_dim)

        # ── Path B: Fluctuation-LSTM (IMF 3-4) ──
        self.fluct_lstm = nn.LSTM(
            input_size=2, hidden_size=fluct_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fluct_fc1 = nn.Linear(fluct_hidden, 50)
        self.fluct_drop = nn.Dropout(dropout)
        self.fluct_fc2 = nn.Linear(50, output_dim)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (batch, seq_len, 4)
            IMF channels 1-4 stacked.

        Returns
        -------
        out : (batch, output_dim)
            Predicted power at horizons 1..output_dim.
        """
        # split into low- / high-frequency groups
        imf_low = x[:, :, :2]    # (B, T, 2)   IMF1+IMF2
        imf_high = x[:, :, 2:4]  # (B, T, 2)   IMF3+IMF4

        # ── Trend path ──
        t_out, _ = self.trend_lstm(imf_low)     # (B, T, trend_hidden)
        t_out = t_out[:, -1, :]                  # last-step hidden
        t_out = torch.relu(self.trend_fc1(t_out))
        t_out = self.trend_drop(t_out)
        t_out = self.trend_fc2(t_out)            # (B, output_dim)

        # ── Fluctuation path ──
        f_out, _ = self.fluct_lstm(imf_high)    # (B, T, fluct_hidden)
        f_out = f_out[:, -1, :]
        f_out = torch.relu(self.fluct_fc1(f_out))
        f_out = self.fluct_drop(f_out)
        f_out = self.fluct_fc2(f_out)            # (B, output_dim)

        out = t_out + f_out
        return out
