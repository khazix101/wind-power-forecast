"""VMD-LSTM + CNN-LSTM Hybrid model for 24h multi-step wind power forecasting.

Architecture:
  Input  → (batch, seq_len, weather_dim + n_imfs)
  ├─ Path A (CNN-LSTM, h=1~cnn_out): weather features → Conv1D×2 → MaxPool → LSTM → FC(cnn_out)
  ├─ Path B (VMD-LSTM, h=cnn_out+1~24): IMFs → Trend-LSTM + Fluctuation-LSTM → Sum → FC(vmd_out)
  └─ Concat → clamp[0, capacity] → final 24h power prediction
"""

import torch
import torch.nn as nn


class VMDLSTMHybrid(nn.Module):
    """CNN-LSTM (weather) + VMD-LSTM (IMFs) hybrid with learned fusion.

    Both paths output the full 24h horizon; a learned gate fuses them
    sample-by-sample and hour-by-hour.
    """

    def __init__(self, weather_dim=8, n_imfs=4, output_dim=24,
                 trend_hidden=128, fluct_hidden=64, n_layers=1,
                 dropout=0.3, fc_hidden=64, path_b_dropout=None,
                 capacity=2000.0,
                 conv1_filters=64, conv2_filters=128, conv_kernel=3,
                 pool_size=2, cnn_lstm_hidden=32, cnn_lstm_layers=1,
                 path_a_dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.capacity = capacity
        pb_drop = dropout if path_b_dropout is None else path_b_dropout

        # ── Path A: CNN-LSTM (short horizon, raw + weather features) ──
        conv_pad = conv_kernel // 2
        self.conv1 = nn.Conv1d(in_channels=weather_dim, out_channels=conv1_filters,
                               kernel_size=conv_kernel, padding=conv_pad)
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                               kernel_size=conv_kernel, padding=conv_pad)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.cnn_lstm = nn.LSTM(
            input_size=conv2_filters, hidden_size=cnn_lstm_hidden,
            num_layers=cnn_lstm_layers, batch_first=True,
            dropout=path_a_dropout if cnn_lstm_layers > 1 else 0.0,
        )
        self.cnn_drop = nn.Dropout(path_a_dropout)
        self.cnn_fc = nn.Linear(cnn_lstm_hidden, output_dim)

        # ── Path B: Trend-LSTM (IMF 1-2) ──
        self.trend_lstm = nn.LSTM(
            input_size=2, hidden_size=trend_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=pb_drop if n_layers > 1 else 0.0,
        )
        self.trend_fc1 = nn.Linear(trend_hidden, fc_hidden)
        self.trend_drop = nn.Dropout(pb_drop)
        self.trend_fc2 = nn.Linear(fc_hidden, output_dim)

        # ── Path B: Fluctuation-LSTM (IMF 3-4) ──
        self.fluct_lstm = nn.LSTM(
            input_size=2, hidden_size=fluct_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=pb_drop if n_layers > 1 else 0.0,
        )
        self.fluct_fc1 = nn.Linear(fluct_hidden, fc_hidden)
        self.fluct_drop = nn.Dropout(pb_drop)
        self.fluct_fc2 = nn.Linear(fc_hidden, output_dim)

        self.gate_fc = nn.Linear(cnn_lstm_hidden + trend_hidden + fluct_hidden, output_dim)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (batch, seq_len, weather_dim + n_imfs)
            First weather_dim cols = weather features (Path A).
            Last n_imfs cols = IMF channels (Path B).

        Returns
        -------
        out : (batch, output_dim)
            Predicted power at horizons 1..output_dim.
        """
        # split input: weather features (Path A) and IMFs (Path B)
        weather_dim = x.size(2) - 4  # infer from input (total - 4 IMFs)
        x_weather = x[:, :, :weather_dim]      # (B, T, weather_dim)
        x_imfs    = x[:, :, weather_dim:]       # (B, T, 4)

        # ── Path A: CNN-LSTM ──
        a = x_weather.permute(0, 2, 1)          # (B, weather_dim, T)
        a = torch.relu(self.conv1(a))
        a = torch.relu(self.conv2(a))
        a = self.pool(a)
        a = a.permute(0, 2, 1)
        a_out, _ = self.cnn_lstm(a)
        a_hidden = a_out[:, -1, :]               # (B, cnn_lstm_hidden)
        a_pred = self.cnn_fc(self.cnn_drop(a_hidden))  # (B, output_dim)

        # ── Path B: Trend-LSTM (IMF 1-2) + Fluctuation-LSTM (IMF 3-4) ──
        imf_low  = x_imfs[:, :, :2]
        imf_high = x_imfs[:, :, 2:4]

        t_out, _ = self.trend_lstm(imf_low)
        t_hidden = t_out[:, -1, :]               # (B, trend_hidden)
        t_pred = self.trend_fc2(self.trend_drop(torch.relu(self.trend_fc1(t_hidden))))

        f_out, _ = self.fluct_lstm(imf_high)
        f_hidden = f_out[:, -1, :]               # (B, fluct_hidden)
        f_pred = self.fluct_fc2(self.fluct_drop(torch.relu(self.fluct_fc1(f_hidden))))

        b_pred = t_pred + f_pred                  # (B, output_dim)

        # ── Learned fusion gate ──
        gate = torch.sigmoid(self.gate_fc(torch.cat([a_hidden, t_hidden, f_hidden], dim=-1)))
        out = gate * a_pred + (1 - gate) * b_pred
        return out
