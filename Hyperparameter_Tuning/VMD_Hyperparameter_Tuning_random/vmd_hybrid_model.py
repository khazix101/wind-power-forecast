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
    """CNN-LSTM (short term) + VMD-LSTM (long term) hybrid.

    Parameters
    ----------
    weather_dim      : number of weather/raw features in input (default 8)
    n_imfs           : number of IMF channels in input (default 4)
    cnn_out          : first N hours predicted by CNN-LSTM path (default 12)
    output_dim       : total forecast horizon (default 24)
    trend_hidden     : hidden size for Trend-LSTM path (default 100)
    fluct_hidden     : hidden size for Fluctuation-LSTM path (default 128)
    n_layers         : LSTM layers for VMD paths (default 2)
    dropout          : dropout rate for VMD paths (default 0.3)
    fc_hidden        : FC hidden layer size for Path B (default 50)
    path_b_dropout   : dropout for Path B (None → uses dropout) (default None)
    capacity         : max power (kW) for output clamping (default 2000.0)
    conv1_filters    : Conv1D layer 1 output channels (default 64)
    conv2_filters    : Conv1D layer 2 output channels (default 128)
    conv_kernel      : Conv1D kernel size (default 3)
    pool_size        : MaxPool1d kernel size (default 2)
    cnn_lstm_hidden  : hidden size for CNN-LSTM path (default 50)
    cnn_lstm_layers  : LSTM layers for CNN path (default 1)
    path_a_dropout   : dropout rate for CNN path FC (default 0.3)
    """

    def __init__(self, weather_dim=8, n_imfs=4, cnn_out=12, output_dim=24,
                 trend_hidden=100, fluct_hidden=128, n_layers=2,
                 dropout=0.3, fc_hidden=50, path_b_dropout=None,
                 capacity=2000.0,
                 conv1_filters=64, conv2_filters=128, conv_kernel=3,
                 pool_size=2, cnn_lstm_hidden=50, cnn_lstm_layers=1,
                 path_a_dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.cnn_out = cnn_out
        self.vmd_out = output_dim - cnn_out
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
        self.cnn_fc = nn.Linear(cnn_lstm_hidden, cnn_out)

        # ── Path B: Trend-LSTM (IMF 1-2) ──
        self.trend_lstm = nn.LSTM(
            input_size=2, hidden_size=trend_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=pb_drop if n_layers > 1 else 0.0,
        )
        self.trend_fc1 = nn.Linear(trend_hidden, fc_hidden)
        self.trend_drop = nn.Dropout(pb_drop)
        self.trend_fc2 = nn.Linear(fc_hidden, self.vmd_out)

        # ── Path B: Fluctuation-LSTM (IMF 3-4) ──
        self.fluct_lstm = nn.LSTM(
            input_size=2, hidden_size=fluct_hidden,
            num_layers=n_layers, batch_first=True,
            dropout=pb_drop if n_layers > 1 else 0.0,
        )
        self.fluct_fc1 = nn.Linear(fluct_hidden, fc_hidden)
        self.fluct_drop = nn.Dropout(pb_drop)
        self.fluct_fc2 = nn.Linear(fc_hidden, self.vmd_out)

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
        a = x_weather.permute(0, 2, 1)          # (B, weather_dim, T) for Conv1D
        a = torch.relu(self.conv1(a))
        a = torch.relu(self.conv2(a))
        a = self.pool(a)
        a = a.permute(0, 2, 1)
        a_out, _ = self.cnn_lstm(a)
        a_out = a_out[:, -1, :]
        a_out = self.cnn_drop(a_out)
        a_out = self.cnn_fc(a_out)

        # ── Path B: Trend-LSTM (IMF 1-2) + Fluctuation-LSTM (IMF 3-4) ──
        imf_low  = x_imfs[:, :, :2]    # (B, T, 2)
        imf_high = x_imfs[:, :, 2:4]   # (B, T, 2)

        t_out, _ = self.trend_lstm(imf_low)     # (B, T, trend_hidden)
        t_out = t_out[:, -1, :]
        t_out = torch.relu(self.trend_fc1(t_out))
        t_out = self.trend_drop(t_out)
        t_out = self.trend_fc2(t_out)            # (B, vmd_out)

        f_out, _ = self.fluct_lstm(imf_high)    # (B, T, fluct_hidden)
        f_out = f_out[:, -1, :]
        f_out = torch.relu(self.fluct_fc1(f_out))
        f_out = self.fluct_drop(f_out)
        f_out = self.fluct_fc2(f_out)            # (B, vmd_out)

        b_out = t_out + f_out                    # (B, vmd_out)

        # ── Concatenate both paths ──
        out = torch.cat([a_out, b_out], dim=1)   # (B, cnn_out + vmd_out)
        return out
