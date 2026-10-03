import torch
import torch.nn as nn

class CNNBiGRUClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNBiGRUClassifier, self).__init__()
        
        # 1. Spatial Feature Extractor (CNN Layers)
        # Input shape expected: [Batch, 1, 25, 13] (1 Channel, 25 Freq bins, 13 Time steps)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Condense frequency, preserve time sequence
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))   # Final downsample on spatial features
        )
        
        # After CNN processing, the frequency dimension 25 drops to 6 (25 / 2 / 2 = 6)
        # Spatial feature size per time step = 64 channels * 6 features = 384
        self.feature_dim = 64 * 6 
        
        # 2. Temporal Sequencer (Bidirectional GRU)
        # Processes sequences from both directions (past-to-future and future-to-past)
        self.bigru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 3. Dense Classification Output Head
        # Bidirectional outputs double the hidden size dimension (128 * 2 = 256)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, 1, 25, 13]
        batch_size, channels, freq_bins, time_steps = x.size()
        
        # Extract visual structures
        cnn_features = self.cnn(x) # Shape: [Batch, 64, 6, 13]
        
        # Reshape to present sequential tokens to the GRU over time
        # Desired GRU Shape: [Batch, Sequence_Length (Time), Features]
        cnn_features = cnn_features.permute(0, 3, 1, 2) # [Batch, 13, 64, 6]
        rnn_input = cnn_features.reshape(batch_size, time_steps, self.feature_dim) # [Batch, 13, 384]
        
        # Run through temporal network
        rnn_output, _ = self.bigru(rnn_input) # Shape: [Batch, 13, 256]
        
        # Pull the absolute final hidden context output step (many-to-one mapping)
        last_time_step = rnn_output[:, -1, :] # Shape: [Batch, 256]
        
        # Classify activity
        logits = self.fc(last_time_step)
        return logits

if __name__ == "__main__":
    # Quick sanity framework check
    mock_input = torch.randn(2, 1, 25, 13)
    model = CNNBiGRUClassifier(num_classes=5)
    out = model(mock_input)
    print("Architecture Check Successful! Output logit shape:", out.shape)