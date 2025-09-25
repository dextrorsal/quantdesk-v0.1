import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int = 20  # Number of features
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 30  # Lookback period

class TradingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM for sequential data
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output layers
        self.direction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 3)  # 3 classes: Long, Short, Neutral
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()  # Confidence between 0 and 1
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
        self.to(self.device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_layers(x)  # Shape: (batch_size, sequence_length, hidden_size)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, sequence_length, hidden_size)
        
        # Self-attention
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output for prediction
        final_features = attention_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Get predictions
        direction_logits = self.direction_head(final_features)  # Shape: (batch_size, 3)
        confidence = self.confidence_head(final_features)  # Shape: (batch_size, 1)
        
        return direction_logits, confidence
    
    def prepare_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare features from DataFrame"""
        # Technical indicators (using ta-lib via pandas_ta)
        df['rsi'] = df['close'].rolling(window=14).apply(lambda x: talib.RSI(x.values)[-1])
        df['atr'] = df.ta.atr(length=14)
        df['supertrend'] = df.ta.supertrend(length=10, multiplier=3.0)['SUPERT_10_3.0']
        df['ema_fast'] = df.ta.ema(length=8)
        df['ema_slow'] = df.ta.ema(length=21)
        df['macd'] = df.ta.macd(fast=12, slow=26)['MACD_12_26_9']
        df['macd_signal'] = df.ta.macd(fast=12, slow=26)['MACDs_12_26_9']
        df['macd_hist'] = df.ta.macd(fast=12, slow=26)['MACDh_12_26_9']
        df['bbands_upper'] = df.ta.bbands(length=20)['BBU_20_2.0']
        df['bbands_lower'] = df.ta.bbands(length=20)['BBL_20_2.0']
        
        # Price changes and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['mom'] = df.ta.mom(length=10)
        df['roc'] = df.ta.roc(length=10)
        
        # Trend strength
        df['adx'] = df.ta.adx(length=14)['ADX_14']
        df['dmi_plus'] = df.ta.adx(length=14)['DMP_14']
        df['dmi_minus'] = df.ta.adx(length=14)['DMN_14']
        
        # Select and normalize features
        feature_columns = [
            'rsi', 'atr', 'supertrend', 'ema_fast', 'ema_slow',
            'macd', 'macd_signal', 'macd_hist', 'bbands_upper', 'bbands_lower',
            'returns', 'volatility', 'volume_ratio', 'mom', 'roc',
            'adx', 'dmi_plus', 'dmi_minus'
        ]
        
        # Create sequences
        sequences = []
        for i in range(len(df) - self.config.sequence_length + 1):
            sequence = df[feature_columns].iloc[i:i+self.config.sequence_length].values
            sequences.append(sequence)
        
        # Convert to tensor and normalize
        features = torch.tensor(sequences, dtype=torch.float32)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features.to(self.device)
    
    def predict_next(self, features: torch.Tensor) -> Tuple[str, float]:
        """Predict next movement"""
        self.eval()
        with torch.no_grad():
            direction_logits, confidence = self(features.unsqueeze(0))
            direction_probs = F.softmax(direction_logits, dim=1)
            direction_idx = direction_probs.argmax(dim=1).item()
            confidence_val = confidence.item()
            
            directions = ['LONG', 'SHORT', 'NEUTRAL']
            return directions[direction_idx], confidence_val 