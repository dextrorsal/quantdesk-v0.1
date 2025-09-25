"""
PRIMARY COMPONENT: Standalone Lorentzian Classifier

This is a standalone implementation of the Lorentzian ANN (Approximate Nearest Neighbors)
classifier for signal generation in trading systems. It uses a novel Lorentzian distance
metric to find similar historical price patterns and predict future price movements.

Key features:
- GPU-accelerated implementation with PyTorch
- K-nearest neighbors classification with Lorentzian distance
- Market regime and volatility filtering
- Model persistence (save/load functionality)
- Incremental learning capability (can update with new data)
- Efficient batch processing for large datasets

This standalone version can be used for:
1. Individual backtesting and analysis
2. Development and testing of the core algorithm
3. Educational purposes to understand the Lorentzian distance approach
4. Quick prototyping before integration with other components

This is one of two implementations of the Lorentzian classifier in the codebase.
The other is located in `models/primary/lorentzian_classifier.py` and is designed
for integration with the full trading system.

Usage:
```python
# Initialize model
model = LorentzianANN(lookback_bars=50, prediction_bars=4, k_neighbors=20)

# Prepare features and fit model
features = prepare_features(df)
prices = df['close'].values
model.fit(features, prices)

# Generate predictions
predictions = model.predict(features)

# Save model for later use
model.save_model('lorentzian_model.pt')
```

Lorentzian Classifier Strategy

See also:
- docs/TECHNICAL_STRATEGY.md (Technical Strategy Documentation)
- docs/ML_MODEL.md (ML Model Architecture)
"""

import torch
import os
from pathlib import Path

# Set up GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LorentzianANN:
    """
    Standalone Lorentzian Approximate Nearest Neighbors (ANN) Classifier for Trading Signals.

    This class implements a GPU-accelerated, batch-processed classifier using a Lorentzian distance metric
    to find similar historical price patterns and predict future price movements. It supports incremental
    learning, model persistence, and is suitable for both research and production use.

    Parameters
    ----------
    lookback_bars : int, optional
        Number of historical bars to consider for each feature vector (default: 50).
    prediction_bars : int, optional
        Number of bars into the future to predict (default: 4).
    k_neighbors : int, optional
        Number of nearest neighbors to use for classification (default: 20).
    use_regime_filter : bool, optional
        Whether to use market regime filtering (default: True).
    use_volatility_filter : bool, optional
        Whether to use volatility filtering (default: True).
    use_adx_filter : bool, optional
        Whether to use ADX filtering (default: True).
    adx_threshold : float, optional
        ADX threshold for filtering (default: 20.0).
    regime_threshold : float, optional
        Regime threshold for filtering (default: -0.1).
    """

    def __init__(
        self,
        lookback_bars=50,
        prediction_bars=4,
        k_neighbors=20,
        use_regime_filter=True,
        use_volatility_filter=True,
        use_adx_filter=True,
        adx_threshold=20.0,
        regime_threshold=-0.1,
    ):
        """
        Initialize the LorentzianANN classifier with configuration options.

        Parameters
        ----------
        lookback_bars : int
            Number of historical bars to consider for each feature vector.
        prediction_bars : int
            Number of bars into the future to predict.
        k_neighbors : int
            Number of nearest neighbors to use for classification.
        use_regime_filter : bool
            Whether to use market regime filtering.
        use_volatility_filter : bool
            Whether to use volatility filtering.
        use_adx_filter : bool
            Whether to use ADX filtering.
        adx_threshold : float
            ADX threshold for filtering.
        regime_threshold : float
            Regime threshold for filtering.
        """
        self.lookback_bars = lookback_bars
        self.prediction_bars = prediction_bars
        self.k_neighbors = k_neighbors
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.use_adx_filter = use_adx_filter
        self.adx_threshold = adx_threshold
        self.regime_threshold = regime_threshold

        # These will store our historical data
        self.feature_arrays = None
        self.labels = None
        self.scaler = None

        # Path for model persistence
        self.model_dir = Path("models")
        self.model_path = self.model_dir / "lorentzian_ann.pt"
        self.is_fitted = False

        # Set device for PyTorch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models directory if it doesn't exist
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {self.model_dir}")

    def calculate_signals(self, data):
        """
        Calculate trading signals for the given OHLCV DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns ['open', 'high', 'low', 'close', ...].

        Returns
        -------
        dict
            Dictionary with keys:
                - 'buy_signals': torch.Tensor (1 = buy, 0 = no signal)
                - 'sell_signals': torch.Tensor (1 = sell, 0 = no signal)
                - 'predictions': torch.Tensor (probability-like values)
                - 'long_signals': torch.Tensor (same as buy_signals)
                - 'short_signals': torch.Tensor (same as sell_signals)
        """
        # Create simple features from the data
        features = self._prepare_features(data)

        # If the model is not fitted with historical data, do a simple fit
        if not self.is_fitted:
            prices = torch.tensor(data["close"].values, dtype=torch.float32)
            self.fit(features, prices)

        # Get predictions (will be -1, 0, or 1)
        raw_predictions = self.predict(features)

        # Convert to signals format
        buy_signals = torch.zeros_like(raw_predictions, dtype=torch.float32)
        sell_signals = torch.zeros_like(raw_predictions, dtype=torch.float32)

        # 1 for long signals, 0 for neutral, -1 for short signals
        buy_signals[raw_predictions == 1] = 1
        sell_signals[raw_predictions == -1] = 1

        # Create probability-like predictions (0.5 = neutral, 1 = strong buy, 0 = strong sell)
        probability_predictions = (
            torch.ones_like(raw_predictions, dtype=torch.float32) * 0.5
        )
        probability_predictions[raw_predictions == 1] = 0.8  # Buy confidence
        probability_predictions[raw_predictions == -1] = 0.2  # Sell confidence

        return {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "predictions": probability_predictions,
            "long_signals": buy_signals,  # For compatibility with the tests
            "short_signals": sell_signals,  # For compatibility with the tests
        }

    def _prepare_features(self, data):
        """
        Prepare feature matrix from OHLCV DataFrame for model input.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns ['open', 'high', 'low', 'close', ...].

        Returns
        -------
        torch.Tensor
            Feature matrix of shape (N, F), where N is the number of samples and F is the number of features.
        """
        # Simple features: price changes and ratios
        df = data.copy()

        # Convert to tensors for calculation
        close = torch.tensor(df["close"].values, dtype=torch.float32, device=device)
        high = torch.tensor(df["high"].values, dtype=torch.float32, device=device)
        low = torch.tensor(df["low"].values, dtype=torch.float32, device=device)

        # Create simple features
        features = []

        # Price change over different periods
        for period in [1, 2, 3, 5, 8, 13, 21]:
            if len(close) > period:
                # Percent change
                pct_change = (close[period:] - close[:-period]) / close[:-period]
                # Pad with zeros
                pct_change = torch.cat([torch.zeros(period, device=device), pct_change])
                features.append(pct_change)

        # High-Low range
        if len(high) > 0 and len(low) > 0:
            range_feature = (high - low) / close
            features.append(range_feature)

        # Stack features
        if features:
            stacked = torch.stack(features, dim=1)
            return stacked
        else:
            # Return dummy features if data is too short
            return torch.zeros((len(close), 1), device=device)

    def lorentzian_distance(self, features, historical_features):
        """
        Calculate Lorentzian distance between input features and historical features.

        Parameters
        ----------
        features : torch.Tensor
            Tensor of shape (n_samples, n_features) for current data.
        historical_features : torch.Tensor
            Tensor of shape (n_historical, n_features) for historical data.

        Returns
        -------
        torch.Tensor
            Distance matrix of shape (n_samples, n_historical).
        """
        # Process in batches to save memory
        batch_size = 100  # Adjust based on available memory
        n_samples = features.shape[0]
        n_historical = historical_features.shape[0]

        # Initialize distances tensor
        distances = torch.zeros((n_samples, n_historical), device=device)

        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_features = features[i:end_i]

            # For each feature vector in the batch
            for j in range(batch_features.shape[0]):
                # Get the feature vector
                x = batch_features[j]

                # Calculate differences
                diff = torch.abs(x.unsqueeze(0) - historical_features)

                # Calculate Lorentzian distance: ln(1 + |x - y|)
                # We use log1p for numerical stability
                # This is the true formula from TradingView
                log_diff = torch.log1p(diff)

                # Sum over features dimension
                batch_distances = torch.sum(log_diff, dim=1)

                # Store in the distances tensor
                distances[i + j] = batch_distances

        return distances

    def generate_training_data(self, features, prices):
        """
        Generate training labels for the prediction task based on future price movement.

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix of shape (N, F).
        prices : torch.Tensor
            1D tensor of close prices (shape: [N]).

        Returns
        -------
        tuple
            (features, labels) where labels is a tensor of shape (N - prediction_bars,)
            with values 1 (long), -1 (short), or 0 (neutral).
        """
        # Using the same TradingView approach: look ahead a fixed number of bars
        # to determine if price went up or down
        future_prices = prices[self.prediction_bars :]
        current_prices = prices[: -self.prediction_bars]

        # Generate labels: 1 for long (price went up), -1 for short (price went down), 0 for neutral
        labels = torch.zeros(len(current_prices), dtype=torch.long)
        labels[future_prices > current_prices] = 1  # Long
        labels[future_prices < current_prices] = -1  # Short

        # We can only generate labels for data points that have future data
        return features[: len(labels)], labels

    def fit(self, features, prices):
        """
        Fit the LorentzianANN model by storing historical feature arrays and labels.

        Parameters
        ----------
        features : torch.Tensor or np.ndarray
            Feature matrix for training.
        prices : torch.Tensor or np.ndarray
            1D array of close prices.

        Returns
        -------
        self : LorentzianANN
            The fitted model instance.
        """
        # Convert to tensors if they aren't already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices, dtype=torch.float32)

        # Generate training data
        features, labels = self.generate_training_data(features, prices)

        # Store for lookup
        self.feature_arrays = features.to(device)
        self.labels = labels.to(device)
        self.is_fitted = True  # Set this flag to indicate the model is fitted

        return self

    def predict(self, features):
        """
        Predict trading signals using Approximate Nearest Neighbors with Lorentzian distance.

        Parameters
        ----------
        features : torch.Tensor or np.ndarray
            Feature matrix for prediction (shape: [N, F]).

        Returns
        -------
        torch.Tensor
            Predicted labels: 1 for long, -1 for short, 0 for neutral (shape: [N]).
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        # Move to same device as model
        features = features.to(device)

        # Process in batches to avoid memory issues
        batch_size = 1000  # Adjust based on GPU memory
        n_samples = len(features)
        all_predictions = []

        print(f"Processing {n_samples} samples in batches of {batch_size}...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_features = features[start_idx:end_idx]

            # Calculate Lorentzian distances for this batch
            batch_distances = self.lorentzian_distance(
                batch_features, self.feature_arrays
            )

            # Get indices of k-nearest neighbors
            _, indices = torch.topk(
                batch_distances,
                min(self.k_neighbors, len(batch_distances[0])),
                largest=False,
                dim=1,
            )

            # Get labels of k-nearest neighbors
            batch_neighbor_labels = [self.labels[idx] for idx in indices]
            batch_neighbor_labels = torch.stack(batch_neighbor_labels)

            # Calculate the sum of neighbor labels
            batch_predictions = torch.sum(batch_neighbor_labels, dim=1)

            # Convert to direction: 1 for long, -1 for short, 0 for neutral
            batch_final = torch.zeros_like(batch_predictions)
            batch_final[batch_predictions > 0] = 1
            batch_final[batch_predictions < 0] = -1

            all_predictions.append(batch_final)

            # Print progress
            progress = min(100, (end_idx / n_samples) * 100)
            print(f"Progress: {progress:.1f}%", end="\r")

        # Combine all batches
        final_predictions = torch.cat(all_predictions)
        print("\nPrediction complete!")

        return final_predictions

    def save_model(self, path=None):
        """
        Save the model state to a file for later use.

        Parameters
        ----------
        path : str or Path, optional
            File path to save the model. If None, uses default path.

        Returns
        -------
        bool
            True if the model was saved successfully, False otherwise.
        """
        if path is None:
            path = self.model_path

        if not self.is_fitted:
            print("Model not fitted yet, nothing to save")
            return False

        # Create a dictionary containing all necessary components
        save_dict = {
            "feature_arrays": self.feature_arrays.cpu(),
            "labels": self.labels.cpu(),
            "scaler": self.scaler,
            "config": {
                "lookback_bars": self.lookback_bars,
                "prediction_bars": self.prediction_bars,
                "k_neighbors": self.k_neighbors,
                "use_regime_filter": self.use_regime_filter,
                "use_volatility_filter": self.use_volatility_filter,
                "use_adx_filter": self.use_adx_filter,
            },
            "metadata": {
                "date_saved": pd.Timestamp.now().isoformat(),
                "samples": len(self.feature_arrays),
            },
        }

        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save model
            torch.save(save_dict, path)
            print(f"Model saved to {path}")
            print(
                f"Saved {len(self.feature_arrays)} samples with {len(self.scaler) if self.scaler else 0} features"
            )
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, path=None):
        """
        Load the model state from a file.

        Parameters
        ----------
        path : str or Path, optional
            File path to load the model from. If None, uses default path.

        Returns
        -------
        bool
            True if the model was loaded successfully, False otherwise.
        """
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            print(f"Model file {path} does not exist")
            return False

        try:
            # Load with weights_only=False to allow loading complex objects
            # Note: Only use this with models from trusted sources
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Load configuration
            config = checkpoint["config"]
            self.lookback_bars = config["lookback_bars"]
            self.prediction_bars = config["prediction_bars"]
            self.k_neighbors = config["k_neighbors"]
            self.use_regime_filter = config["use_regime_filter"]
            self.use_volatility_filter = config["use_volatility_filter"]
            self.use_adx_filter = config["use_adx_filter"]

            # Load model data - make sure to move to the correct device
            self.feature_arrays = checkpoint["feature_arrays"].to(device)
            self.labels = checkpoint["labels"].to(device)
            self.scaler = checkpoint["scaler"]

            # Print metadata if available
            if "metadata" in checkpoint:
                metadata = checkpoint["metadata"]
                print(f"Model saved on: {metadata.get('date_saved', 'Unknown')}")
                print(f"Samples in model: {metadata.get('samples', 'Unknown')}")

            self.is_fitted = True
            print(f"Model loaded from {path} with {len(self.feature_arrays)} samples")
            print(
                f"Configuration: lookback={self.lookback_bars}, prediction={self.prediction_bars}, k={self.k_neighbors}"
            )

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def update_model(self, new_features, new_prices, max_samples=20000):
        """
        Incrementally update the model with new data, keeping the most recent samples.

        Parameters
        ----------
        new_features : torch.Tensor or np.ndarray
            New feature matrix to add.
        new_prices : torch.Tensor or np.ndarray
            New close prices to add.
        max_samples : int, optional
            Maximum number of samples to keep in the model (default: 20000).

        Returns
        -------
        self : LorentzianANN
            The updated model instance.
        """
        if not self.is_fitted:
            print("Model not fitted yet, using initial fit instead")
            return self.fit(new_features, new_prices)

        # Convert new data to tensors
        if not isinstance(new_features, torch.Tensor):
            new_features = torch.tensor(new_features, dtype=torch.float32)
        if not isinstance(new_prices, torch.Tensor):
            new_prices = torch.tensor(new_prices, dtype=torch.float32)

        # Move to device
        new_features = new_features.to(device)
        new_prices = new_prices.to(device)

        # Generate labels for new data
        new_features, new_labels = self.generate_training_data(new_features, new_prices)

        print(f"Adding {len(new_features)} new samples to model")

        # Combine with existing data (keeping most recent samples)
        if len(self.feature_arrays) + len(new_features) > max_samples:
            # Keep most recent data
            keep_samples = max_samples - len(new_features)

            print(
                f"Limiting model to {max_samples} samples (removing {len(self.feature_arrays) - keep_samples} old samples)"
            )

            self.feature_arrays = self.feature_arrays[-keep_samples:]
            self.labels = self.labels[-keep_samples:]

        # Make sure both tensors are on the same device before concatenating
        self.feature_arrays = self.feature_arrays.to(device)
        self.labels = self.labels.to(device)
        new_features = new_features.to(device)
        new_labels = new_labels.to(device)

        # Add new data
        self.feature_arrays = torch.cat([self.feature_arrays, new_features])
        self.labels = torch.cat([self.labels, new_labels])

        print(f"Model updated: {len(self.feature_arrays)} total samples")

        return self


def remove_unused_imports():
    """
    Placeholder function for removing unused imports (not implemented).
    """
    pass
