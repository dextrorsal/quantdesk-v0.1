"""
Neon Batch Loader - Loads batches of data efficiently for ML training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import logging

class TradingDataset(Dataset):
    """Dataset for trading data"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize dataset
        
        Args:
            X: Feature array
            y: Target array
            device: Device to load data to
        """
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
        
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        return self.X[idx], self.y[idx]

class NeonBatchLoader:
    """Loads batches of data from Neon database"""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize batch loader
        
        Args:
            batch_size: Number of samples per batch
            num_workers: Number of worker processes
            device: Device to load data to
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and testing dataloaders
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        try:
            # Create datasets
            train_dataset = TradingDataset(X_train, y_train, self.device)
            test_dataset = TradingDataset(X_test, y_test, self.device)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle_train,
                num_workers=self.num_workers,
                pin_memory=True if self.device == 'cuda' else False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True if self.device == 'cuda' else False
            )
            
            return train_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error creating dataloaders: {str(e)}")
            raise
            
    def get_batch_info(self, dataloader: DataLoader) -> dict:
        """
        Get information about batches
        
        Args:
            dataloader: DataLoader to get info about
            
        Returns:
            Dictionary with batch information
        """
        try:
            return {
                'num_batches': len(dataloader),
                'batch_size': self.batch_size,
                'total_samples': len(dataloader.dataset),
                'feature_shape': next(iter(dataloader))[0].shape[1:],
                'device': self.device
            }
            
        except Exception as e:
            self.logger.error(f"Error getting batch info: {str(e)}")
            raise 