"""
Base authentication class for exchange API authentication.
Provides a common interface for all exchange-specific authentication implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from ...core.models import ExchangeCredentials


class BaseAuth(ABC):
    """
    Abstract base class for exchange authentication.
    
    This class defines the interface that all exchange-specific authentication
    implementations must follow. It provides methods for generating authentication
    headers and signatures required for API requests.
    """
    
    def __init__(self, credentials: Optional[ExchangeCredentials] = None):
        """
        Initialize the authentication handler.
        
        Args:
            credentials: Exchange API credentials
        """
        self.credentials = credentials
    
    @abstractmethod
    def get_auth_headers(self, method: str, endpoint: str, 
                         params: Optional[Dict] = None, 
                         data: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate authentication headers for an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            Dictionary of authentication headers
        """
        pass
    
    @abstractmethod
    def is_authenticated(self) -> bool:
        """
        Check if the authentication handler has valid credentials.
        
        Returns:
            True if authenticated, False otherwise
        """
        pass
    
    def update_credentials(self, credentials: ExchangeCredentials) -> None:
        """
        Update the authentication credentials.
        
        Args:
            credentials: New exchange API credentials
        """
        self.credentials = credentials