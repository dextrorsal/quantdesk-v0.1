"""
Core exceptions for the Ultimate Data Fetcher.
"""

class DataFetcherError(Exception):
    """Base exception for all data fetcher errors."""
    pass

class ConfigurationError(DataFetcherError):
    """Raised when there's an error in configuration."""
    pass

class ExchangeError(DataFetcherError):
    """Base class for exchange-related errors."""
    pass

class ValidationError(DataFetcherError):
    """Raised when data validation fails."""
    pass

class StorageError(DataFetcherError):
    """Raised when data storage operations fail."""
    pass

class RateLimitError(ExchangeError):
    """Raised when exchange rate limit is hit."""
    pass

class ApiError(ExchangeError):
    """Raised when an API request fails."""
    pass

class DataProcessingError(DataFetcherError):
    """Raised when data processing fails."""
    pass

class AuthError(DataFetcherError):
    """Raised when authentication fails."""
    pass

class CredentialError(AuthError):
    """Raised when there's an issue with credentials."""
    pass

class NotInitializedError(ExchangeError):
    """Raised when trying to use a component that hasn't been initialized."""
    pass