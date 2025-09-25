# Exchange Authentication Module

This module provides authentication handlers for various cryptocurrency exchanges. It separates authentication logic from exchange handlers, allowing for cleaner code and better separation of concerns.

## Overview

The authentication module follows these design principles:

1. **Separation of Concerns**: Authentication is handled separately from exchange API calls
2. **Consistent Interface**: All exchange authentication handlers implement the same interface
3. **Flexibility**: Support for different authentication methods (API keys, wallet-based, etc.)
4. **Security**: Secure handling of credentials

## Authentication Handlers

### Base Authentication Handler

The `BaseAuth` class defines the interface that all exchange-specific authentication handlers must implement:

```python
class BaseAuth(ABC):
    def __init__(self, credentials: Optional[ExchangeCredentials] = None):
        # Initialize with credentials
        
    @abstractmethod
    def get_auth_headers(self, method: str, endpoint: str, 
                         params: Optional[Dict] = None, 
                         data: Optional[Dict] = None) -> Dict[str, str]:
        # Generate authentication headers
        
    @abstractmethod
    def is_authenticated(self) -> bool:
        # Check if authentication is valid
```

### Drift Authentication

The `DriftAuth` class provides authentication for the Drift exchange using Solana wallet integration:

```python
from src.core.models import ExchangeCredentials
from src.exchanges.auth import DriftAuth

# Create credentials with Solana private key
credentials = ExchangeCredentials(
    additional_params={
        "private_key": "your_base58_encoded_private_key",
        "program_id": "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH",  # Optional
        "rpc_url": "https://api.mainnet-beta.solana.com"  # Optional
    }
)

# Initialize authentication handler
auth_handler = DriftAuth(credentials)

# Check if authentication is valid
if auth_handler.is_authenticated():
    # Initialize Drift client
    client = await auth_handler.initialize_client()
    
    # Use the client for authenticated operations
    user_account = await client.get_user_account()
    
    # Close the client when done
    await auth_handler.close()
```

### Binance Authentication

The `BinanceAuth` class provides authentication for the Binance exchange using API key and secret:

```python
from src.core.models import ExchangeCredentials
from src.exchanges.auth import BinanceAuth

# Create credentials with API key and secret
credentials = ExchangeCredentials(
    api_key="your_api_key",
    api_secret="your_api_secret"
)

# Initialize authentication handler
auth_handler = BinanceAuth(credentials)

# Generate authentication headers for a request
headers = auth_handler.get_auth_headers(
    method="GET",
    endpoint="/api/v3/account",
    params={"timestamp": 1234567890}
)
```

### Coinbase Authentication

The `CoinbaseAuth` class provides authentication for the Coinbase exchange using API key and secret:

```python
from src.core.models import ExchangeCredentials
from src.exchanges.auth import CoinbaseAuth

# Create credentials with API key and secret
credentials = ExchangeCredentials(
    api_key="your_api_key",
    api_secret="your_api_secret"
)

# Initialize authentication handler
auth_handler = CoinbaseAuth(credentials)

# Generate authentication headers for a request
headers = auth_handler.get_auth_headers(
    method="GET",
    endpoint="/accounts",
    params={"limit": 100}
)
```

## Integration with Exchange Handlers

Exchange handlers can use authentication handlers to generate authentication headers for API requests:

```python
from src.exchanges.auth import DriftAuth

class DriftHandler(BaseExchangeHandler):
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        
        # Initialize authentication handler
        self._auth_handler = DriftAuth(config.credentials)
        
    async def _get_authenticated_client(self) -> DriftClient:
        """Get an authenticated Drift client for operations requiring authentication."""
        if not self._auth_handler.is_authenticated():
            raise ExchangeError("Authentication required for this operation")
            
        if not self.client:
            await self.start()
            
        return self.client
```

## Examples

See the `examples` directory for complete examples of how to use the authentication handlers:

- `drift_auth_example.py`: Example of using the Drift authentication handler
- `binance_auth_example.py`: Example of using the Binance authentication handler (coming soon)
- `coinbase_auth_example.py`: Example of using the Coinbase authentication handler (coming soon)