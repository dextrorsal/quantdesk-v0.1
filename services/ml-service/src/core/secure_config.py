"""
Secure Configuration Management
Handles environment variables and secrets securely
"""
import os
import secrets
import string
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration management with environment variables"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize secure configuration
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        self.env_file = env_file or self._find_env_file()
        self._load_environment()
        self._validate_required_vars()
    
    def _find_env_file(self) -> str:
        """Find the .env file in the project root"""
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
        
        if not env_file.exists():
            logger.warning(f"No .env file found at {env_file}")
            logger.info("Please copy env.example to .env and configure your secrets")
        
        return str(env_file)
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        else:
            logger.warning(f"Environment file {self.env_file} not found")
    
    def _validate_required_vars(self):
        """Validate that required environment variables are set"""
        required_vars = [
            "JWT_SECRET_KEY",
            "WS_TOKEN",
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.info("Please set these variables in your .env file")
    
    @staticmethod
    def generate_secret_key(length: int = 64) -> str:
        """Generate a cryptographically secure secret key"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def get_exchange_config(self, exchange: str) -> Dict[str, str]:
        """Get exchange configuration securely"""
        exchange_upper = exchange.upper()
        
        config = {
            "api_key": os.getenv(f"{exchange_upper}_API_KEY"),
            "secret_key": os.getenv(f"{exchange_upper}_SECRET_KEY"),
        }
        
        # Add exchange-specific fields
        if exchange_upper == "BITGET":
            config["passphrase"] = os.getenv("BITGET_PASSPHRASE")
        
        # Validate that all required fields are present
        missing_fields = [k for k, v in config.items() if not v]
        if missing_fields:
            logger.warning(f"Missing {exchange} configuration: {missing_fields}")
        
        return config
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration securely"""
        return {
            "supabase_url": os.getenv("SUPABASE_URL"),
            "supabase_key": os.getenv("SUPABASE_KEY"),
            "supabase_service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            "neon_connection_string": os.getenv("NEON_CONNECTION_STRING"),
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "jwt_secret_key": os.getenv("JWT_SECRET_KEY"),
            "ws_token": os.getenv("WS_TOKEN"),
            "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return {
            "default_stake_amount": float(os.getenv("DEFAULT_STAKE_AMOUNT", "1000")),
            "max_open_trades": int(os.getenv("MAX_OPEN_TRADES", "3")),
            "default_timeframe": os.getenv("DEFAULT_TIMEFRAME", "5m"),
            "max_drawdown_percent": float(os.getenv("MAX_DRAWDOWN_PERCENT", "10")),
            "stop_loss_percent": float(os.getenv("STOP_LOSS_PERCENT", "5")),
            "take_profit_percent": float(os.getenv("TAKE_PROFIT_PERCENT", "10")),
        }
    
    def get_external_services_config(self) -> Dict[str, str]:
        """Get external services configuration"""
        return {
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
            "sonar_token": os.getenv("SONAR_TOKEN"),
            "sonar_host_url": os.getenv("SONAR_HOST_URL"),
            "github_token": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"

# Global configuration instance
config = SecureConfig()

# Convenience functions
def get_exchange_config(exchange: str) -> Dict[str, str]:
    """Get exchange configuration"""
    return config.get_exchange_config(exchange)

def get_database_config() -> Dict[str, str]:
    """Get database configuration"""
    return config.get_database_config()

def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    return config.get_security_config()

def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration"""
    return config.get_trading_config()

def get_external_services_config() -> Dict[str, str]:
    """Get external services configuration"""
    return config.get_external_services_config()

def generate_secret_key(length: int = 64) -> str:
    """Generate a cryptographically secure secret key"""
    return SecureConfig.generate_secret_key(length)
