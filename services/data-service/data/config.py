# src/core/config.py
"""
Configuration management for the Ultimate Data Fetcher.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import json
import configparser

from .exceptions import ConfigurationError
from .models import ExchangeCredentials


@dataclass
class StorageConfig:
    """Configuration for data storage."""

    # Base paths
    data_path: Path = Path("data")

    # Historical data paths
    historical_path: Path = None
    historical_raw_path: Path = None
    historical_processed_path: Path = None

    # Live data paths
    live_path: Path = None
    live_raw_path: Path = None
    live_processed_path: Path = None

    use_compression: bool = False
    backup_enabled: bool = False
    backup_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize paths that weren't explicitly set."""
        # Set up historical paths
        if self.historical_path is None:
            self.historical_path = self.data_path / "historical"

        if self.historical_raw_path is None:
            self.historical_raw_path = self.historical_path / "raw"

        if self.historical_processed_path is None:
            self.historical_processed_path = self.historical_path / "processed"

        # Set up live paths
        if self.live_path is None:
            self.live_path = self.data_path / "live"

        if self.live_raw_path is None:
            self.live_raw_path = self.live_path / "raw"

        if self.live_processed_path is None:
            self.live_processed_path = self.live_path / "processed"

        # Ensure all paths exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.historical_path.mkdir(parents=True, exist_ok=True)
        self.historical_raw_path.mkdir(parents=True, exist_ok=True)
        self.historical_processed_path.mkdir(parents=True, exist_ok=True)
        self.live_path.mkdir(parents=True, exist_ok=True)
        self.live_raw_path.mkdir(parents=True, exist_ok=True)
        self.live_processed_path.mkdir(parents=True, exist_ok=True)

        if self.backup_enabled and self.backup_path:
            self.backup_path.mkdir(parents=True, exist_ok=True)

    @property
    def tfrecord_path(self) -> Path:
        """Path to TFRecord data."""
        return self.historical_processed_path / "tfrecords"


@dataclass
class ExchangeConfig:
    """Configuration for an exchange."""

    name: str
    credentials: ExchangeCredentials
    rate_limit: int
    markets: List[str]
    base_url: str
    enabled: bool = True

    def lower(self) -> str:
        """Return the lowercase name of the exchange."""
        return self.name.lower()


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    log_level: str
    log_file: Path
    console_logging: bool = True
    file_logging: bool = True


class SolanaConfig:
    """Configuration for Solana blockchain connections."""

    def __init__(self, config_dict=None):
        """Initialize Solana configuration."""
        config_dict = config_dict or {}

        # Get the full RPC URL from environment variable or config
        self.rpc_url = os.environ.get(
            "HELIUS_RPC_ENDPOINT",
            config_dict.get("rpc_url", "https://api.mainnet-beta.solana.com"),
        )

        self.commitment = config_dict.get("commitment", "confirmed")
        self.timeout = config_dict.get("timeout", 30)


class Config:
    """Main configuration class."""

    def __init__(
        self,
        env_file: str = ".env",
        indicator_config_path: str = "config/indicator_settings.json",
        storage=None,
        exchanges=None,
        logging=None,
        config_path=None,
    ):
        """
        Initialize configuration using environment variables from a .env file and additional JSON settings.
        Optionally, you can pass pre-built config objects for storage, exchanges, and logging.
        """
        self.project_root = Path(__file__).parent.parent.parent
        # Load .env file
        self._config_data: Dict[str, Any] = {}
        self._load_env(env_file)
        # Load indicator config (from JSON) and merge into _config_data
        self._load_indicator_config(indicator_config_path)

        # Initialize sub-config objects if not passed in
        self.storage = storage or self._init_storage_config()
        self.exchanges = exchanges or self._init_exchange_configs()
        self.logging = logging or self._init_logging_config()

        # Add Solana configuration
        self.solana = SolanaConfig(config_dict=config_path)

        # --- Ensure Supabase config is always set from environment variables ---
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = {
            "url": supabase_url,
            "key": supabase_key,
            "enabled": bool(supabase_url and supabase_key),
        }
        print(
            "DEBUG: Supabase config loaded:",
            {k: (v if k != "key" else "REDACTED") for k, v in self.supabase.items()},
        )

    @classmethod
    def from_ini(cls, config_path: str = ".env"):
        """Load configuration from INI file."""
        config = configparser.ConfigParser()
        config.read(config_path)

        # Create config dictionary
        config_dict = {}

        # Add sections to config dictionary
        for section in config.sections():
            config_dict[section.lower()] = dict(config[section])

        # Add solana section if not present
        if "solana" not in config_dict:
            config_dict["solana"] = {}

        return cls(config_dict)

    def _load_env(self, env_file: str = ".env"):
        """Load environment variables from .env file using python-dotenv."""
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            print(
                f"Warning: {env_file} not found, using existing environment variables"
            )
        # Update internal config dictionary with current environment variables
        self._config_data.update(os.environ)

    def _load_indicator_config(
        self, config_path: str = "config/indicator_settings.json"
    ):
        """
        Load indicator settings from a JSON file and merge them into the configuration data.
        """
        config_file_path = Path(config_path)
        if config_file_path.exists():
            try:
                with open(config_file_path, "r") as f:
                    json_config = json.load(f)
                    self._config_data.update(json_config)
                    print(f"Debug: JSON config loaded successfully from: {config_path}")
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Error parsing JSON config file: {e}")
        else:
            print(f"Warning: {config_path} not found, using default settings.")

    def get(self, *possible_keys, default: Any = None) -> Any:
        """
        Retrieve a configuration value using nested keys.

        - If you call get('indicators', {}), then we interpret 'indicators'
        as the key and {} as the default value.
        - Otherwise, if you pass multiple string keys, we handle them as nested keys.
        """
        # 1. Handle the special case where the user calls get('something', <dict>)
        #    which we interpret as a single key + a default dict
        if len(possible_keys) == 2 and isinstance(possible_keys[1], dict):
            keys = (possible_keys[0],)
            default_value = possible_keys[1]
        else:
            # Normal usage: all arguments are string keys, or there's just one key
            keys = possible_keys
            default_value = default

        current_level = self._config_data

        try:
            for key in keys:
                # We must ensure key is a string (typical usage). If not, it's an error.
                if not isinstance(key, str):
                    raise TypeError(
                        f"Each key must be a string, but got {key!r} of type {type(key).__name__}."
                    )

                if isinstance(current_level, dict) and key in current_level:
                    current_level = current_level[key]

                    # REDACT sensitive fields named "secret" or "key" (case-insensitive)
                    if "secret" in key.lower() or "key" in key.lower():
                        masked_display = "REDACTED"
                    else:
                        masked_display = current_level

                    print(f"DEBUG: Requested {keys} → now at level: {masked_display}")
                else:
                    return default_value

            # If we end on a dictionary, return a copy to avoid accidental mutation
            if isinstance(current_level, dict):
                return current_level.copy()
            return current_level

        except TypeError as e:
            raise TypeError(
                f"Invalid access pattern for keys {keys}. "
                f"Ensure all intermediate values are dicts. Current level: {current_level}"
            ) from e

    def _init_storage_config(self) -> StorageConfig:
        """Initialize storage configuration."""
        try:
            data_path = Path(self.get("DATA_PATH", default="data"))

            # Adjust paths to use the new structure
            return StorageConfig(
                data_path=data_path,
                use_compression=self.get("USE_COMPRESSION", default="false").lower()
                == "true",
                backup_enabled=self.get("BACKUP_ENABLED", default="false").lower()
                == "true",
                backup_path=Path(self.get("BACKUP_PATH", default="data/backup"))
                if self.get("BACKUP_PATH")
                else None,
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize storage config: {e}")

    def _init_exchange_configs(self) -> Dict[str, ExchangeConfig]:
        """Initialize exchange configurations."""
        configs = {}

        if self.get("DRIFT_ENABLED", default="true").lower() == "true":
            configs["drift"] = ExchangeConfig(
                name="drift",
                credentials=ExchangeCredentials(
                    api_key=self.get("DRIFT_API_KEY"),
                    api_secret=self.get("DRIFT_API_SECRET"),
                ),
                rate_limit=int(self.get("DRIFT_RATE_LIMIT", default="10")),
                markets=self.get(
                    "DRIFT_MARKETS", default="SOL-PERP,BTC-PERP,ETH-PERP"
                ).split(","),
                base_url=self.get(
                    "DRIFT_BASE_URL",
                    default="https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH",
                ),
                enabled=True,
            )

        if self.get("BINANCE_ENABLED", default="true").lower() == "true":
            # Use the standard symbol format with a hyphen.
            configs["binance"] = ExchangeConfig(
                name="binance",
                credentials=ExchangeCredentials(
                    api_key=self.get("BINANCE_API_KEY"),
                    api_secret=self.get("BINANCE_API_SECRET"),
                ),
                rate_limit=int(self.get("BINANCE_RATE_LIMIT", default="20")),
                markets=self.get(
                    "BINANCE_MARKETS",
                    default="BTCUSDT,ETHUSDT,SOLUSDT,BTCUSDC,SOLUSDC,ETHUSDC",
                ).split(","),
                base_url=self.get(
                    "BINANCE_BASE_URL", default="https://api.binance.com"
                ),
                enabled=True,
            )

        if self.get("COINBASE_ENABLED", default="true").lower() == "true":
            configs["coinbase"] = ExchangeConfig(
                name="coinbase",
                credentials=ExchangeCredentials(
                    api_key=self.get("COINBASE_API_KEY"),
                    api_secret=self.get("COINBASE_API_SECRET"),
                ),
                rate_limit=int(self.get("COINBASE_RATE_LIMIT", default="15")),
                markets=self.get(
                    "COINBASE_MARKETS", default="BTC-USD,ETH-USD,SOL-USD"
                ).split(","),
                base_url=self.get(
                    "COINBASE_BASE_URL", default="https://api.coinbase.com"
                ),
                enabled=True,
            )

        return configs

    def _init_logging_config(self) -> LoggingConfig:
        """Initialize logging configuration."""
        log_path = Path(self.get("LOG_FILE", default="logs/data_fetcher.log"))
        if not log_path.is_absolute():
            log_path = self.project_root / log_path

        return LoggingConfig(
            log_level=self.get("LOG_LEVEL", default="INFO"),
            log_file=log_path,
            console_logging=self.get("CONSOLE_LOGGING", default="true").lower()
            == "true",
            file_logging=self.get("FILE_LOGGING", default="true").lower() == "true",
        )

    def validate(self):
        """Validate the entire configuration."""
        if not self.exchanges:
            raise ConfigurationError("No exchanges configured")

        for exchange in self.exchanges.values():
            if exchange.enabled and not exchange.markets:
                raise ConfigurationError(f"No markets configured for {exchange.name}")

        self.storage.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        if self.storage.backup_enabled and self.storage.backup_path:
            self.storage.backup_path.mkdir(parents=True, exist_ok=True)

        self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)
