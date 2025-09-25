#!/usr/bin/env python3
"""
Security limits for mainnet trading.
Enforces transaction size, fee, slippage limits, position limits, and emergency controls for safe trading.
"""

import logging
from typing import Dict, Any, Optional
from functools import wraps
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default limits - these are only fallbacks if nothing is provided
DEFAULT_LIMITS = {
    "max_trade_size": 0.01,  # 0.01 SOL maximum per trade
    "max_fee": 0.00001,  # 0.00001 SOL maximum fee
    "slippage_bps": 300,  # 3% maximum slippage (300 basis points)
    "enable_trading": True,  # Enable/disable trading
    "test_mode": True,  # Default to test mode (set to False for real trading)
    "requires_confirmation": True,  # Require confirmation for trades
    # New position limits
    "max_position_size": {
        "SOL-PERP": 1.0,  # Maximum 1 SOL position
        "BTC-PERP": 0.01,  # Maximum 0.01 BTC position
        "ETH-PERP": 0.1,  # Maximum 0.1 ETH position
        "default": 0.1,  # Default limit for other markets
    },
    # Leverage limits
    "max_leverage": {
        "SOL-PERP": 5,  # 5x max leverage for SOL
        "BTC-PERP": 3,  # 3x max leverage for BTC
        "ETH-PERP": 4,  # 4x max leverage for ETH
        "default": 2,  # 2x default leverage limit
    },
    # Daily volume limits (in SOL equivalent)
    "daily_volume_limit": 10.0,  # Maximum 10 SOL daily volume
    # Emergency controls
    "emergency_shutdown_triggers": {
        "loss_threshold_pct": 5.0,  # Emergency stop if 5% loss in position
        "volume_spike_multiplier": 3.0,  # Stop if volume spikes 3x above average
        "max_drawdown_pct": 10.0,  # Maximum allowed drawdown percentage
        "position_age_hours": 24,  # Alert on positions older than 24 hours
    },
}


class SecurityLimits:
    """
    Security limits enforcer for mainnet trading with enhanced position controls.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize security limits.

        Args:
            config_path: Path to configuration file (optional)
            **kwargs: Override default limits with provided values
        """
        self.limits = DEFAULT_LIMITS.copy()
        self.daily_volume = Decimal("0")
        self.last_volume_reset = datetime.utcnow()
        self.positions = {}  # Track open positions
        self.trading_enabled = True
        self.emergency_shutdown = False
        self._lock = threading.Lock()  # Thread safety for volume tracking

        # Create logs directory
        self.log_dir = Path(os.path.expanduser("~/.config/mainnet_trading/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self._update_limits(config)
                logger.info(f"Loaded security limits from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load security limits from {config_path}: {e}")

        # Override with any provided kwargs
        self._update_limits(kwargs)

        # Log the current limits
        self._log_limits_configuration()

    def _update_limits(self, new_limits: Dict[str, Any]):
        """Update limits with new values, handling nested dictionaries."""
        for key, value in new_limits.items():
            if key in self.limits:
                if isinstance(self.limits[key], dict) and isinstance(value, dict):
                    self.limits[key].update(value)
                else:
                    self.limits[key] = value

    def _log_limits_configuration(self):
        """Log the current security limits."""
        logger.info("Security limits configuration:")
        for key, value in self.limits.items():
            logger.info(f"  - {key}: {value}")

    def validate_trade_size(
        self, market: str, size: float, quote_price: Optional[float] = None
    ) -> bool:
        """
        Validate trade size against limits.

        Args:
            market: Market symbol (e.g., "SOL-USDC" or "SOL-PERP")
            size: Trade size in base currency
            quote_price: Price quote for non-SOL markets (to convert to SOL value)
        Returns:
            True if valid, False otherwise
        """
        # Early return if trading is disabled
        if not self.limits["enable_trading"]:
            logger.warning("Trading is currently disabled in security limits")
            return False

        # Use max_position_size if available for this market
        max_pos_size = self.limits.get("max_position_size", {}).get(
            market, self.limits.get("max_position_size", {}).get("default", None)
        )
        if max_pos_size is not None:
            if size > max_pos_size:
                logger.warning(
                    f"Trade size {size} exceeds max_position_size {max_pos_size} for {market}"
                )
                return False
            return True

        # For SOL spot markets, use max_trade_size
        if market.startswith("SOL"):
            if size > self.limits["max_trade_size"]:
                logger.warning(
                    f"Trade size {size} SOL exceeds maximum allowed {self.limits['max_trade_size']} SOL"
                )
                return False
            return True

        # For non-SOL markets with a quote price, calculate equivalent SOL value
        elif quote_price is not None:
            estimated_sol_value = size * quote_price / 100  # Example conversion
            if estimated_sol_value > self.limits["max_trade_size"]:
                logger.warning(
                    f"Estimated SOL value {estimated_sol_value} of trade exceeds "
                    f"maximum allowed {self.limits['max_trade_size']} SOL"
                )
                return False
            return True
        else:
            logger.warning(
                f"Cannot validate non-SOL market {market} without a quote price"
            )
            return False

    def validate_position_size(
        self, market: str, new_size: float, current_size: float = 0
    ) -> bool:
        """
        Validate if a new position size is within limits.

        Args:
            market: Market symbol (e.g., "SOL-PERP")
            new_size: Proposed new position size
            current_size: Current position size (default 0)

        Returns:
            True if valid, False otherwise
        """
        if self.emergency_shutdown:
            logger.error("Emergency shutdown active - no new positions allowed")
            return False

        max_size = self.limits["max_position_size"].get(
            market, self.limits["max_position_size"]["default"]
        )

        total_size = abs(current_size + new_size)
        if total_size > max_size:
            logger.warning(
                f"Position size {total_size} exceeds maximum allowed {max_size} for {market}"
            )
            return False
        return True

    def validate_leverage(self, market: str, leverage: float) -> bool:
        """
        Validate if leverage is within limits.

        Args:
            market: Market symbol
            leverage: Proposed leverage

        Returns:
            True if valid, False otherwise
        """
        max_leverage = self.limits["max_leverage"].get(
            market, self.limits["max_leverage"]["default"]
        )

        if leverage > max_leverage:
            logger.warning(
                f"Leverage {leverage}x exceeds maximum allowed {max_leverage}x for {market}"
            )
            return False
        return True

    def update_daily_volume(
        self, trade_size: float, quote_price: Optional[float] = None
    ) -> bool:
        """
        Update and check daily volume limits.

        Args:
            trade_size: Size of the trade
            quote_price: Price for conversion to SOL value

        Returns:
            True if within limits, False otherwise
        """
        with self._lock:
            # Reset daily volume if it's a new day
            now = datetime.utcnow()
            if (now - self.last_volume_reset).days >= 1:
                self.daily_volume = Decimal("0")
                self.last_volume_reset = now

            # Convert trade size to SOL value if needed
            if quote_price is not None:
                sol_value = Decimal(str(trade_size * quote_price / 100))
            else:
                sol_value = Decimal(str(trade_size))

            # Check if this trade would exceed daily limit
            new_volume = self.daily_volume + sol_value
            if new_volume > Decimal(str(self.limits["daily_volume_limit"])):
                logger.warning(
                    f"Trade would exceed daily volume limit of {self.limits['daily_volume_limit']} SOL"
                )
                return False

            self.daily_volume = new_volume
            return True

    def check_emergency_shutdown(
        self,
        current_loss_pct: Optional[float] = None,
        current_volume: Optional[float] = None,
        current_drawdown: Optional[float] = None,
    ) -> bool:
        """
        Check if emergency shutdown should be triggered.

        Args:
            current_loss_pct: Current loss percentage if any
            current_volume: Current trading volume
            current_drawdown: Current drawdown percentage

        Returns:
            True if shutdown triggered, False otherwise
        """
        triggers = self.limits["emergency_shutdown_triggers"]

        # Check loss threshold
        if (
            current_loss_pct is not None
            and current_loss_pct > triggers["loss_threshold_pct"]
        ):
            logger.error(
                f"Emergency shutdown: Loss {current_loss_pct}% exceeds threshold "
                f"{triggers['loss_threshold_pct']}%"
            )
            self.emergency_shutdown = True
            return True

        # Check volume spike
        if current_volume is not None and Decimal(
            str(current_volume)
        ) > self.daily_volume * Decimal(str(triggers["volume_spike_multiplier"])):
            logger.error(
                f"Emergency shutdown: Volume spike detected "
                f"({current_volume} > {self.daily_volume * Decimal(str(triggers['volume_spike_multiplier']))})"
            )
            self.emergency_shutdown = True
            return True

        # Check drawdown
        if (
            current_drawdown is not None
            and current_drawdown > triggers["max_drawdown_pct"]
        ):
            logger.error(
                f"Emergency shutdown: Drawdown {current_drawdown}% exceeds maximum "
                f"{triggers['max_drawdown_pct']}%"
            )
            self.emergency_shutdown = True
            return True

        return False

    def reset_emergency_shutdown(self):
        """Reset emergency shutdown state - requires manual intervention."""
        if self.emergency_shutdown:
            logger.warning("Resetting emergency shutdown state - manual override")
            self.emergency_shutdown = False

    def log_trade(self, trade_info: Dict[str, Any], status: str = "initiated"):
        """
        Log trade information.

        Args:
            trade_info: Trade information dictionary
            status: Trade status (initiated, executed, failed)
        """
        # Create a new log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            **trade_info,
        }

        # Determine log file path (one file per day)
        log_file = self.log_dir / f"trades_{datetime.utcnow().strftime('%Y%m%d')}.json"

        try:
            # Load existing logs or create new array
            logs = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    logs = json.load(f)

            # Add new log entry
            logs.append(log_entry)

            # Write back to file
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def validate_swap_size(self, market: str, usd_value: float) -> bool:
        """
        Validate if a token swap size is within limits.

        Args:
            market: Market symbol (e.g., "SOL-USDC")
            usd_value: USD value of the swap

        Returns:
            True if valid, False otherwise
        """
        if self.emergency_shutdown:
            logger.error("Emergency shutdown active - no swaps allowed")
            return False

        # Default max swap size in USD
        default_max_swap_size_usd = 100.0  # $100

        # Get market-specific limit if available
        # First try to get from max_swap_size if defined
        if "max_swap_size" in self.limits:
            max_swap_size = self.limits["max_swap_size"].get(
                market,
                self.limits["max_swap_size"].get("default", default_max_swap_size_usd),
            )
        else:
            # Fall back to max_position_size (treating swap similarly to position size)
            max_swap_size = self.limits["max_position_size"].get(
                market, self.limits["max_position_size"]["default"]
            )
            # Convert to USD value using a rough approximation
            # This is a simplification - in production you'd use current market prices
            if market.startswith("SOL"):
                max_swap_size *= 80  # Approximate SOL price in USD
            elif market.startswith("BTC"):
                max_swap_size *= 40000  # Approximate BTC price
            elif market.startswith("ETH"):
                max_swap_size *= 2000  # Approximate ETH price
            else:
                max_swap_size *= 50  # Generic approximation

        if usd_value > max_swap_size:
            logger.warning(
                f"Swap value ${usd_value} exceeds maximum allowed ${max_swap_size} for {market}"
            )
            return False

        return True

    def validate_slippage(self, slippage_bps: int) -> bool:
        """
        Validate if slippage is within limits.

        Args:
            slippage_bps: Slippage in basis points (1% = 100 bps)

        Returns:
            True if valid, False otherwise
        """
        max_slippage = self.limits.get("slippage_bps", 300)  # Default to 3%

        if slippage_bps > max_slippage:
            logger.warning(
                f"Slippage {slippage_bps / 100}% exceeds maximum allowed {max_slippage / 100}%"
            )
            return False

        return True


# This function doesn't enforce limits, but passes them through for APIs that handle it themselves
def get_api_parameters(security_limits: SecurityLimits) -> Dict[str, Any]:
    """
    Get parameters for API calls based on security limits.

    Args:
        security_limits: SecurityLimits instance

    Returns:
        Dictionary of parameters for API calls
    """
    return {
        "max_trade_size": security_limits.limits["max_trade_size"],
        "max_fee": security_limits.limits["max_fee"],
        "slippage_bps": security_limits.limits["slippage_bps"],
        "requires_confirmation": security_limits.limits["requires_confirmation"],
        "test_mode": security_limits.limits["test_mode"],
    }
