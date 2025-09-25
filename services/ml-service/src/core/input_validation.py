"""
Input Validation and Sanitization Module
Provides secure input validation for all API endpoints
"""
import re
import logging
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pydantic import BaseModel, validator, Field
import html

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class InputValidator:
    """Comprehensive input validation class"""
    
    # Security patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\b(OR|AND)\s+'.*'\s*=\s*'.*')",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"(\b(rm|del|format|shutdown|reboot|kill|ps|netstat)\b)",
        r"(\b(cat|ls|dir|type|more|less|head|tail)\s+)",
    ]
    
    @classmethod
    def validate_string(cls, value: Any, max_length: int = 255, allow_empty: bool = True) -> str:
        """Validate and sanitize string input"""
        if value is None:
            if allow_empty:
                return ""
            raise ValidationError("Value cannot be None")
        
        # Convert to string
        str_value = str(value).strip()
        
        # Check length
        if len(str_value) > max_length:
            raise ValidationError(f"String too long. Maximum length: {max_length}")
        
        # Check for empty string
        if not str_value and not allow_empty:
            raise ValidationError("Value cannot be empty")
        
        # Sanitize HTML entities
        sanitized = html.escape(str_value)
        
        # Check for malicious patterns
        cls._check_malicious_patterns(sanitized)
        
        return sanitized
    
    @classmethod
    def validate_number(cls, value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Validate numeric input"""
        if value is None:
            raise ValidationError("Number cannot be None")
        
        try:
            # Convert to float
            num_value = float(value)
            
            # Check range
            if min_val is not None and num_value < min_val:
                raise ValidationError(f"Number too small. Minimum: {min_val}")
            
            if max_val is not None and num_value > max_val:
                raise ValidationError(f"Number too large. Maximum: {max_val}")
            
            return num_value
            
        except (ValueError, TypeError):
            raise ValidationError("Invalid number format")
    
    @classmethod
    def validate_decimal(cls, value: Any, precision: int = 8) -> Decimal:
        """Validate decimal input for financial data"""
        if value is None:
            raise ValidationError("Decimal cannot be None")
        
        try:
            decimal_value = Decimal(str(value))
            
            # Check precision
            if decimal_value.as_tuple().exponent < -precision:
                raise ValidationError(f"Decimal precision too high. Maximum: {precision}")
            
            # Check for negative values (if not allowed)
            if decimal_value < 0:
                raise ValidationError("Decimal cannot be negative")
            
            return decimal_value
            
        except (InvalidOperation, ValueError, TypeError):
            raise ValidationError("Invalid decimal format")
    
    @classmethod
    def validate_symbol(cls, value: Any) -> str:
        """Validate trading symbol format"""
        symbol = cls.validate_string(value, max_length=20, allow_empty=False)
        
        # Trading symbol patterns: BASE/QUOTE, BASE-QUOTE, or BASEQUOTE (like BTCUSDT)
        symbol_pattern = r"^[A-Z]{2,10}([/-][A-Z]{2,10}|[A-Z]{2,10})$"
        
        if not re.match(symbol_pattern, symbol):
            raise ValidationError("Invalid symbol format. Use format: BTC/USDT, BTC-USDT, or BTCUSDT")
        
        return symbol
    
    @classmethod
    def validate_timeframe(cls, value: Any) -> str:
        """Validate trading timeframe"""
        timeframe = cls.validate_string(value, max_length=10, allow_empty=False)
        
        # Valid timeframes
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        
        if timeframe not in valid_timeframes:
            raise ValidationError(f"Invalid timeframe. Valid options: {valid_timeframes}")
        
        return timeframe
    
    @classmethod
    def validate_order_type(cls, value: Any) -> str:
        """Validate order type"""
        order_type = cls.validate_string(value, max_length=20, allow_empty=False)
        
        valid_types = ["market", "limit", "stop", "stop_limit"]
        
        if order_type.lower() not in valid_types:
            raise ValidationError(f"Invalid order type. Valid options: {valid_types}")
        
        return order_type.lower()
    
    @classmethod
    def validate_order_side(cls, value: Any) -> str:
        """Validate order side"""
        side = cls.validate_string(value, max_length=10, allow_empty=False)
        
        valid_sides = ["buy", "sell"]
        
        if side.lower() not in valid_sides:
            raise ValidationError(f"Invalid order side. Valid options: {valid_sides}")
        
        return side.lower()
    
    @classmethod
    def validate_limit(cls, value: Any, max_limit: int = 1000) -> int:
        """Validate limit parameter (for pagination, etc.)"""
        limit = cls.validate_number(value, min_val=1, max_val=max_limit)
        return int(limit)
    
    @classmethod
    def validate_timestamp(cls, value: Any) -> datetime:
        """Validate timestamp input"""
        if value is None:
            raise ValidationError("Timestamp cannot be None")
        
        try:
            if isinstance(value, (int, float)):
                # Unix timestamp
                return datetime.fromtimestamp(value)
            elif isinstance(value, str):
                # ISO format or other string formats
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            elif isinstance(value, datetime):
                return value
            else:
                raise ValidationError("Invalid timestamp format")
                
        except (ValueError, TypeError, OSError):
            raise ValidationError("Invalid timestamp format")
    
    @classmethod
    def validate_json_input(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON input data"""
        if not isinstance(data, dict):
            raise ValidationError("Input must be a JSON object")
        
        # Check for excessive nesting
        if cls._get_nesting_depth(data) > 10:
            raise ValidationError("JSON nesting too deep")
        
        # Check for excessive size
        json_str = str(data)
        if len(json_str) > 10000:  # 10KB limit
            raise ValidationError("JSON payload too large")
        
        return data
    
    @classmethod
    def _check_malicious_patterns(cls, value: str) -> None:
        """Check for malicious patterns in input"""
        value_lower = value.lower()
        
        # Check SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {value[:50]}...")
                raise ValidationError("Invalid input detected")
        
        # Check XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {value[:50]}...")
                raise ValidationError("Invalid input detected")
        
        # Check command injection patterns
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                logger.warning(f"Potential command injection detected: {value[:50]}...")
                raise ValidationError("Invalid input detected")
    
    @classmethod
    def _get_nesting_depth(cls, obj: Any, current_depth: int = 0) -> int:
        """Get nesting depth of JSON object"""
        if isinstance(obj, dict):
            return max((cls._get_nesting_depth(v, current_depth + 1) for v in obj.values()), default=current_depth)
        elif isinstance(obj, list):
            return max((cls._get_nesting_depth(item, current_depth + 1) for item in obj), default=current_depth)
        else:
            return current_depth

# Pydantic models for API validation
class TradingSymbolRequest(BaseModel):
    """Trading symbol validation model"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return InputValidator.validate_symbol(v)

class TimeframeRequest(BaseModel):
    """Timeframe validation model"""
    timeframe: str = Field(..., description="Trading timeframe")
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        return InputValidator.validate_timeframe(v)

class OrderRequest(BaseModel):
    """Order request validation model"""
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    order_type: str = Field(..., description="Order type")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, gt=0, description="Order price (for limit orders)")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return InputValidator.validate_symbol(v)
    
    @validator('side')
    def validate_side(cls, v):
        return InputValidator.validate_order_side(v)
    
    @validator('order_type')
    def validate_order_type(cls, v):
        return InputValidator.validate_order_type(v)
    
    @validator('quantity')
    def validate_quantity(cls, v):
        return InputValidator.validate_decimal(v, precision=8)
    
    @validator('price')
    def validate_price(cls, v):
        if v is not None:
            return InputValidator.validate_decimal(v, precision=8)
        return v

class PaginationRequest(BaseModel):
    """Pagination request validation model"""
    limit: int = Field(10, ge=1, le=1000, description="Number of items per page")
    offset: int = Field(0, ge=0, description="Number of items to skip")
    
    @validator('limit')
    def validate_limit(cls, v):
        return InputValidator.validate_limit(v)

class MarketDataRequest(BaseModel):
    """Market data request validation model"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Trading timeframe")
    limit: int = Field(100, ge=1, le=1000, description="Number of candles")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return InputValidator.validate_symbol(v)
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        return InputValidator.validate_timeframe(v)
    
    @validator('limit')
    def validate_limit(cls, v):
        return InputValidator.validate_limit(v)

# Utility functions for easy validation
def validate_trading_input(symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
    """Validate trading-related input parameters"""
    try:
        validated_symbol = InputValidator.validate_symbol(symbol)
        validated_timeframe = InputValidator.validate_timeframe(timeframe)
        validated_limit = InputValidator.validate_limit(limit)
        
        return {
            "symbol": validated_symbol,
            "timeframe": validated_timeframe,
            "limit": validated_limit,
            "valid": True
        }
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            "valid": False,
            "error": str(e)
        }

def validate_order_input(symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
    """Validate order input parameters"""
    try:
        validated_symbol = InputValidator.validate_symbol(symbol)
        validated_side = InputValidator.validate_order_side(side)
        validated_type = InputValidator.validate_order_type(order_type)
        validated_quantity = InputValidator.validate_decimal(quantity)
        validated_price = InputValidator.validate_decimal(price) if price else None
        
        return {
            "symbol": validated_symbol,
            "side": validated_side,
            "order_type": validated_type,
            "quantity": validated_quantity,
            "price": validated_price,
            "valid": True
        }
    except ValidationError as e:
        logger.error(f"Order validation error: {e}")
        return {
            "valid": False,
            "error": str(e)
        }

def sanitize_user_input(input_data: Any) -> Any:
    """Sanitize user input to prevent XSS and injection attacks"""
    if isinstance(input_data, str):
        return InputValidator.validate_string(input_data)
    elif isinstance(input_data, dict):
        return {k: sanitize_user_input(v) for k, v in input_data.items()}
    elif isinstance(input_data, list):
        return [sanitize_user_input(item) for item in input_data]
    else:
        return input_data
