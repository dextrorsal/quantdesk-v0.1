#!/usr/bin/env python3
"""
Time Utilities for CLI

Extended time utilities providing:
- Timestamp conversions
- Time range validation
- Time string parsing
- Common time operations
"""

from datetime import datetime, timezone, timedelta
from typing import Union, Optional, Tuple

from src.core.time_utils import (
    convert_timestamp_to_datetime,
    get_timestamp_from_datetime,
    get_current_timestamp
)

def parse_time_string(time_str: str, default_tz: Optional[timezone] = None) -> datetime:
    """
    Parse a time string into a datetime object.
    Supports formats:
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM:SS
    - YYYY-MM-DD HH:MM:SS±HH:MM
    
    Args:
        time_str: Time string to parse
        default_tz: Default timezone if not specified in string
        
    Returns:
        datetime: Parsed datetime object
    """
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=default_tz or timezone.utc)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Invalid time format: {time_str}")

def validate_time_range(start: Union[str, datetime], end: Union[str, datetime]) -> Tuple[datetime, datetime]:
    """
    Validate a time range and convert to datetime objects.
    
    Args:
        start: Start time (string or datetime)
        end: End time (string or datetime)
        
    Returns:
        tuple: (start_datetime, end_datetime)
    """
    # Convert strings to datetime if needed
    if isinstance(start, str):
        start = parse_time_string(start)
    if isinstance(end, str):
        end = parse_time_string(end)
    
    # Ensure both have timezones
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    
    # Validate range
    if start >= end:
        raise ValueError("Start time must be before end time")
    
    # Don't allow future end times
    now = datetime.now(timezone.utc)
    if end > now:
        end = now
    
    return start, end

def get_time_range(period: str) -> Tuple[datetime, datetime]:
    """
    Get a time range based on a period string.
    Supported periods: 1h, 4h, 1d, 1w, 1m, 3m, 6m, 1y
    
    Args:
        period: Time period string
        
    Returns:
        tuple: (start_datetime, end_datetime)
    """
    now = datetime.now(timezone.utc)
    
    # Parse period string
    amount = int(period[:-1])
    unit = period[-1].lower()
    
    # Calculate delta based on unit
    if unit == 'h':
        delta = timedelta(hours=amount)
    elif unit == 'd':
        delta = timedelta(days=amount)
    elif unit == 'w':
        delta = timedelta(weeks=amount)
    elif unit == 'm':
        delta = timedelta(days=amount * 30)  # Approximate
    elif unit == 'y':
        delta = timedelta(days=amount * 365)  # Approximate
    else:
        raise ValueError(f"Invalid period unit: {unit}")
    
    start = now - delta
    return start, now

def format_timestamp(timestamp: Union[int, float], fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Format a timestamp as a string.
    
    Args:
        timestamp: Unix timestamp
        fmt: Output format string
        
    Returns:
        str: Formatted time string
    """
    dt = convert_timestamp_to_datetime(timestamp)
    return dt.strftime(fmt)

def get_resolution_seconds(resolution: str) -> int:
    """
    Get number of seconds for a resolution string.
    
    Args:
        resolution: Resolution string (e.g., "1m", "5m", "1h", "1d")
        
    Returns:
        int: Number of seconds
    """
    amount = int(resolution[:-1])
    unit = resolution[-1].lower()
    
    if unit == 'm':
        return amount * 60
    elif unit == 'h':
        return amount * 3600
    elif unit == 'd':
        return amount * 86400
    elif unit == 'w':
        return amount * 604800
    else:
        raise ValueError(f"Invalid resolution unit: {unit}")

def align_timestamp(timestamp: int, resolution: str) -> int:
    """
    Align a timestamp to a resolution boundary.
    
    Args:
        timestamp: Unix timestamp
        resolution: Resolution string
        
    Returns:
        int: Aligned timestamp
    """
    seconds = get_resolution_seconds(resolution)
    return timestamp - (timestamp % seconds) 