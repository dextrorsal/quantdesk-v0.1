from datetime import datetime, timezone
from typing import Union

def convert_timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """Convert a Unix timestamp to a datetime object."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

def get_timestamp_from_datetime(dt: datetime) -> int:
    """Convert a datetime object to a Unix timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def get_current_timestamp() -> int:
    """Get the current Unix timestamp."""
    return int(datetime.now(timezone.utc).timestamp()) 