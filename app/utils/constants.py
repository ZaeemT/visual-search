from typing import Any, Dict
from pymongo import DESCENDING

DEFAULT_ORDER: Dict[str, int] = {"createdAt": DESCENDING} 
DEFAULT_PER_PAGE: int = 10
PAGE: int = 1

REFRESH_TOKEN_EXPIRY: int = 3   # number of days