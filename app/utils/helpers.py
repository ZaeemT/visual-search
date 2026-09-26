from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone, timedelta
import uuid
import math
import jwt
from pydantic import BaseModel
from ..core.responses import ServiceResponse
from passlib.context import CryptContext
from .constants import DEFAULT_ORDER, DEFAULT_PER_PAGE, PAGE
from ..core.config import get_settings

setting = get_settings()

password_context = CryptContext(
    schemes=['bcrypt'],
)

class PaginationParams(BaseModel):
    skip: int
    limit: int

class QueryParams(BaseModel):
    page: int = PAGE
    per_page: int = DEFAULT_PER_PAGE
    order: Dict[str, Any] = DEFAULT_ORDER
    filter: Dict[str, Any] = {}
    without_pagination: bool = False

class Utils:
    @staticmethod
    def get_current_time():
        """
        Returns:
            str: Current UTC time in ISO format
        """
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def generate_password_hash(password: str) -> str:
        hash = password_context.hash(password)
        return hash
    
    @staticmethod
    def verify_password(password: str, hash: str) -> bool:
        return password_context.verify(password, hash)
    
    @staticmethod
    def create_access_token(user_data: Dict[str, Any], expiry: Optional[timedelta] = None, refresh: bool = False) -> Union[tuple[str, str], str]:
        exp = (expiry + datetime.now(timezone.utc)) if expiry is not None else (datetime.now(timezone.utc) + timedelta(minutes=setting.ACCESS_TOKEN_EXPIRE_MINUTES))
        
        payload = {
            'user': user_data,
            'exp': exp,
            'jti': str(uuid.uuid4()),
            'refresh': refresh
        }

        token = jwt.encode(
            payload,
            key=setting.JWT_SECRET,
            algorithm=setting.JWT_ALGORITHM
        )

        if expiry is not None:
            return token, exp.isoformat()

        return token
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            token_data = jwt.decode(
                jwt=token,
                key=setting.JWT_SECRET,
                algorithms=[setting.JWT_ALGORITHM]
            )

            return token_data
        except jwt.PyJWKError as jwte:
            print(jwte)
            # logging.exception(jwte)
            return None

        except Exception as e:
            print(e)
            # logging.exception(e)
            return None
    
    @staticmethod
    def get_response(success: bool, message: str, data: Any = None, status_code: int = 200) -> ServiceResponse:
        """
        Format response data.
        """
        data = Utils.sanitize_data(data)
        return ServiceResponse(
            success=success,
            message=message,
            data=data,
            status_code=status_code
        )
    
    @staticmethod
    def sanitize_data(data: Any, fields_to_remove: Optional[List[str]] = None) -> Any:
        """
        Converts '_id' fields to str and removes specified fields from the data.

        Args:
            data (Any): The data to sanitize (dict or list of dicts).
            fields_to_remove (Optional[List[str]]): List of field names to remove.

        Returns:
            Any: Sanitized data.
        """
        def process_item(item):
            if isinstance(item, dict):
                # Convert '_id' to str if present
                if '_id' in item and item['_id'] is not None:
                    item['_id'] = str(item['_id'])
                # Remove specified fields
                if fields_to_remove:
                    for field in fields_to_remove:
                        item.pop(field, None)
                # Recursively process nested dicts and lists
                for key, value in item.items():
                    if isinstance(value, dict) or isinstance(value, list):
                        item[key] = process_item(value)
                return item
            elif isinstance(item, list):
                return [process_item(i) for i in item]
            return item

        return process_item(data)


    @staticmethod
    def calculate_pagination(current_page: int, per_page: int) -> PaginationParams:
        """
        Calculate skip and limit values for database queries.
        
        Args:
            current_page (int): Current page number (1-based)
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, int]: Dictionary containing 'skip' and 'limit' values
        """

        skip = per_page * (current_page - 1)
        limit = per_page
        return PaginationParams(skip=skip, limit=limit)
    
    @staticmethod
    def paginate(data: Any, current_page: int, per_page: int, total: int, key: str = "data") -> Dict[str, Any]:
        """
        Format data with pagination metadata.
        
        Args:
            data: The data to be paginated (list, dict, etc.)
            current_page (int): Current page number
            per_page (int): Number of items per page
            total (int): Total number of items
            key (str): Key name for the data in response
            
        Returns:
            Dict[str, Any]: Formatted response with data and pagination info
        """

        if data is None:
            data = []

        return {
            key: data,
            "pagination": {
                "current_page": int(current_page),
                "per_page": int(per_page),
                "total": int(total),
                "last_page": math.ceil(int(total) / int(per_page)) if per_page > 0 else 1,
                "first_page": 1
            }
        }
    

