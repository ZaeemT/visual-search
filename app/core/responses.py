from pydantic import BaseModel
from typing import Any, Optional

class ServiceResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    status_code: int = 200

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_details: Optional[Any] = None
    status_code: int = 500
