from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator
from typing import Optional, Annotated
from .model import UserRole

class UserGetDto(BaseModel):
    user_id: Annotated[str, Field(description="User ID")]

class UserCreateDto(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    name: Annotated[str, Field(min_length=3, max_length=50, description="User's full name")]
    email: Annotated[EmailStr, Field(max_length=100, description="User's email address")]
    password: Annotated[str, Field(min_length=8, max_length=128, description="User password")]
    role: Annotated[UserRole, Field(default=UserRole.USER, description="User role")]

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdateDto(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    name: Optional[Annotated[str, Field(
        min_length=2,
        max_length=50,
        description="User's full name"
    )]] = None
    role: Optional[UserRole] = None

class UserDeleteDto(BaseModel):
    user_id: Annotated[str, Field(description="User ID")]
