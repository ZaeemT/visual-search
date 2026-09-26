from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator
from typing import Optional, Annotated
from ..users import UserRole


class SignUpDto(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, use_enum_values=True
    )

    name: Annotated[str, Field(min_length=3, max_length=50)]
    email: Annotated[EmailStr, Field(max_length=100)]
    password: Annotated[str, Field(min_length=8, max_length=128)]
    role: Annotated[UserRole, Field(default=UserRole.USER)]

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginDto(BaseModel):
    email: Annotated[EmailStr, Field(min_length=3, max_length=50)]
    password: Annotated[str, Field()]


class RefreshTokenDto(BaseModel):
    token: Annotated[str, Field()]


class ChangePasswordDto(BaseModel):
    old_password: Annotated[str, Field(min_length=8, max_length=128)]
    new_password: Annotated[str, Field(min_length=8, max_length=128)]

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v