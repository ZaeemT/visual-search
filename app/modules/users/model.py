from pydantic import EmailStr, BaseModel
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class UserModel(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: UserRole

    class Config:
        collection = "users"
