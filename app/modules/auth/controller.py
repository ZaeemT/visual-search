from fastapi import APIRouter, Depends, Query, Body, Path
from typing import Annotated
from ...core.responses import ServiceResponse
from ...core.security import AccessTokenBearer
from .service import AuthService
from .dto import ChangePasswordDto, RefreshTokenDto, SignUpDto, LoginDto

router = APIRouter()
access_token_bearer = AccessTokenBearer()

def get_auth_service():
    """Dependency to get AuthService instance"""
    return AuthService()

@router.post("/signup")
async def sign_up(user_data: Annotated[SignUpDto, Body()], auth_service: AuthService = Depends(get_auth_service)) -> ServiceResponse:
    return await auth_service.sign_up(user_data)

@router.post("/login")
async def login(user_creds: Annotated[LoginDto, Body()], auth_service: AuthService = Depends(get_auth_service)) -> ServiceResponse:
    return await auth_service.login(user_creds)

@router.post("/refresh-token")
async def refresh_token(data: Annotated[RefreshTokenDto, Body()], auth_service: AuthService = Depends(get_auth_service)) -> ServiceResponse:
    return await auth_service.refresh_token(data)

@router.put("/change-password")
async def change_password(data: Annotated[ChangePasswordDto, Body()], request_user = Depends(access_token_bearer), auth_service: AuthService = Depends(get_auth_service)) -> ServiceResponse:
    # print("request_user", request_user)
    return await auth_service.change_password(data, request_user)

@router.get("/current-user")
async def get_current_user(request_user = Depends(access_token_bearer), auth_service: AuthService = Depends(get_auth_service)) -> ServiceResponse:
    return await auth_service.get_current_user(request_user)