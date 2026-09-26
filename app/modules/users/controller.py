from fastapi import APIRouter, Depends, Query, Body, Path
from typing import Annotated
from ...core.responses import ServiceResponse
from ...utils.helpers import QueryParams
from .service import UserService
from .dto import UserGetDto, UserCreateDto, UserUpdateDto, UserDeleteDto

router = APIRouter()

def get_user_service() -> UserService:
    """Dependency to get UserService instance"""
    return UserService()


@router.get("/")
async def get_all_users(query_params: Annotated[QueryParams, Query()] = QueryParams(), user_service: UserService = Depends(get_user_service)) -> ServiceResponse:
    return await user_service.get_all(query_params)

@router.get("/{user_id}")
async def get_user_by_id(path_params: Annotated[UserGetDto, Path()], user_service: UserService = Depends(get_user_service)) -> ServiceResponse:
    return await user_service.get_by_id(path_params.user_id)

@router.post("/")
async def create_user(user_data: Annotated[UserCreateDto, Body()], user_service: UserService = Depends(get_user_service)) -> ServiceResponse:
    return await user_service.create(user_data.model_dump())

@router.patch("/{user_id}")
async def update_user(path_params: Annotated[UserGetDto, Path()], user_data: Annotated[UserUpdateDto, Body()], user_service: UserService = Depends(get_user_service)) -> ServiceResponse:
    return await user_service.update(path_params.user_id, user_data.model_dump(exclude_unset=True))

@router.delete("/{user_id}")
async def delete_user(path_params: Annotated[UserDeleteDto, Path()], user_service: UserService = Depends(get_user_service)) -> ServiceResponse:
    return await user_service.delete(path_params.user_id)
