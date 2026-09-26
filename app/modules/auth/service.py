from datetime import timedelta, datetime
from typing import Any, Dict
from bson import ObjectId
from ...core.exceptions import BadRequestException, UnauthorizedException, NotFoundException
from ...utils.constants import REFRESH_TOKEN_EXPIRY
from ...utils.helpers import Utils
from ..users import UserRepository
from .repository import RefreshTokenRepository
from .model import RefreshTokenModel
from .dto import ChangePasswordDto, RefreshTokenDto, SignUpDto, LoginDto


class AuthService:

    def __init__(self) -> None:
        self.user_repository = UserRepository()
        self.refresh_token_repository = RefreshTokenRepository()

    async def sign_up(self, sign_up_data: SignUpDto):
        exist = await self.user_repository.GetOne({"email": sign_up_data.email})

        if exist is not None:
            raise BadRequestException("Email already exists")

        hash_password = Utils.generate_password_hash(sign_up_data.password)

        sign_up_data.password = hash_password

        user = await self.user_repository.Upsert(
            {"email": sign_up_data.email}, sign_up_data.model_dump()
        )
        if user is not None and "password" in user:
            user.pop("password")

        return Utils.get_response(True, "User created successfully", user, 201)

    async def login(self, login_data: LoginDto):
        user_dict = await self.user_repository.GetOne({"email": login_data.email})

        if user_dict is None:
            raise UnauthorizedException()

        password_match = Utils.verify_password(login_data.password, user_dict['password'])
        if password_match is False:
            raise UnauthorizedException()

        # Create user payload for token (exclude password)
        user_payload = Utils.sanitize_data(user_dict, ['password'])

        access_token = Utils.create_access_token(user_payload)

        expiry = timedelta(days=REFRESH_TOKEN_EXPIRY)
        refresh_token, expiry_date = Utils.create_access_token(
            user_payload,
            expiry,
            refresh=True,
        )

        refresh_token_details = RefreshTokenModel(
            user_id=str(user_dict["_id"]), 
            token=refresh_token,
            expiry_date=expiry_date
        )

        await self.refresh_token_repository.Upsert(
            { 'user_id': str(user_dict['_id']) },
            refresh_token_details.model_dump()
        )

        response = {
            'user': user_payload,
            'access_token': access_token,
            'refresh_token': refresh_token
        }

        return Utils.get_response(True, "Login successfully", response)

    async def refresh_token(self, data: RefreshTokenDto):
        token = data.token
        refresh_token_details = await self.refresh_token_repository.GetOne({"token": token})

        if refresh_token_details is None:
            raise UnauthorizedException()

        if Utils.get_current_time() > refresh_token_details['expiry_date']:
            raise UnauthorizedException()

        user_dict = await self.user_repository.GetById(ObjectId(refresh_token_details['user_id']))

        if user_dict is None:
            raise UnauthorizedException()

        user_payload = Utils.sanitize_data(user_dict, ['password'])

        access_token = Utils.create_access_token(user_payload)
        expiry = timedelta(days=REFRESH_TOKEN_EXPIRY)
        refresh_token, expiry_date = Utils.create_access_token(
            user_payload,
            expiry,
            refresh=True,
        )

        await self.refresh_token_repository.Update(
            { 'user_id': str(user_dict['_id']) },
            {
                'token': refresh_token,
                'expiry_date': expiry_date
            }
        )

        return Utils.get_response(True, "Token refreshed successfully", {
            'access_token': access_token,
            'refresh_token': refresh_token
        })
        
    async def change_password(self, data: ChangePasswordDto, request_user: Dict[str, Any]):
        user = request_user.get('user')
        user_exist = await self.user_repository.GetById(ObjectId(user['_id']))
        if not user_exist:
            raise NotFoundException("User not found")
        
        password_match = Utils.verify_password(data.old_password, user_exist['password'])
        if password_match is False:
            raise UnauthorizedException()
        
        new_hash_password = Utils.generate_password_hash(data.new_password)
        updated_user = await self.user_repository.UpdateById(ObjectId(user['_id']), {'password': new_hash_password})

        updated_user = Utils.sanitize_data(updated_user, ['password'])

        return Utils.get_response(True, "Password changed successfully", updated_user)

    async def get_current_user(self, request_user: Dict[str, Any]):
        user_email = request_user['user']['email']

        user = await self.user_repository.GetOne({"email": user_email})

        user = Utils.sanitize_data(user, ['password'])

        return Utils.get_response(True, "User found successfully", user)