from fastapi import Request
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials

from .exceptions import ForbiddenException, NotImplementedException
from ..utils.helpers import Utils
from typing import Union


class TokenBearer(HTTPBearer):
    def __init__(self, auto_error=True):
        super().__init__(auto_error=auto_error)

    async def __call__(
        self, request: Request
    ) -> Union[HTTPAuthorizationCredentials, None]:

        creds = await super().__call__(request)

        if not creds:
            raise ForbiddenException("Authorization header required")

        token_data = Utils.decode_token(creds.credentials)

        # print(f"Creds data: {creds.credentials}")

        token = creds.credentials
        if not self.token_valid(token):
            raise ForbiddenException("Invalid or expired token")

        self.verify_token_data(token_data)

        return token_data

    def token_valid(self, token: str) -> bool:
        token_data = Utils.decode_token(token)

        if token_data is not None:
            return True

        return False

    def verify_token_data(self, token_data: dict) -> None:
        raise NotImplementedError("Please Override this method in the child classes")


class AccessTokenBearer(TokenBearer):

    def verify_token_data(self, token_data: dict) -> None:
        if token_data and token_data["refresh"]:
            raise ForbiddenException("Please provide an access token")


class RefershTokenBearer(TokenBearer):

    def verify_token_data(self, token_data: dict) -> None:
        if token_data and not token_data["refresh"]:
            raise ForbiddenException("Please provide a refresh token")
