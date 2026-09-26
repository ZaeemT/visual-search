from ..base.base_service import BaseService
from .repository import UserRepository

class UserService(BaseService):

    def __init__(self) -> None:
        super().__init__(UserRepository())