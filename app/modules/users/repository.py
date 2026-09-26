from ..base.base_repository import BaseRepository
from ...database import get_collection
from .model import UserModel

class UserRepository(BaseRepository):

    def __init__(self) -> None:
        super().__init__(get_collection(UserModel.Config.collection))
