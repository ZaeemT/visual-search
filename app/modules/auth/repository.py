from .model import RefreshTokenModel
from ..base import BaseRepository
from ...database import get_collection

class RefreshTokenRepository(BaseRepository):

    def __init__(self) -> None:
        super().__init__(get_collection(RefreshTokenModel.Config.collection))