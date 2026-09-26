from pydantic import BaseModel

class RefreshTokenModel(BaseModel):
    token: str
    user_id: str
    expiry_date: str    # datetime stored in isoformat

    class Config:
        collection = "refresh-tokens"