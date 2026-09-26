from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from typing import Optional
import logging
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class Database:
    """
    Database connection manager
    """

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
            

    async def connect_to_mongodb(self):
        """
        Connect to MongoDB using the URI from settings
        """
        try:
            logger.info("Connecting to MongoDB...")
            settings = get_settings()
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[settings.DB_NAME]
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        
    async def close_mongodb_connection(self):
        """
        Close the MongoDB connection
        """
        try:
            if self.client:
                logger.info("Closing MongoDB connection...")
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")


# Create a global instance of the Database class
database = Database()

def get_collection(collection_name: str) -> AsyncIOMotorCollection:
    if database.db is None:
        raise Exception("Database connection not established")
    return database.db[collection_name]