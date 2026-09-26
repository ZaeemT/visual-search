from abc import ABC
from typing import Any, Dict, Union, List, Optional
from bson import ObjectId
from pymongo import ReturnDocument
from motor.motor_asyncio import AsyncIOMotorCollection
from ...utils.helpers import Utils, PaginationParams
from ...core.exceptions import DatabaseException

class BaseRepository(ABC):
    """
    Abstract base repository class for data access.
    """

    def __init__(self, collection) -> None:
        self.collection: AsyncIOMotorCollection = collection

    async def List(self, filter: Dict[str, Any], order: Dict[str, Any], page: int, per_page: int, without_pagination: bool = False) -> Union[Dict, List]:
        """
        List documents with pagination and filtering
        
        Args:
            filter_query: MongoDB filter query
            order: Sort order dict e.g. {"created_at": "desc"}
            page: Page number (1-based)
            per_page: Number of documents per page
            without_pagination: Return all results without pagination
            projection: Fields to include/exclude
        
        Returns:
            List of documents or paginated response
        """
        try:
            filter["deleted_at"] = None  # Exclude soft-deleted items
            pagination: PaginationParams = Utils.calculate_pagination(page, per_page)
        
            total_count = await self.collection.count_documents(filter)
            if without_pagination:
                data = await self.collection.find(filter).sort(order).to_list()
                return data

            data = await self.collection.find(filter).sort(order).skip(pagination.skip).limit(pagination.limit).to_list()
            return Utils.paginate(data, page, per_page, total_count, self.collection.name.lower())
        except Exception as e:
            raise DatabaseException(f"Error fetching documents: {str(e)}") 
        
    async def GetOne(self, filter: Dict[str, Any], projection: Optional[Any] = None) -> Union[Dict, None]:
        """
        Get a single document by filter.

        Args:
            filter_query: MongoDB filter query
            projection: Fields to include/exclude
        
        Returns:
            Document or None
        """
        try:
            filter["deleted_at"] = None  # Exclude soft-deleted items
            return await self.collection.find_one(filter, projection)
        except Exception as e:
            raise DatabaseException(f"Error fetching document: {str(e)}")
        
    async def GetById(self, doc_id: ObjectId, projection: Optional[Any] = None) -> Union[Dict, None]:
        """
        Get a single document by ID.

        Args:
            doc_id: Document ID
            projection: Fields to include/exclude

        Returns:
            Document or None
        """
        try:
            filter = {"_id": doc_id, "deleted_at": None}

            return await self.collection.find_one(filter, projection)
        except Exception as e:
            raise DatabaseException(f"Error fetching document by ID: {str(e)}")

    async def Create(self, data: Any) -> ObjectId:
        """
        Add new document
        
        Args:
            create_data: Document data to insert
        
        Returns:
            Created document ID
        """
        try:
            data["created_at"] = Utils.get_current_time()
            data["updated_at"] = None
            data["deleted_at"] = None  # Soft delete field

            result = await self.collection.insert_one(data)
            return result.inserted_id
        except Exception as e:
            raise DatabaseException(f"Error inserting document: {str(e)}")
        
    async def Upsert(self, filter: Dict[str, Any], data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update document or insert if not exists
        
        Args:
            filter_query: Filter to find document
            update_data: Data to update/insert
        
        Returns:
            Updated/inserted document
        """
        try:
            now = Utils.get_current_time()

            # Prepare update data
            update_doc = {
                "$set": {
                    **data,
                    "updated_at": now
                },
                "$setOnInsert": {
                    "created_at": now,
                    # "updated_at": None,
                    "deleted_at": None,
                }
            }

            result = await self.collection.find_one_and_update(
                filter,
                update_doc,
                upsert=True,
                return_document=ReturnDocument.AFTER
            )

            return result  # Return the updated/inserted document
        
        except Exception as e:
            raise DatabaseException(f"Error upserting document: {str(e)}")
    
    async def Update(self, filter: Dict[str, Any], data: Any) -> Optional[Dict[str, Any]]:
        """
        Update document
        
        Args:
            filter_query: Filter to find document
            data: Data to update
        
        Returns:
            Updated document
        """
        try:
            filter["deleted_at"] = None  # Exclude soft-deleted items
            now = Utils.get_current_time()

            result = await self.collection.find_one_and_update(
                filter,
                {"$set": {**data, "updated_at": now}},
                upsert=False,
                return_document=ReturnDocument.AFTER
            )

            return result
        except Exception as e:
            raise DatabaseException(f"Error updating document: {str(e)}")

    async def UpdateById(self, doc_id: ObjectId, data: Any) -> Optional[Dict[str, Any]]:
        """
        Update document by ID
        
        Args:
            doc_id: Document ID to update
            data: Data to update
        
        Returns:
            Updated document
        """
        try:
            filter = {"_id": doc_id, "deleted_at": None}

            now = Utils.get_current_time()
            result = await self.collection.find_one_and_update(
                filter,
                {"$set": {**data, "updated_at": now}},
                upsert=False,
                return_document=ReturnDocument.AFTER
            )

            return result
        except Exception as e:
            raise DatabaseException(f"Error updating document by ID: {str(e)}")
        
    async def Count(self, filter: Dict[str, Any]) -> int:
        """
        Count documents matching filter
        
        Args:
            filter_query: MongoDB filter query
        
        Returns:
            Count of matching documents
        """
        try:
            filter["deleted_at"] = None  # Exclude soft-deleted items
            return await self.collection.count_documents(filter)
        except Exception as e:
            raise DatabaseException(f"Error counting documents: {str(e)}")
        
    async def Delete(self, filter: Dict[str, Any]) -> bool:
        """
        Soft delete document by ID

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            filter["deleted_at"] = None  # Exclude soft-deleted items
            result = await self.collection.update_one(
                filter,
                {"$set": {"deleted_at": Utils.get_current_time()}},
            )
            return result.modified_count > 0
        except Exception as e:
            raise DatabaseException(f"Error deleting document: {str(e)}")

    async def DeleteById(self, doc_id: ObjectId) -> bool:
        """
        Soft delete document by ID

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            if isinstance(doc_id, ObjectId):
                filter = {"_id": ObjectId(doc_id), "deleted_at": None}

            result = await self.collection.update_one(
                filter,
                {"$set": {"deleted_at": Utils.get_current_time()}},
            )
            return result.modified_count > 0
        except Exception as e:
            raise DatabaseException(f"Error deleting document by ID: {str(e)}")
        