from abc import ABC
from typing import Dict, Any
from .base_repository import BaseRepository
from bson import ObjectId
from ...core.responses import ServiceResponse
from ...utils.helpers import Utils, QueryParams
from ...core.exceptions import NotFoundException, DatabaseException

class BaseService(ABC):
    """
    Abstract base service class for business logic.
    """

    def __init__(self, repository: BaseRepository) -> None:
        self.repository = repository

    async def get_by_id(self, id: str) -> ServiceResponse:
        """
        Get a document by its ID.
        
        Args:
            id (ObjectId): The ID of the document to retrieve.
        
        Returns:
            ServiceResponse: Response containing the document or an error message.
        """
        try:
            document = await self.repository.GetById(ObjectId(id))
            if not document:
                raise NotFoundException("Record not found")
            
            return Utils.get_response(True, "Record retrieved successfully", document)
        except Exception as e:
            raise DatabaseException(f"Error retrieving document: {str(e)}")
    
    async def get_all(self, query_params: QueryParams) -> ServiceResponse:
        """
        Get all records with pagination and filtering
        
        Args:
            query_params: Query parameters including page, per_page, filter, order
        
        Returns:
            ServiceResponse with paginated data
        """
        try:
            page = query_params.page
            per_page = query_params.per_page
            order = query_params.order
            filter_query = query_params.filter
            without_pagination = query_params.without_pagination

            response = await self.repository.List(filter_query, order, page, per_page, without_pagination)
            return Utils.get_response(True, "Records retrieved successfully", response)

        except Exception as e:
            raise DatabaseException(f"Error retrieving documents: {str(e)}")

    async def create(self, data: Dict[str, Any]) -> ServiceResponse:
        """
        Create new record
        
        Args:
            data: Data to create record
        
        Returns:
            ServiceResponse with created record
        """
        try:
            doc_id = await self.repository.Create(data)

            document = await self.repository.GetById(doc_id)
            if not document:
                raise NotFoundException("Record not found")
            
            return Utils.get_response(True, "Record created successfully", document, 201)
        except Exception as e:
            raise DatabaseException(f"Error creating document: {str(e)}")
        
    async def update(self, id: str, data: Dict[str, Any]) -> ServiceResponse:
        """ 
        Update record by ID
        
        Args:
            id: Document ID to update
            data: Data to update
        
        Returns:
            ServiceResponse with updated record
        """
        try:
            document = await self.repository.GetById(ObjectId(id))
            if not document:
                raise NotFoundException("Record not found")
            
            updated_document = await self.repository.UpdateById(ObjectId(id), data)
            if not updated_document:
                raise NotFoundException("Record not found after update")
            
            return Utils.get_response(True, "Record updated successfully", updated_document)
        except Exception as e:
            raise DatabaseException(f"Error updating document: {str(e)}")
        
    async def delete(self, id: str) -> ServiceResponse:
        """
        Delete record by ID (soft delete)
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            ServiceResponse confirming deletion
        """
        try:
            document = await self.repository.GetById(ObjectId(id))
            if not document:
                raise NotFoundException("Record not found")
            
            deleted = await self.repository.DeleteById(ObjectId(id))
            if not deleted:
                raise NotFoundException("Record not deleted successfully")
            
            return Utils.get_response(True, "Record deleted successfully")
        except Exception as e:
            raise DatabaseException(f"Error deleting document: {str(e)}")