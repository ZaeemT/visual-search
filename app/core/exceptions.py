from fastapi import status


class BaseAPIException(Exception):
    """Base exception class for custom API exceptions."""

    def __init__(
        self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class NotFoundException(BaseAPIException):
    """Exception raised when a resource is not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND)


class BadRequestException(BaseAPIException):
    """Exception raised for invalid input or bad request."""

    def __init__(self, message: str = "Bad request"):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)


class DatabaseException(BaseAPIException):
    """Exception raised for database-related errors."""

    def __init__(self, message: str = "Database error"):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ForbiddenException(BaseAPIException):
    """Exception raised for forbidden access."""

    def __init__(self, message: str = "Forbidden error"):
        super().__init__(message, status_code=status.HTTP_403_FORBIDDEN)


class UnauthorizedException(BaseAPIException):
    """Exception rasied for unauthourized access"""

    def __init__(self, message: str = "Unauthourized Access"):
        super().__init__(message, status_code=status.HTTP_401_UNAUTHORIZED)


class NotImplementedException(BaseAPIException):
    """Exception raised for not implemented errors."""

    def __init__(self, message: str = "Not implemented"):
        super().__init__(message, status_code=status.HTTP_501_NOT_IMPLEMENTED)