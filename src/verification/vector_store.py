from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VectorRecord:
    """Represents a record in the vector store."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class VectorStoreInterface(ABC):
    """Abstract interface for vector store operations."""

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results to return
            filter_conditions: Optional metadata filters
            
        Returns:
            List of VectorRecord objects sorted by similarity
        """
        pass

    @abstractmethod
    async def add_record(self, record: VectorRecord) -> None:
        """Add a single record to the vector store.
        
        Args:
            record: VectorRecord to add
        """
        pass

    @abstractmethod
    async def add_records(self, records: List[VectorRecord]) -> None:
        """Add multiple records to the vector store.
        
        Args:
            records: List of VectorRecord objects to add
        """
        pass

    @abstractmethod
    async def delete_record(self, record_id: str) -> None:
        """Delete a record by ID.
        
        Args:
            record_id: ID of record to delete
        """
        pass

    @abstractmethod
    async def get_record(self, record_id: str) -> Optional[VectorRecord]:
        """Retrieve a record by ID.
        
        Args:
            record_id: ID of record to retrieve
            
        Returns:
            VectorRecord if found, None otherwise
        """
        pass