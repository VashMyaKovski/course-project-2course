import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from .vector_store import VectorRecord

logger = logging.getLogger(__name__)


class MetadataFilter:
    """Filter vector store records by metadata criteria."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def filter_by_date(
        self,
        records: List[VectorRecord],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_field: str = "date"
    ) -> List[VectorRecord]:
        """
        Filter records by date range.
        
        Args:
            records: List of VectorRecord objects
            start_date: Start date in ISO 8601 format (inclusive)
            end_date: End date in ISO 8601 format (inclusive)
            date_field: Name of the date field in metadata
            
        Returns:
            Filtered list of VectorRecord objects
        """
        if not start_date and not end_date:
            return records

        filtered_records = []
        for record in records:
            date_str = record.metadata.get(date_field)
            if not date_str:
                continue
                
            try:
                record_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                self.logger.warning(f"Invalid date format in record {record.id}: {date_str}")
                continue

            if start_date:
                start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if record_date < start:
                    continue
                    
            if end_date:
                end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                if record_date > end:
                    continue
                    
            filtered_records.append(record)
            
        return filtered_records

    def filter_by_entities(
        self,
        records: List[VectorRecord],
        entities: List[str],
        entity_field: str = "entities",
        match_all: bool = False
    ) -> List[VectorRecord]:
        """
        Filter records by entities.
        
        Args:
            records: List of VectorRecord objects
            entities: List of entity strings to match
            entity_field: Name of the entities field in metadata (should be list)
            match_all: If True, require all entities to be present; if False, require any
            
        Returns:
            Filtered list of VectorRecord objects
        """
        if not entities:
            return records

        filtered_records = []
        for record in records:
            record_entities = record.metadata.get(entity_field, [])
            if not isinstance(record_entities, list):
                record_entities = [str(record_entities)]
                
            if match_all:
                if all(entity in record_entities for entity in entities):
                    filtered_records.append(record)
            else:
                if any(entity in record_entities for entity in entities):
                    filtered_records.append(record)
                    
        return filtered_records

    def filter_by_source(
        self,
        records: List[VectorRecord],
        sources: List[str],
        source_field: str = "source",
        exact_match: bool = True
    ) -> List[VectorRecord]:
        """
        Filter records by source.
        
        Args:
            records: List of VectorRecord objects
            sources: List of source strings to match
            source_field: Name of the source field in metadata
            exact_match: If True, require exact match; if False, allow partial match
            
        Returns:
            Filtered list of VectorRecord objects
        """
        if not sources:
            return records

        filtered_records = []
        for record in records:
            source = record.metadata.get(source_field)
            if not source:
                continue
                
            if exact_match:
                if source in sources:
                    filtered_records.append(record)
            else:
                if any(s in source for s in sources):
                    filtered_records.append(record)
                    
        return filtered_records

    def apply_filters(
        self,
        records: List[VectorRecord],
        filters: Dict[str, Any]
    ) -> List[VectorRecord]:
        """
        Apply multiple filters sequentially.
        
        Args:
            records: List of VectorRecord objects
            filters: Dictionary of filter criteria (keys: filter type, values: filter params)
            
        Returns:
            Filtered list of VectorRecord objects
        """
        filtered = records
        
        # Apply date filter
        if 'date' in filters:
            filtered = self.filter_by_date(filtered, **filters['date'])
            
        # Apply entities filter
        if 'entities' in filters:
            filtered = self.filter_by_entities(filtered, **filters['entities'])
            
        # Apply source filter
        if 'source' in filters:
            filtered = self.filter_by_source(filtered, **filters['source'])
            
        return filtered