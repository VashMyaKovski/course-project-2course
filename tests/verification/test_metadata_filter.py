import pytest
from datetime import datetime
from src.verification.metadata_filter import MetadataFilter
from src.verification.vector_store import VectorRecord
import numpy as np


@pytest.fixture
def sample_records():
    """Create sample VectorRecord objects for testing."""
    return [
        VectorRecord(
            id="1",
            text="First record",
            embedding=np.array([0.1] * 768),
            metadata={"date": "2023-01-15", "entities": ["Apple", "iPhone"], "source": "TechNews"}
        ),
        VectorRecord(
            id="2",
            text="Second record",
            embedding=np.array([0.2] * 768),
            metadata={"date": "2023-05-20", "entities": ["Google", "Android"], "source": "TechBlog"}
        ),
        VectorRecord(
            id="3",
            text="Third record",
            embedding=np.array([0.3] * 768),
            metadata={"date": "2022-12-01", "entities": ["Microsoft", "Windows"], "source": "NewsDaily"}
        )
    ]


def test_filter_by_date_inclusive(sample_records):
    """Test date filtering with inclusive ranges."""
    filter_obj = MetadataFilter()
    
    # Filter records from 2023-01-01 to 2023-12-31
    filtered = filter_obj.filter_by_date(
        sample_records, 
        start_date="2023-01-01", 
        end_date="2023-12-31"
    )
    
    # Should include records 1 and 2 (Jan and May 2023), exclude record 3 (Dec 2022)
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "2"}


def test_filter_by_date_start_only(sample_records):
    """Test date filtering with only start date."""
    filter_obj = MetadataFilter()
    
    # Filter records from 2023-01-01 onwards
    filtered = filter_obj.filter_by_date(
        sample_records, 
        start_date="2023-01-01"
    )
    
    # Should include records 1 and 2 (both in 2023), exclude record 3 (2022)
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "2"}


def test_filter_by_date_end_only(sample_records):
    """Test date filtering with only end date."""
    filter_obj = MetadataFilter()
    
    # Filter records up to 2023-03-01
    filtered = filter_obj.filter_by_date(
        sample_records, 
        end_date="2023-03-01"
    )
    
    # Should include record 1 (Jan 15, 2023) and record 3 (Dec 1, 2022), exclude record 2 (May 20, 2023)
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "3"}


def test_filter_by_date_no_filters(sample_records):
    """Test date filtering with no filters returns all records."""
    filter_obj = MetadataFilter()
    
    filtered = filter_obj.filter_by_date(sample_records)
    assert len(filtered) == 3
    assert {r.id for r in filtered} == {"1", "2", "3"}


def test_filter_by_date_invalid_date(sample_records):
    """Test date filtering with invalid date format."""
    filter_obj = MetadataFilter()
    
    # Create a record with invalid date
    invalid_record = VectorRecord(
        id="4",
        text="Invalid date record",
        embedding=np.array([0.4] * 768),
        metadata={"date": "not-a-date"}
    )
    
    records_with_invalid = sample_records + [invalid_record]
    filtered = filter_obj.filter_by_date(records_with_invalid, start_date="2023-01-01")
    
    # Should still return the valid records, skipping the invalid one
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "2"}


def test_filter_by_entities_any(sample_records):
    """Test entity filtering with match_any (default)."""
    filter_obj = MetadataFilter()
    
    # Filter for records containing either "Apple" or "Google"
    filtered = filter_obj.filter_by_entities(
        sample_records, 
        entities=["Apple", "Google"]
    )
    
    # Should include record 1 (has Apple) and record 2 (has Google)
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "2"}


def test_filter_by_entities_all(sample_records):
    """Test entity filtering with match_all=True."""
    filter_obj = MetadataFilter()
    
    # Filter for records containing BOTH "Apple" and "iPhone"
    filtered = filter_obj.filter_by_entities(
        sample_records, 
        entities=["Apple", "iPhone"],
        match_all=True
    )
    
    # Should include only record 1 (has both Apple and iPhone)
    assert len(filtered) == 1
    assert filtered[0].id == "1"


def test_filter_by_entities_empty_list(sample_records):
    """Test entity filtering with empty entity list returns all records."""
    filter_obj = MetadataFilter()
    
    filtered = filter_obj.filter_by_entities(sample_records, entities=[])
    assert len(filtered) == 3
    assert {r.id for r in filtered} == {"1", "2", "3"}


def test_filter_by_source_exact(sample_records):
    """Test source filtering with exact match."""
    filter_obj = MetadataFilter()
    
    # Filter for records from "TechNews" or "TechBlog"
    filtered = filter_obj.filter_by_source(
        sample_records, 
        sources=["TechNews", "TechBlog"]
    )
    
    # Should include record 1 (TechNews) and record 2 (TechBlog)
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "2"}


def test_filter_by_source_partial(sample_records):
    """Test source filtering with partial match."""
    filter_obj = MetadataFilter()
    
    # Filter for records containing "News" in source (partial match)
    filtered = filter_obj.filter_by_source(
        sample_records, 
        sources=["News"],
        exact_match=False
    )
    
    # Should include record 1 (TechNews contains "News") and record 3 (NewsDaily contains "News")
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "3"}


def test_filter_by_source_empty_list(sample_records):
    """Test source filtering with empty source list returns all records."""
    filter_obj = MetadataFilter()
    
    filtered = filter_obj.filter_by_source(sample_records, sources=[])
    assert len(filtered) == 3
    assert {r.id for r in filtered} == {"1", "2", "3"}


def test_apply_filters_combined(sample_records):
    """Test applying multiple filters combined."""
    filter_obj = MetadataFilter()
    
    filters = {
        "date": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "entities": {"entities": ["Apple"], "match_all": False},
        "source": {"sources": ["TechNews"], "exact_match": True}
    }
    
    filtered = filter_obj.apply_filters(sample_records, filters)
    
    # Should include only record 1:
    # - Date: Jan 15, 2023 is within 2023
    # - Entities: contains "Apple" 
    # - Source: exactly "TechNews"
    assert len(filtered) == 1
    assert filtered[0].id == "1"


def test_apply_filters_partial(sample_records):
    """Test applying only some filters."""
    filter_obj = MetadataFilter()
    
    filters = {
        "date": {"start_date": "2023-01-01"},
        # No entities or source filters
    }
    
    filtered = filter_obj.apply_filters(sample_records, filters)
    
    # Should include records 1 and 2 (both from 2023)
    assert len(filtered) == 2
    assert {r.id for r in filtered} == {"1", "2"}