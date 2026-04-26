# Tests for Verification Module

This directory contains tests for the verification module components.

## Running Tests

To run all tests:

```bash
pytest tests/ -v
```

To run tests for a specific component:

```bash
pytest tests/verification/test_nli_classifier.py -v
```

## Test Structure

- `test_nli_classifier.py` - Tests for NLI classifier
- `test_metadata_filter.py` - Tests for metadata filtering
- `test_result_aggregator.py` - Tests for result aggregation
- `test_verification_service.py` - Tests for the main verification service
- `test_report_generator.py` - Tests for report generation

## Mocking

The tests use mocking extensively to isolate components:
- Vector store interactions are mocked
- NLI model calls are mocked using unittest.mock.patch
- External dependencies are simulated

## Requirements

- pytest
- pytest-asyncio (for async tests)
- torch
- transformers

Install test dependencies with:
```bash
pip install pytest pytest-asyncio torch transformers
```