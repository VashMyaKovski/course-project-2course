from loguru import logger

from src.ingestion.docs_parser import DocumentParser, IngestedDocument


class IngestionPipeline:
    """Load an input document and return normalized text payload."""

    def __init__(self, parser: DocumentParser | None = None):
        self.parser = parser or DocumentParser()

    def run(self, file_path: str) -> IngestedDocument:
        logger.info(f"Ingestion started for file: {file_path}")
        document = self.parser.parse(file_path)
        _, metadata = next(iter(document.items()))
        logger.info(
            f"Ingestion finished: uuid={metadata['uuid']}, "
            f"chars={metadata['content_length']}"
        )
        return document
