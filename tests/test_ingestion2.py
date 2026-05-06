from src.ingestion.docs_parser import DocumentParser, parse_document
from src.ingestion.pipeline import IngestionPipeline

if __name__ == '__main__':
    result = DocumentParser().parse('tests/doc.txt')
    print(result)