import os
import sys
import unittest
import dotenv

from langchain_community.embeddings import SparkLLMTextEmbeddings
from langchain_qdrant import QdrantVectorStore

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

dotenv.load_dotenv(dotenv_path=os.path.join(project_root, '.env'))


class TestDocumentLoader(unittest.TestCase):
    def test_load_document_json(self):
        from loader.loader_factories import DocumentLoaderFactory
        document_loader = DocumentLoaderFactory.get_loader('../data/weather.json')
        documents = document_loader.load('../data/weather.json')
        self.assertIsNotNone(documents)

    def test_load_document_pdf(self):
        from loader.loader_factories import DocumentLoaderFactory
        document_loader = DocumentLoaderFactory.get_loader('../data/hotcloud16_burns.pdf')
        documents = document_loader.load('../data/hotcloud16_burns.pdf')

        embeddings = SparkLLMTextEmbeddings()
        QdrantVectorStore.from_documents(
            documents,
            embeddings,
            url="http://localhost:6333",
            collection_name="rag_docs",
        )

        self.assertIsNotNone(documents)

    def test_load_document_csv(self):
        from loader.loader_factories import DocumentLoaderFactory
        document_loader = DocumentLoaderFactory.get_loader('../data/house-price.csv')
        documents = document_loader.load('../data/house-price.csv')
        self.assertIsNotNone(documents)
