import unittest
from unittest.mock import MagicMock
from doc_embeddings import DocEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class TestDocEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embeddings_model = HuggingFaceEmbeddings()

    def test_embedding_flow(self):

        doc_embeddings = DocEmbeddings(self.embeddings_model)

        doc_embeddings.use_in_context()
        self.mock_embeddings.assert_called_once()

        # 测试vectorstore的add_texts方法是否被调用
        # 这里只是示例，具体调用方式需要根据实际的DocEmbeddings类定义来修改
        doc_embeddings.add_texts(["test text"])
        self.mock_vectorstore.add_texts.assert_called_once_with(["test text"])

        # 测试vectorstore的search方法，比如假设search_by_text是其中一个方法
        # 这里只是示例，具体调用方式需要根据实际的DocEmbeddings类定义来修改
        doc_embeddings.search_by_text("search text")
        self.mock_vectorstore.search_by_text.assert_called_once_with("search text")

    # 可以继续添加更多的测试用例，以测试不同的功能和边界条件


if __name__ == '__main__':
    unittest.main()