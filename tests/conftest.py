import os
import pytest
from sqlalchemy import URL
import dotenv

from config.database.database_manager import DatabaseManager
from preprocess.index_log import Base as IndexLogBase
from conversation import Base as ConversationBase

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables for testing"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    dotenv.load_dotenv(env_path)

@pytest.fixture
def db_manager():
    """Create a database manager with in-memory SQLite for testing"""
    url = URL.create("sqlite", database=":memory:")
    manager = DatabaseManager(url)
    
    # Create all tables
    IndexLogBase.metadata.create_all(manager.engine)
    ConversationBase.metadata.create_all(manager.engine)

    return manager 