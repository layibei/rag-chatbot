from sqlalchemy import Column, String, DateTime, create_engine, Integer
from sqlalchemy.orm import declarative_base, sessionmaker

class IndexLog():
    __tablename__ = 'index_logs'

    id = Column(Integer, primary_key=True)
    source = Column(String, unique=True)
    indexed_time = Column(DateTime)
    indexed_by = Column(String)
    modified_time = Column(DateTime)
    modified_by = Column(String)
    checksum = Column(String, unique=True)