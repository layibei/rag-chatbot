from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()
class IndexLog(Base):
    __tablename__ = 'index_logs'

    id = Column(Integer, primary_key=True)
    source = Column(String, unique=True)
    indexed_time = Column(DateTime)
    indexed_by = Column(String)
    modified_time = Column(DateTime)
    modified_by = Column(String)
    checksum = Column(String, unique=True)