from sqlalchemy import Column, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from datetime import datetime, UTC
from sqlalchemy.exc import IntegrityError

Base = declarative_base()

class DistributedLock(Base):
    __tablename__ = 'distributed_locks'
    
    id = Column(String(255), primary_key=True)
    lock_key = Column(String(255), unique=True, nullable=False)
    instance_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False)

class DistributedLockHelper:
    def __init__(self, session_factory, instance_name):
        self.session_factory = session_factory
        self.instance_name = instance_name

    def acquire_lock(self, lock_key: str) -> bool:
        session = self.session_factory()
        try:
            lock = DistributedLock(
                id=f"{lock_key}_{datetime.now(UTC).timestamp()}",
                lock_key=lock_key,
                instance_name=self.instance_name,
                created_at=datetime.now(UTC)
            )
            session.add(lock)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False
        finally:
            session.close()

    def release_lock(self, lock_key: str):
        session = self.session_factory()
        try:
            session.query(DistributedLock).filter(
                DistributedLock.lock_key == lock_key,
                DistributedLock.instance_name == self.instance_name
            ).delete()
            session.commit()
        finally:
            session.close() 