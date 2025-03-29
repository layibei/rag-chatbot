import json
import time
import asyncio
import threading
from queue import Queue
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Integer, Text, inspect
from sqlalchemy.ext.declarative import declarative_base
import datetime
from config.database.database_manager import DatabaseManager
from utils.logging_util import logger

Base = declarative_base()

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), nullable=False, index=True)
    step = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # START, END, ERROR
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    details = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<AuditLog(request_id='{self.request_id}', step='{self.step}', status='{self.status}')>"


class AuditLogger:
    """Audit Logger for tracking workflow steps and performance with async writing"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._ensure_table_exists()
        
        # Create log queue
        self.log_queue = Queue()
        
        # Flag for shutdown state - ensure this is initialized
        self.shutting_down = False
        
        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audit logger background worker started")
    
    @staticmethod
    def initialize_tables(engine):
        """Initialize audit log tables"""
        inspector = inspect(engine)
        if not inspector.has_table(AuditLog.__tablename__):
            logger.info(f"Creating audit log table: {AuditLog.__tablename__}")
            Base.metadata.create_all(engine, tables=[AuditLog.__table__])
            logger.info("Audit log table created successfully")
        else:
            logger.info(f"Audit log table already exists: {AuditLog.__tablename__}")
    
    def _ensure_table_exists(self):
        """Ensure audit log table exists"""
        try:
            AuditLogger.initialize_tables(self.db_manager.engine)
        except Exception as e:
            logger.error(f"Failed to initialize audit log table: {e}")
    
    @classmethod
    def init_database(cls, config):
        """Initialize database tables - called on application startup"""
        try:
            # Get the database URI instead of the manager
            postgres_uri = config.get_db_manager()
            
            # Create a database manager with the URI
            db_manager = DatabaseManager(postgres_uri)
            
            # Initialize audit log tables
            cls.initialize_tables(db_manager.engine)
            logger.info("Audit log database tables initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audit log database: {e}")
            return False
    
    def _log_worker(self):
        """Background worker thread responsible for writing logs to database"""
        logger.info("Audit log worker thread started")
        
        while not self.shutting_down or not self.log_queue.empty():
            try:
                # Get log entry, wait up to 1 second
                try:
                    log_entry = self.log_queue.get(timeout=1)
                except Exception:
                    # Queue is empty, continue waiting
                    continue
                
                # Write to database
                with self.db_manager.session() as session:
                    session.add(log_entry)
                    session.commit()
                
                # Mark task as done
                self.log_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audit log worker: {e}")
                # Short pause after error to avoid high CPU usage
                time.sleep(0.1)
    
    def log_step(self, request_id: str, user_id: str, session_id: str, 
                step: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Asynchronously log a step"""
        try:
            # Create log entry
            log_entry = AuditLog(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                step=step,
                status=status,
                details=json.dumps(details) if details else None
            )
            
            # Add log entry to queue
            self.log_queue.put(log_entry)
            logger.debug(f"Audit log queued: {step} - {status}")
            
        except Exception as e:
            logger.error(f"Error queueing audit log: {e}")
    
    def start_step(self, request_id: str, user_id: str, session_id: str, 
                  step: str, details: Optional[Dict[str, Any]] = None):
        """Log step start"""
        self.log_step(request_id, user_id, session_id, step, "START", details)
        return time.time()  # Return start time for calculating execution time
    
    def end_step(self, request_id: str, user_id: str, session_id: str, 
                step: str, start_time: float, details: Optional[Dict[str, Any]] = None):
        """Log step end"""
        execution_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        if details is None:
            details = {}
        
        # Put execution time at the beginning of details
        execution_details = {
            "execution_time_ms": execution_time
        }
        execution_details.update(details)  # Add other details
        
        self.log_step(request_id, user_id, session_id, step, "END", execution_details)
    
    def error_step(self, request_id: str, user_id: str, session_id: str, 
                  step: str, error: Exception, details: Optional[Dict[str, Any]] = None):
        """Log step error"""
        if details is None:
            details = {}
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        error_details.update(details)
        
        self.log_step(request_id, user_id, session_id, step, "ERROR", error_details)
    
    def shutdown(self):
        """Shutdown the audit logger and wait for all logs to be written"""
        logger.info("Shutting down audit logger...")
        self.shutting_down = True
        
        # Wait for all queued logs to be written
        if self.log_queue.qsize() > 0:
            logger.info(f"Waiting for {self.log_queue.qsize()} audit logs to be written...")
            self.log_queue.join()
        
        logger.info("Audit logger shutdown complete")


# Singleton pattern for global audit logger
_audit_logger = None

def get_audit_logger(db_manager: DatabaseManager = None) -> AuditLogger:
    """Get audit logger instance"""
    global _audit_logger
    if _audit_logger is None and db_manager is not None:
        _audit_logger = AuditLogger(db_manager)
    return _audit_logger 