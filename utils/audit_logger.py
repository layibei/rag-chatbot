import json
import time
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
    """审计日志记录器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._ensure_table_exists()
    
    @staticmethod
    def initialize_tables(engine):
        """初始化审计日志表"""
        inspector = inspect(engine)
        if not inspector.has_table(AuditLog.__tablename__):
            logger.info(f"Creating audit log table: {AuditLog.__tablename__}")
            Base.metadata.create_all(engine, tables=[AuditLog.__table__])
            logger.info("Audit log table created successfully")
        else:
            logger.info(f"Audit log table already exists: {AuditLog.__tablename__}")
    
    def _ensure_table_exists(self):
        """确保审计日志表存在"""
        try:
            AuditLogger.initialize_tables(self.db_manager.engine)
        except Exception as e:
            logger.error(f"Failed to initialize audit log table: {e}")
    
    @classmethod
    def init_database(cls, config):
        """初始化数据库表 - 应用启动时调用"""
        try:
            db_manager = DatabaseManager(config.get_postgres_uri())
            # 初始化审计日志表
            cls.initialize_tables(db_manager.engine)
            logger.info("Audit log database tables initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audit log database: {e}")
            return False
    
    def log_step(self, request_id: str, user_id: str, session_id: str, 
                step: str, status: str, details: Optional[Dict[str, Any]] = None):
        """记录步骤到审计日志"""
        try:
            with self.db_manager.get_session() as session:
                log_entry = AuditLog(
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id,
                    step=step,
                    status=status,
                    details=json.dumps(details) if details else None
                )
                session.add(log_entry)
                session.commit()
                logger.debug(f"Audit log recorded: {step} - {status}")
        except Exception as e:
            logger.error(f"Error logging to audit table: {e}")
    
    def start_step(self, request_id: str, user_id: str, session_id: str, 
                  step: str, details: Optional[Dict[str, Any]] = None):
        """记录步骤开始"""
        self.log_step(request_id, user_id, session_id, step, "START", details)
        return time.time()  # 返回开始时间，用于计算执行时间
    
    def end_step(self, request_id: str, user_id: str, session_id: str, 
                step: str, start_time: float, details: Optional[Dict[str, Any]] = None):
        """记录步骤结束"""
        execution_time = time.time() - start_time
        if details is None:
            details = {}
        details["execution_time"] = execution_time
        self.log_step(request_id, user_id, session_id, step, "END", details)
    
    def error_step(self, request_id: str, user_id: str, session_id: str, 
                  step: str, error: Exception, details: Optional[Dict[str, Any]] = None):
        """记录步骤错误"""
        if details is None:
            details = {}
        details["error"] = str(error)
        self.log_step(request_id, user_id, session_id, step, "ERROR", details)


# 单例模式，全局审计日志记录器
_audit_logger = None

def get_audit_logger(db_manager: DatabaseManager = None) -> AuditLogger:
    """获取审计日志记录器实例"""
    global _audit_logger
    if _audit_logger is None and db_manager is not None:
        _audit_logger = AuditLogger(db_manager)
    return _audit_logger 