import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# 导入项目中的 AuditLogger
from utils.audit_logger import AuditLogger, AuditLog, Base

# 简单的数据库管理器
class SimpleDBManager:
    def __init__(self):
        # 使用 SQLite 内存数据库
        self.engine = create_engine('sqlite:///:memory:')
        self.Session = sessionmaker(bind=self.engine)
        
        # 创建表
        Base.metadata.create_all(self.engine)
    
    @contextmanager
    def get_session(self):
        """获取数据库会话的上下文管理器"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# 测试函数
def test_audit_logger():
    print("Starting audit logger test...")
    
    # 创建数据库管理器和审计日志记录器
    db_manager = SimpleDBManager()
    audit_logger = AuditLogger(db_manager)
    
    # 测试记录步骤
    print("\nTest 1: Basic logging")
    audit_logger.log_step(
        request_id="test-request",
        user_id="test-user",
        session_id="test-session",
        step="test-step",
        status="START",
        details={"test": "data"}
    )
    
    # 验证记录是否存在
    with db_manager.get_session() as session:
        log = session.query(AuditLog).first()
        assert log is not None, "Log entry not found"
        assert log.request_id == "test-request", f"Expected request_id 'test-request', got '{log.request_id}'"
        assert log.status == "START", f"Expected status 'START', got '{log.status}'"
        details = json.loads(log.details)
        assert details == {"test": "data"}, f"Expected details {{'test': 'data'}}, got {details}"
    print("Test 1 passed!")
    
    # 测试工作流程
    print("\nTest 2: Workflow (start-end)")
    start_time = audit_logger.start_step(
        request_id="workflow-test",
        user_id="test-user",
        session_id="test-session",
        step="workflow-step",
        details={"workflow": "start"}
    )
    
    # 等待一小段时间
    time.sleep(0.1)
    
    # 结束步骤
    audit_logger.end_step(
        request_id="workflow-test",
        user_id="test-user",
        session_id="test-session",
        step="workflow-step",
        start_time=start_time,
        details={"workflow": "end"}
    )
    
    # 验证记录是否存在
    with db_manager.get_session() as session:
        logs = session.query(AuditLog).filter_by(request_id="workflow-test").all()
        assert len(logs) == 2, f"Expected 2 log entries, got {len(logs)}"
        
        # 验证开始记录
        start_logs = [log for log in logs if log.status == "START"]
        assert len(start_logs) == 1, f"Expected 1 START log, got {len(start_logs)}"
        start_log = start_logs[0]
        start_details = json.loads(start_log.details)
        assert start_details == {"workflow": "start"}, f"Expected start details {{'workflow': 'start'}}, got {start_details}"
        
        # 验证结束记录
        end_logs = [log for log in logs if log.status == "END"]
        assert len(end_logs) == 1, f"Expected 1 END log, got {len(end_logs)}"
        end_log = end_logs[0]
        end_details = json.loads(end_log.details)
        assert end_details["workflow"] == "end", f"Expected end_details['workflow'] to be 'end', got '{end_details.get('workflow')}'"
        assert "execution_time" in end_details, "execution_time not found in end details"
        assert end_details["execution_time"] > 0, f"Expected execution_time > 0, got {end_details['execution_time']}"
    print("Test 2 passed!")
    
    # 测试错误记录
    print("\nTest 3: Error logging")
    try:
        raise ValueError("Test error")
    except Exception as e:
        audit_logger.error_step(
            request_id="error-test",
            user_id="test-user",
            session_id="test-session",
            step="error-step",
            error=e
        )
    
    # 验证错误记录是否存在
    with db_manager.get_session() as session:
        log = session.query(AuditLog).filter_by(request_id="error-test").first()
        assert log is not None, "Error log entry not found"
        assert log.status == "ERROR", f"Expected status 'ERROR', got '{log.status}'"
        details = json.loads(log.details)
        assert "error" in details, "error field not found in details"
        assert details["error"] == "Test error", f"Expected error message 'Test error', got '{details['error']}'"
    print("Test 3 passed!")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_audit_logger() 