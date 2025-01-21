from datetime import datetime, UTC
from typing import Dict, Any, Optional
import json
from utils.logging_util import logger

class QueryMetricsCollector:
    """Simple metrics collector with JSON logging for analytics"""
    
    def __init__(self):
        self.logger = logger
        
    def record_query_event(self, 
                          event_type: str,
                          user_id: str, 
                          session_id: str, 
                          request_id: str,
                          metadata: Dict[str, Any] = None):
        """Record a query event with full context"""
        try:
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "metadata": metadata or {}
            }
            
            self.logger.info("Query event", extra={"query_metrics": event})
            return event
            
        except Exception as e:
            self.logger.error(f"Error recording metrics: {str(e)}")
            return None

    def record_step_timing(self,
                          step_name: str,
                          start_time: datetime,
                          user_id: str,
                          session_id: str,
                          request_id: str,
                          metadata: Dict[str, Any] = None):
        """Record timing for workflow steps"""
        duration = (datetime.now(UTC) - start_time).total_seconds()
        
        return self.record_query_event(
            event_type="step_completion",
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata={
                "step_name": step_name,
                "duration_seconds": duration,
                **(metadata or {})
            }
        ) 