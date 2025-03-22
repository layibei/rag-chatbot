# utils/async_mdc.py

import asyncio
from contextvars import ContextVar, copy_context
from typing import Dict, Any, Optional
from utils.logger_init import logger

class MDCContext:
    """Centralized MDC context management"""
    # Use ContextVar for thread-safe context storage
    _context: ContextVar[Dict[str, Any]] = ContextVar('mdc_context', default={})
    
    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get current MDC context"""
        try:
            return dict(cls._context.get())
        except Exception as e:
            logger.error(f"Error getting MDC context: {str(e)}")
            return {}
    
    @classmethod
    def set_context(cls, **kwargs):
        """Set MDC context values"""
        try:
            current = cls.get_context()
            current.update(kwargs)
            cls._context.set(current)
        except Exception as e:
            logger.error(f"Error setting MDC context: {str(e)}")
    
    @classmethod
    def clear_context(cls):
        """Clear MDC context"""
        try:
            cls._context.set({})
        except Exception as e:
            logger.error(f"Error clearing MDC context: {str(e)}")

class MDCTaskFactory:
    """Task factory that automatically preserves MDC context"""
    
    @staticmethod
    def create_task(loop: asyncio.AbstractEventLoop, coro: Any, *, name: str = None) -> asyncio.Task:
        """Create a new task with MDC context preservation"""
        # Capture current context
        context = MDCContext.get_context()
        
        async def wrapped_coro():
            try:
                # Restore context at task start
                MDCContext.set_context(**context)
                return await coro
            finally:
                # Clean up at task end
                MDCContext.clear_context()
                
        return asyncio.Task(wrapped_coro(), loop=loop, name=name)

def setup_mdc():
    """Initialize MDC handling for the application"""
    try:
        loop = asyncio.get_event_loop()
        loop.set_task_factory(MDCTaskFactory.create_task)
        logger.info("MDC task factory initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MDC task factory: {str(e)}")
        raise