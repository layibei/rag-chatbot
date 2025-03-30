import os
import json
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.logging_util import logger
from utils.audit_logger import get_audit_logger
from config.database.database_manager import DatabaseManager
from config.common_settings import CommonConfig

# Initialize config and audit logger
base_config = CommonConfig()
db_manager = DatabaseManager(base_config.get_db_manager())
audit_logger = get_audit_logger(db_manager)

# Get project root directory
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
qa_data_path = os.path.join(project_root, "data", "qa_pairs.json")

router = APIRouter(tags=["qa_management"])

class QAPair(BaseModel):
    question: str
    answer: str
    category: Optional[str] = None
    citations: Optional[List[str]] = None
    suggested_questions: Optional[List[str]] = None

class QAUploadResponse(BaseModel):
    success: bool
    message: str
    count: int
    backup_path: Optional[str] = None

def validate_qa_pairs(data: List[Dict[str, Any]]) -> bool:
    """Validate the structure of QA pairs data"""
    if not isinstance(data, list):
        return False
    
    for item in data:
        if not isinstance(item, dict):
            return False
        if "question" not in item or "answer" not in item:
            return False
        if not isinstance(item["question"], str) or not isinstance(item["answer"], str):
            return False
    
    return True

def create_backup(file_path: str) -> str:
    """Create a backup of the existing QA pairs file"""
    if not os.path.exists(file_path):
        return None
    
    # Create backups directory if it doesn't exist
    backup_dir = os.path.join(os.path.dirname(file_path), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"qa_pairs_{timestamp}.json"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Copy the file
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup of QA pairs at {backup_path}")
    
    return backup_path

@router.post("/upload-qa-pairs", response_model=QAUploadResponse)
async def upload_qa_pairs(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_user_id: str = Header(...),
    x_session_id: str = Header(default=None),
    x_request_id: str = Header(default=None)
):
    """
    Upload a JSON file to replace the existing QA pairs
    
    The file must be a valid JSON array containing objects with at least 'question' and 'answer' fields.
    """
    # Record request start
    start_time = audit_logger.start_step(
        x_request_id, x_user_id, x_session_id, 
        "upload_qa_pairs", {
            "filename": file.filename,
            "content_type": file.content_type
        }
    )
    
    try:
        # Check file type
        if not file.filename.endswith('.json'):
            error_msg = "Uploaded file must be a JSON file"
            audit_logger.end_step(
                x_request_id, x_user_id, x_session_id, 
                "upload_qa_pairs", start_time, {
                    "error": error_msg,
                    "status": "error"
                }
            )
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read file content
        content = await file.read()
        
        try:
            # Parse JSON
            qa_data = json.loads(content.decode('utf-8'))
            
            # Validate data structure
            if not validate_qa_pairs(qa_data):
                error_msg = "Invalid QA pairs format. Each item must have 'question' and 'answer' fields."
                audit_logger.end_step(
                    x_request_id, x_user_id, x_session_id, 
                    "upload_qa_pairs", start_time, {
                        "error": error_msg,
                        "status": "error"
                    }
                )
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Create backup of existing file
            backup_path = None
            if os.path.exists(qa_data_path):
                backup_path = create_backup(qa_data_path)
            
            # Write new data to file
            with open(qa_data_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, ensure_ascii=False, indent=2)
            
            # Log success
            response = QAUploadResponse(
                success=True,
                message=f"Successfully uploaded {len(qa_data)} QA pairs",
                count=len(qa_data),
                backup_path=backup_path
            )
            
            # Add background task to reload QA data in FastQAMatcher instances
            background_tasks.add_task(reload_qa_matchers)
            
            # Record request end
            audit_logger.end_step(
                x_request_id, x_user_id, x_session_id, 
                "upload_qa_pairs", start_time, {
                    "qa_count": len(qa_data),
                    "backup_created": backup_path is not None,
                    "status": "success"
                }
            )
            
            return response
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON format"
            audit_logger.end_step(
                x_request_id, x_user_id, x_session_id, 
                "upload_qa_pairs", start_time, {
                    "error": error_msg,
                    "status": "error"
                }
            )
            raise HTTPException(status_code=400, detail=error_msg)
            
    except Exception as e:
        # Log error
        error_msg = f"Error uploading QA pairs: {str(e)}"
        logger.error(error_msg)
        audit_logger.error_step(
            x_request_id, x_user_id, x_session_id, 
            "upload_qa_pairs", e, {
                "status": "error"
            }
        )
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/qa-pairs", response_model=List[QAPair])
async def get_qa_pairs(
    x_user_id: str = Header(...),
    x_session_id: str = Header(default=None),
    x_request_id: str = Header(default=None)
):
    """Get the current QA pairs"""
    # Record request start
    start_time = audit_logger.start_step(
        x_request_id, x_user_id, x_session_id, 
        "get_qa_pairs", {}
    )
    
    try:
        if not os.path.exists(qa_data_path):
            # Return empty list if file doesn't exist
            audit_logger.end_step(
                x_request_id, x_user_id, x_session_id, 
                "get_qa_pairs", start_time, {
                    "status": "success",
                    "qa_count": 0,
                    "file_exists": False
                }
            )
            return []
        
        with open(qa_data_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # Record request end
        audit_logger.end_step(
            x_request_id, x_user_id, x_session_id, 
            "get_qa_pairs", start_time, {
                "status": "success",
                "qa_count": len(qa_data)
            }
        )
        
        return qa_data
        
    except Exception as e:
        # Log error
        error_msg = f"Error retrieving QA pairs: {str(e)}"
        logger.error(error_msg)
        audit_logger.error_step(
            x_request_id, x_user_id, x_session_id, 
            "get_qa_pairs", e, {
                "status": "error"
            }
        )
        raise HTTPException(status_code=500, detail=error_msg)

# Global registry of FastQAMatcher instances
_qa_matcher_registry = []

def register_qa_matcher(matcher):
    """Register a FastQAMatcher instance for reloading"""
    global _qa_matcher_registry
    if matcher not in _qa_matcher_registry:
        _qa_matcher_registry.append(matcher)
        logger.debug(f"Registered QA matcher, total registered: {len(_qa_matcher_registry)}")

def reload_qa_matchers():
    """Reload QA data in all registered FastQAMatcher instances"""
    global _qa_matcher_registry
    logger.info(f"Reloading QA data in {len(_qa_matcher_registry)} matcher instances")
    
    for matcher in _qa_matcher_registry:
        try:
            # Reload QA data
            matcher.qa_data = matcher._load_qa_data()
            logger.info(f"Reloaded QA data with {len(matcher.qa_data)} pairs")
        except Exception as e:
            logger.error(f"Error reloading QA data: {str(e)}") 