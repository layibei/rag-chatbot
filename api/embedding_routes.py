import dotenv
from fastapi import APIRouter, HTTPException, Query, Header, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from config.common_settings import CommonConfig
from config.database.database_manager import DatabaseManager
from preprocess.index_log import SourceType
from preprocess.doc_index_log_processor import DocEmbeddingsProcessor
import os
from pathlib import Path

from preprocess.index_log.index_log_helper import IndexLogHelper
import re
from urllib.parse import unquote

from preprocess.index_log.repositories import IndexLogRepository

router = APIRouter()

dotenv.load_dotenv(dotenv_path=os.getcwd()+"/.env")
db_manager = DatabaseManager(os.environ["POSTGRES_URI"])

base_config = CommonConfig()
doc_processor = DocEmbeddingsProcessor(base_config.get_model("embedding"), base_config.get_vector_store(),
                                       IndexLogHelper(IndexLogRepository(db_manager)))

STAGING_PATH = "./data/staging"  # Configure this in your settings


class EmbeddingRequest(BaseModel):
    source: str
    source_type: SourceType


class EmbeddingResponse(BaseModel):
    message: Optional[str]
    source: str
    source_type: str
    id: Optional[int]


class IndexLogResponse(BaseModel):
    id: int
    source: str
    source_type: str
    status: str
    created_at: datetime
    created_by: str
    modified_at: datetime
    modified_by: str
    error_message: Optional[str]


@router.post("/docs", response_model=EmbeddingResponse)
async def add_document(
        request: EmbeddingRequest,
        user_id: str = Header(..., alias="user-id")
):
    try:
        result = doc_processor.add_index_log(
            source=request.source,
            source_type=request.source_type,
            user_id=user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/docs/{log_id}")
async def get_document_by_id(log_id: int):
    try:
        return doc_processor.get_document_by_id(log_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/docs", response_model=List[IndexLogResponse])
async def list_documents(
        page: int = Query(1, gt=0),
        page_size: int = Query(10, gt=0),
        search: Optional[str] = None
):
    try:
        logs = doc_processor.index_log_helper.list_logs(
            page=page,
            page_size=page_size,
            search=search
        )
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by:
    1. Decoding URL-encoded characters
    2. Removing or replacing special characters
    3. Replacing spaces with underscores
    """
    # Decode URL-encoded characters
    filename = unquote(filename)
    
    # Remove or replace special characters, keeping only alphanumeric, dots, dashes and underscores
    filename = re.sub(r'[^\w\-\.]', '_', filename)
    
    # Replace multiple underscores with single underscore
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    return filename


@router.post("/docs/upload", response_model=EmbeddingResponse)
async def upload_document(
    file: UploadFile = File(...),
    source_type: SourceType = Query(...),
    user_id: str = Header(..., alias="user-id")
):
    try:
        # 1. Validate file extension matches source_type
        file_extension = Path(file.filename).suffix.lower()
        if not _is_valid_extension(file_extension[1:], source_type):  # Remove the dot
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{file_extension}' does not match source type '{source_type}'"
            )

        # 2. Create staging directory if it doesn't exist
        os.makedirs(STAGING_PATH, exist_ok=True)
        
        # 3. Generate staging file path with sanitized filename
        safe_filename = sanitize_filename(file.filename)
        staging_file_path = os.path.join(STAGING_PATH, safe_filename)
        
        # 4. Save file to staging
        with open(staging_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 5. Process the document
        result = doc_processor.add_index_log(
            source=staging_file_path,
            source_type=source_type,
            user_id=user_id
        )
        
        return result
        
    except Exception as e:
        # Clean up staging file if there's an error
        if 'staging_file_path' in locals() and os.path.exists(staging_file_path):
            os.remove(staging_file_path)
        raise HTTPException(status_code=500, detail=str(e))


def _is_valid_extension(extension: str, source_type: SourceType) -> bool:
    extension_mapping = {
        SourceType.PDF: ['pdf'],
        SourceType.DOCX: ['docx'],
        SourceType.CSV: ['csv'],
        SourceType.JSON: ['json'],
        SourceType.TEXT: ['txt'],
    }
    return extension in extension_mapping.get(source_type, [])
