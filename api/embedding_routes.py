from enum import Enum

import dotenv
from fastapi import APIRouter, HTTPException, Query, Header, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from config.common_settings import CommonConfig
from preprocess.index_log import SourceType
from preprocess.doc_index_log_processor import DocEmbeddingsProcessor
import os
from pathlib import Path

from preprocess.index_log.index_log_helper import IndexLogHelper
import re
from urllib.parse import unquote

from preprocess.index_log.repositories import IndexLogRepository
from utils.logging_util import logger

from preprocess.loader.loader_factories import DocumentLoaderFactory

router = APIRouter(tags=['pre-process'])

base_config = CommonConfig()
STAGING_PATH = base_config.get_query_config("staging_path")

URL_PATTERN = re.compile(
    r'^https?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


class EmbeddingRequest(BaseModel):
    source: str
    source_type: SourceType


class EmbeddingResponse(BaseModel):
    message: Optional[str]
    source: str
    source_type: str
    id: Optional[str]

class DocumentCategory(Enum):
    FILE = "file"
    WEB_PAGE = "web_page"
    CONFLUENCE = "confluence"


class IndexLogResponse(BaseModel):
    id: str
    source: str
    source_type: str
    status: str
    checksum: str
    created_at: datetime
    created_by: str
    modified_at: datetime
    modified_by: str
    error_message: Optional[str]


@router.post("/docs", response_model=EmbeddingResponse)
def add_document(
        request: EmbeddingRequest,
        x_user_id: str = Header(...)
):
    try:
        doc_processor = DocEmbeddingsProcessor(base_config.get_model("embedding"), base_config.get_vector_store(),
                                               IndexLogHelper(IndexLogRepository(base_config.get_db_manager())))
        result = doc_processor.add_index_log(
            source=request.source,
            source_type=request.source_type,
            user_id=x_user_id
        )
        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/docs/{log_id}")
def get_document_by_id(log_id: str):
    try:
        doc_processor = DocEmbeddingsProcessor(base_config.get_model("embedding"), base_config.get_vector_store(),
                                               IndexLogHelper(IndexLogRepository(base_config.get_db_manager())))
        return doc_processor.get_document_by_id(log_id)
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/docs", response_model=List[IndexLogResponse])
def list_documents(
    page: int = Query(1, gt=0),
    page_size: int = Query(10, gt=0),
    source: Optional[str] = None,
    source_type: Optional[SourceType] = None,
    status: Optional[str] = None,
    created_by: Optional[str] = Query(None, alias="createdBy"),
    from_date: Optional[datetime] = Query(None, alias="fromDate"),
    to_date: Optional[datetime] = Query(None, alias="toDate")
):
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"), 
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager()))
        )
        
        # Build filter conditions
        filters = {}
        if source:
            filters['source'] = source
        if source_type:
            filters['source_type'] = source_type
        if status:
            filters['status'] = status.upper()
        if created_by:
            filters['created_by'] = created_by
        if from_date:
            filters['created_at_from'] = from_date
        if to_date:
            filters['created_at_to'] = to_date
        logger.info(f"Search by filters: {filters}")

        logs = doc_processor.index_log_helper.list_logs(
            page=page,
            page_size=page_size,
            filters=filters
        )
        return logs
    except Exception as e:
        logger.error(e)
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
def upload_document(
    category: DocumentCategory = Form(...),
    file: Optional[UploadFile] = None,
    url: Optional[str] = Form(None),
    x_user_id: str = Header(...)
):
    try:
        logger.info(f"Uploading document with category: {category}")
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"),
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager()))
        )


        if category == DocumentCategory.FILE:
            if not file:
                logger.error("File is required when category is 'file'")
                raise HTTPException(
                    status_code=400,
                    detail="File is required when category is 'file'"
                )

            # Infer source type from file extension
            file_extension = Path(file.filename).suffix.lower()[1:]  # Remove the dot
            source_type = DocumentLoaderFactory.infer_source_type(file_extension)
            logger.info(f"Inferred source type: {source_type}")
            
            if not source_type:
                logger.error(f"Unsupported file type: {file_extension}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}"
                )

            # Create staging directory if it doesn't exist
            staging_path = base_config.get_embedding_config()["staging_path"]
            os.makedirs(staging_path, exist_ok=True)
            
            # Generate staging file path with sanitized filename
            safe_filename = sanitize_filename(file.filename)
            staging_file_path = os.path.join(staging_path, safe_filename)
            
            # Save file to staging
            with open(staging_file_path, "wb") as buffer:
                content = file.read()
                buffer.write(content)

            source = staging_file_path
            source_type = source_type.value  # Convert enum to string

        else:  # WEB_PAGE or CONFLUENCE
            if not url:
                logger.error("URL is required for web page or confluence documents")
                raise HTTPException(
                    status_code=400,
                    detail="URL is required for web page or confluence documents"
                )
            # Add URL validation
            if not URL_PATTERN.match(url):
                logger.error(f"Invalid URL format: {url}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL format. URL must start with http:// or https://"
                )
            source = url
            # Map category to source type
            category_to_source_type = {
                DocumentCategory.WEB_PAGE: SourceType.WEB_PAGE,
                DocumentCategory.CONFLUENCE: SourceType.CONFLUENCE
            }
            source_type = category_to_source_type[category].value

        # Process the document
        result = doc_processor.add_index_log(
            source=source,
            source_type=source_type,
            user_id=x_user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        # Clean up staging file if there's an error
        if 'staging_file_path' in locals() and os.path.exists(staging_file_path):
            os.remove(staging_file_path)
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/docs/{log_id}")
def delete_document(
    log_id: str,
    x_user_id: str = Header(...)
):
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"), 
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager()))
        )
        
        # 1. Check if document exists
        index_log = doc_processor.get_document_by_id(log_id)
        if not index_log:
            raise HTTPException(
                status_code=404,
                detail=f"Document with id {log_id} not found"
            )
            
        # 2. Remove embedded chunks from vector store
        doc_processor._remove_existing_embeddings(
            source=index_log.source,
            source_type=index_log.source_type,
            checksum=index_log.checksum
        )
        
        # 3. Delete index log
        doc_processor.index_log_helper.delete_by_id(log_id)
        
        return {"message": f"Document {log_id} deleted successfully"}
        
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
