import os
import json
import numpy as np
from typing import Dict, Any, Optional, List

from utils.logging_util import logger

# Import the registry function (will be defined in qa_management_routes.py)
# This creates a circular import, so we'll use a workaround
_register_qa_matcher = None

class FastQAMatcher:
    """Fast QA matcher using cross-encoder model for semantic similarity"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        
        # Load QA data
        self.qa_data = self._load_qa_data()
        
        # Initialize cross-encoder model
        self.cross_encoder = self._init_cross_encoder()
        
        # Set threshold from config or use default
        self.threshold = 0.7
        try:
            self.threshold = self.config.get_query_config("search.fast_qa_threshold", 0.7)
        except Exception as e:
            self.logger.warning(f"Could not get fast_qa_threshold from config, using default: {self.threshold}")
        
        # Register this instance for reloading
        global _register_qa_matcher
        if _register_qa_matcher is None:
            # Lazy import to avoid circular imports
            try:
                from api.qa_management_routes import register_qa_matcher
                _register_qa_matcher = register_qa_matcher
            except ImportError:
                self.logger.warning("Could not import register_qa_matcher function")
        
        if _register_qa_matcher:
            _register_qa_matcher(self)
    
    def _load_qa_data(self):
        """Load QA data from JSON file"""
        qa_data = []
        try:
            # Get project root directory
            current_file_path = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
            qa_data_path = os.path.join(project_root, "data", "qa_pairs.json")
            
            if os.path.exists(qa_data_path):
                with open(qa_data_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                self.logger.info(f"Loaded {len(qa_data)} QA pairs from {qa_data_path}")
            else:
                self.logger.warning(f"QA data file not found at {qa_data_path}")
        except Exception as e:
            self.logger.error(f"Error loading QA data: {str(e)}")
        
        return qa_data
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder model"""
        cross_encoder = None
        try:
            cross_encoder = self.config.get_model("rerank")
            self.logger.info("Cross-encoder model loaded successfully for fast QA matching")
        except Exception as e:
            self.logger.error(f"Error loading cross-encoder model: {str(e)}")
        
        return cross_encoder
    
    def find_match(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find matching answer for query from QA data
        
        Args:
            query: User query string
            
        Returns:
            Dict with answer and metadata if good match found, None otherwise
        """
        if not self.qa_data or not self.cross_encoder:
            self.logger.warning("QA data or cross-encoder not available for fast matching")
            return None
        
        try:
            # Prepare pairs for scoring
            pairs = [(query, qa_pair["question"]) for qa_pair in self.qa_data]
            
            # Score all pairs
            scores = self.cross_encoder.predict(pairs)
            
            # Find best match
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            if best_score >= self.threshold:
                result = self.qa_data[best_idx].copy()
                result["similarity"] = float(best_score)
                return result
            
            self.logger.info(f"No good match found. Best score: {best_score:.2f}, threshold: {self.threshold}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in fast QA matching: {str(e)}")
            return None