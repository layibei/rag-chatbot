CREATE TABLE IF NOT EXISTS llm_interactions (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    model VARCHAR(255) NOT NULL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    tokens_total INTEGER,
    latency_ms INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Indexes for querying
    CONSTRAINT llm_interactions_request_id_idx UNIQUE (request_id),
    INDEX llm_interactions_user_session_idx (user_id, session_id),
    INDEX llm_interactions_created_at_idx (created_at)
); 