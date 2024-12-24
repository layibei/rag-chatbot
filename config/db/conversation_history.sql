-- Drop existing table and indexes if they exist
DROP TABLE IF EXISTS conversation_history;
DROP INDEX IF EXISTS idx_conversation_history_user_id;
DROP INDEX IF EXISTS idx_conversation_history_created_at;

-- Create conversation history table
CREATE TABLE IF NOT EXISTS conversation_history (
    id VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    request_id VARCHAR(128) NOT NULL,
    user_input TEXT NOT NULL,
    response TEXT NOT NULL,
    liked BOOLEAN,
    token_usage JSONB,
    created_at TIMESTAMP NOT NULL,
    modified_at TIMESTAMP NOT NULL,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_by VARCHAR(128) NOT NULL,
    modified_by VARCHAR(128) NOT NULL
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_conversation_history_user_id ON conversation_history (user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_session ON conversation_history (user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_request ON conversation_history (user_id, session_id, request_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_created_at ON conversation_history (created_at);

-- Add comment to table
COMMENT ON TABLE conversation_history IS 'Table for storing conversation history records';

-- Add comments to columns
COMMENT ON COLUMN conversation_history.id IS 'Unique identifier for the conversation history record';
COMMENT ON COLUMN conversation_history.user_id IS 'Identifier of the user who created the conversation';
COMMENT ON COLUMN conversation_history.session_id IS 'Identifier for grouping conversations into sessions';
COMMENT ON COLUMN conversation_history.request_id IS 'Unique identifier for each request-response pair';
COMMENT ON COLUMN conversation_history.user_input IS 'The input/question from the user';
COMMENT ON COLUMN conversation_history.response IS 'The response from the system';
COMMENT ON COLUMN conversation_history.liked IS 'Whether the user liked/found helpful this response';
COMMENT ON COLUMN conversation_history.token_usage IS 'JSON containing token usage statistics';
COMMENT ON COLUMN conversation_history.created_at IS 'Timestamp when the record was created';
COMMENT ON COLUMN conversation_history.modified_at IS 'Timestamp when the record was last modified';
COMMENT ON COLUMN conversation_history.is_deleted IS 'Soft delete flag';
COMMENT ON COLUMN conversation_history.created_by IS 'User who created the record';
COMMENT ON COLUMN conversation_history.modified_by IS 'User who last modified the record';
