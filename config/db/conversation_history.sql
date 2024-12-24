-- Drop existing table and indexes if they exist
DROP TABLE IF EXISTS conversation_history;
DROP INDEX IF EXISTS idx_conversation_history_user_id;
DROP INDEX IF EXISTS idx_conversation_history_created_at;

-- Create conversation history table
CREATE TABLE IF NOT EXISTS conversation_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uix_conversation_id UNIQUE (conversation_id)
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_conversation_history_user_id ON conversation_history (user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_history_created_at ON conversation_history (created_at);

-- Add comment to table
COMMENT ON TABLE conversation_history IS 'Table for storing conversation history records';

-- Add comments to columns
COMMENT ON COLUMN conversation_history.id IS 'Unique identifier for the conversation history record';
COMMENT ON COLUMN conversation_history.user_id IS 'Unique identifier for the user';
COMMENT ON COLUMN conversation_history.conversation_id IS 'Unique identifier for the conversation';
COMMENT ON COLUMN conversation_history.message IS 'Text of the conversation message';
COMMENT ON COLUMN conversation_history.created_at IS 'Timestamp when the conversation message was created';
COMMENT ON COLUMN conversation_history.updated_at IS 'Timestamp when the conversation message was last updated';
