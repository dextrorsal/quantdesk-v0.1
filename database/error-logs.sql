-- Error Logs Table
-- This table stores comprehensive error information for monitoring and recovery

CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    error_stack TEXT,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    order_id UUID REFERENCES orders(id) ON DELETE SET NULL,
    position_id UUID REFERENCES positions(id) ON DELETE SET NULL,
    transaction_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_error_logs_operation (operation),
    INDEX idx_error_logs_created_at (created_at),
    INDEX idx_error_logs_user_id (user_id),
    INDEX idx_error_logs_order_id (order_id),
    INDEX idx_error_logs_position_id (position_id)
);

-- Add RLS policies for security
ALTER TABLE error_logs ENABLE ROW LEVEL SECURITY;

-- Policy: System can manage error logs
CREATE POLICY "System can manage error logs" ON error_logs
    FOR ALL USING (true);

-- Comments for documentation
COMMENT ON TABLE error_logs IS 'Comprehensive error logging for monitoring and recovery';
COMMENT ON COLUMN error_logs.operation IS 'Name of the operation that failed';
COMMENT ON COLUMN error_logs.error_message IS 'Error message';
COMMENT ON COLUMN error_logs.error_stack IS 'Full error stack trace';
COMMENT ON COLUMN error_logs.metadata IS 'Additional context data';
