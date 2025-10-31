-- Rate Limit Logs Table
-- This table stores rate limiting information for API endpoints

CREATE TABLE IF NOT EXISTS rate_limit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(255) NOT NULL, -- Rate limit key (user ID or IP address)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_rate_limit_logs_key (key),
    INDEX idx_rate_limit_logs_created_at (created_at),
    INDEX idx_rate_limit_logs_key_created_at (key, created_at)
);

-- Add RLS policies for security
ALTER TABLE rate_limit_logs ENABLE ROW LEVEL SECURITY;

-- Policy: System can manage rate limit logs
CREATE POLICY "System can manage rate limit logs" ON rate_limit_logs
    FOR ALL USING (true);

-- Comments for documentation
COMMENT ON TABLE rate_limit_logs IS 'Rate limiting logs for API endpoints';
COMMENT ON COLUMN rate_limit_logs.key IS 'Rate limit key (user ID or IP address)';
COMMENT ON COLUMN rate_limit_logs.created_at IS 'Timestamp when the request was made';
