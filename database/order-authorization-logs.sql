-- Order Authorization Logs Table
-- This table stores comprehensive audit trail for all order authorization attempts

CREATE TABLE IF NOT EXISTS order_authorization_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    size DECIMAL(20,8) NOT NULL CHECK (size > 0),
    price DECIMAL(20,8),
    order_type VARCHAR(10) NOT NULL CHECK (order_type IN ('market', 'limit')),
    leverage INTEGER NOT NULL CHECK (leverage >= 1 AND leverage <= 100),
    authorized BOOLEAN NOT NULL,
    reason TEXT,
    code VARCHAR(50),
    risk_level VARCHAR(10) CHECK (risk_level IN ('low', 'medium', 'high')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_order_auth_logs_user_id (user_id),
    INDEX idx_order_auth_logs_created_at (created_at),
    INDEX idx_order_auth_logs_authorized (authorized),
    INDEX idx_order_auth_logs_risk_level (risk_level),
    INDEX idx_order_auth_logs_code (code)
);

-- Add RLS policies for security
ALTER TABLE order_authorization_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own authorization logs
CREATE POLICY "Users can view own authorization logs" ON order_authorization_logs
    FOR SELECT USING (auth.uid()::text = user_id::text);

-- Policy: System can insert authorization logs (for audit trail)
CREATE POLICY "System can insert authorization logs" ON order_authorization_logs
    FOR INSERT WITH CHECK (true);

-- Comments for documentation
COMMENT ON TABLE order_authorization_logs IS 'Audit trail for order authorization attempts';
COMMENT ON COLUMN order_authorization_logs.authorized IS 'Whether the order was authorized';
COMMENT ON COLUMN order_authorization_logs.reason IS 'Reason for authorization failure or success';
COMMENT ON COLUMN order_authorization_logs.code IS 'Authorization result code for programmatic handling';
COMMENT ON COLUMN order_authorization_logs.risk_level IS 'Risk level assessment of the order';
