-- Audit Logs Table
-- This table stores comprehensive audit trail for all operations

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    order_id UUID REFERENCES orders(id) ON DELETE SET NULL,
    position_id UUID REFERENCES positions(id) ON DELETE SET NULL,
    transaction_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    details JSONB NOT NULL,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_audit_logs_event_type (event_type),
    INDEX idx_audit_logs_user_id (user_id),
    INDEX idx_audit_logs_order_id (order_id),
    INDEX idx_audit_logs_position_id (position_id),
    INDEX idx_audit_logs_created_at (created_at),
    INDEX idx_audit_logs_action (action),
    INDEX idx_audit_logs_session_id (session_id),
    INDEX idx_audit_logs_user_event_type (user_id, event_type),
    INDEX idx_audit_logs_created_at_event_type (created_at, event_type)
);

-- Add RLS policies for security
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own audit logs
CREATE POLICY "Users can view own audit logs" ON audit_logs
    FOR SELECT USING (auth.uid()::text = user_id::text);

-- Policy: System can manage audit logs
CREATE POLICY "System can manage audit logs" ON audit_logs
    FOR ALL USING (true);

-- Comments for documentation
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for all operations';
COMMENT ON COLUMN audit_logs.event_type IS 'Type of event (order_placement, order_execution, etc.)';
COMMENT ON COLUMN audit_logs.action IS 'Specific action performed';
COMMENT ON COLUMN audit_logs.details IS 'Additional event details in JSON format';
COMMENT ON COLUMN audit_logs.ip_address IS 'IP address of the request';
COMMENT ON COLUMN audit_logs.user_agent IS 'User agent string';
COMMENT ON COLUMN audit_logs.session_id IS 'Session identifier';
