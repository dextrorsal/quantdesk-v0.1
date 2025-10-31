-- Performance Metrics Table
-- This table stores performance monitoring data for all operations

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation VARCHAR(100) NOT NULL,
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Indexes for performance
    INDEX idx_performance_metrics_operation (operation),
    INDEX idx_performance_metrics_timestamp (timestamp),
    INDEX idx_performance_metrics_operation_timestamp (operation, timestamp),
    INDEX idx_performance_metrics_user_id (user_id),
    INDEX idx_performance_metrics_success (success)
);

-- Add RLS policies for security
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;

-- Policy: System can manage performance metrics
CREATE POLICY "System can manage performance metrics" ON performance_metrics
    FOR ALL USING (true);

-- Comments for documentation
COMMENT ON TABLE performance_metrics IS 'Performance monitoring data for all operations';
COMMENT ON COLUMN performance_metrics.operation IS 'Name of the operation being monitored';
COMMENT ON COLUMN performance_metrics.duration_ms IS 'Duration of the operation in milliseconds';
COMMENT ON COLUMN performance_metrics.success IS 'Whether the operation was successful';
COMMENT ON COLUMN performance_metrics.error_message IS 'Error message if operation failed';
