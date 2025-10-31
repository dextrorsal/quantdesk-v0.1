-- Analytics Database Schema for Supabase
-- Comprehensive usage analytics and monitoring data models

-- Request metrics table for storing individual request data
CREATE TABLE IF NOT EXISTS analytics_request_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    tokens_used INTEGER NOT NULL,
    cost DECIMAL(10, 6) NOT NULL,
    quality_score DECIMAL(3, 2) NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
    response_time INTEGER NOT NULL, -- milliseconds
    task_type VARCHAR(100) NOT NULL,
    session_id VARCHAR(255),
    escalation_count INTEGER DEFAULT 0,
    fallback_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Provider utilization summary table
CREATE TABLE IF NOT EXISTS analytics_provider_utilization (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    total_cost DECIMAL(10, 6) NOT NULL DEFAULT 0,
    average_response_time DECIMAL(8, 2) NOT NULL DEFAULT 0,
    success_rate DECIMAL(3, 2) NOT NULL DEFAULT 1.0,
    error_rate DECIMAL(3, 2) NOT NULL DEFAULT 0.0,
    utilization_percentage DECIMAL(5, 2) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(provider, date)
);

-- Cost reports table for storing generated reports
CREATE TABLE IF NOT EXISTS analytics_cost_reports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    time_range_start TIMESTAMP WITH TIME ZONE NOT NULL,
    time_range_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_cost DECIMAL(10, 6) NOT NULL,
    cost_savings DECIMAL(10, 6) NOT NULL,
    baseline_cost DECIMAL(10, 6) NOT NULL,
    savings_percentage DECIMAL(5, 2) NOT NULL,
    roi DECIMAL(5, 2) NOT NULL,
    report_data JSONB NOT NULL, -- Store detailed breakdown
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User satisfaction metrics table
CREATE TABLE IF NOT EXISTS analytics_satisfaction_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date DATE NOT NULL,
    average_quality_score DECIMAL(3, 2) NOT NULL,
    user_satisfaction_rate DECIMAL(3, 2) NOT NULL,
    escalation_rate DECIMAL(3, 2) NOT NULL,
    fallback_rate DECIMAL(3, 2) NOT NULL,
    response_time_metrics JSONB NOT NULL,
    quality_distribution JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date)
);

-- Analytics configuration table
CREATE TABLE IF NOT EXISTS analytics_config (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type VARCHAR(50) NOT NULL DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data retention policies table
CREATE TABLE IF NOT EXISTS analytics_retention_policies (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    retention_days INTEGER NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_cleanup TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_request_metrics_provider ON analytics_request_metrics(provider);
CREATE INDEX IF NOT EXISTS idx_request_metrics_created_at ON analytics_request_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_request_metrics_task_type ON analytics_request_metrics(task_type);
CREATE INDEX IF NOT EXISTS idx_request_metrics_session_id ON analytics_request_metrics(session_id);

CREATE INDEX IF NOT EXISTS idx_provider_utilization_provider_date ON analytics_provider_utilization(provider, date);
CREATE INDEX IF NOT EXISTS idx_provider_utilization_date ON analytics_provider_utilization(date);

CREATE INDEX IF NOT EXISTS idx_cost_reports_time_range ON analytics_cost_reports(time_range_start, time_range_end);
CREATE INDEX IF NOT EXISTS idx_cost_reports_created_at ON analytics_cost_reports(created_at);

CREATE INDEX IF NOT EXISTS idx_satisfaction_metrics_date ON analytics_satisfaction_metrics(date);

-- Row Level Security (RLS) policies
ALTER TABLE analytics_request_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_provider_utilization ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_cost_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_satisfaction_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_retention_policies ENABLE ROW LEVEL SECURITY;

-- RLS policies for service role access
CREATE POLICY "Service role can manage analytics_request_metrics" ON analytics_request_metrics
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can manage analytics_provider_utilization" ON analytics_provider_utilization
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can manage analytics_cost_reports" ON analytics_cost_reports
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can manage analytics_satisfaction_metrics" ON analytics_satisfaction_metrics
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can manage analytics_config" ON analytics_config
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can manage analytics_retention_policies" ON analytics_retention_policies
    FOR ALL USING (auth.role() = 'service_role');

-- Functions for data retention
CREATE OR REPLACE FUNCTION cleanup_old_analytics_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    retention_days INTEGER;
BEGIN
    -- Get retention policy for request metrics
    SELECT retention_days INTO retention_days 
    FROM analytics_retention_policies 
    WHERE table_name = 'analytics_request_metrics' AND enabled = TRUE;
    
    IF retention_days IS NOT NULL THEN
        -- Delete old request metrics
        DELETE FROM analytics_request_metrics 
        WHERE created_at < NOW() - INTERVAL '1 day' * retention_days;
        
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        
        -- Update last cleanup timestamp
        UPDATE analytics_retention_policies 
        SET last_cleanup = NOW() 
        WHERE table_name = 'analytics_request_metrics';
    END IF;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get analytics summary
CREATE OR REPLACE FUNCTION get_analytics_summary(
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_requests', COUNT(*),
        'total_cost', COALESCE(SUM(cost), 0),
        'total_tokens', COALESCE(SUM(tokens_used), 0),
        'average_quality_score', COALESCE(AVG(quality_score), 0),
        'average_response_time', COALESCE(AVG(response_time), 0),
        'provider_count', COUNT(DISTINCT provider),
        'escalation_count', SUM(escalation_count),
        'fallback_count', COUNT(*) FILTER (WHERE fallback_used = TRUE)
    ) INTO result
    FROM analytics_request_metrics
    WHERE created_at BETWEEN start_date AND end_date;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to get provider utilization
CREATE OR REPLACE FUNCTION get_provider_utilization(
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_agg(
        jsonb_build_object(
            'provider', provider,
            'request_count', COUNT(*),
            'total_tokens', SUM(tokens_used),
            'total_cost', SUM(cost),
            'average_response_time', AVG(response_time),
            'average_quality_score', AVG(quality_score),
            'utilization_percentage', (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER())
        )
    ) INTO result
    FROM analytics_request_metrics
    WHERE created_at BETWEEN start_date AND end_date
    GROUP BY provider
    ORDER BY COUNT(*) DESC;
    
    RETURN COALESCE(result, '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- Insert default configuration
INSERT INTO analytics_config (config_key, config_value, config_type, description) VALUES
('data_retention_days', '90', 'integer', 'Number of days to retain analytics data'),
('privacy_compliance', 'true', 'boolean', 'Enable privacy compliance features'),
('anonymize_data', 'true', 'boolean', 'Anonymize user data in analytics'),
('real_time_tracking', 'true', 'boolean', 'Enable real-time analytics tracking'),
('batch_size', '100', 'integer', 'Batch size for analytics data processing'),
('flush_interval', '30000', 'integer', 'Flush interval in milliseconds')
ON CONFLICT (config_key) DO NOTHING;

-- Insert default retention policies
INSERT INTO analytics_retention_policies (table_name, retention_days, enabled) VALUES
('analytics_request_metrics', 90, TRUE),
('analytics_provider_utilization', 365, TRUE),
('analytics_cost_reports', 365, TRUE),
('analytics_satisfaction_metrics', 365, TRUE)
ON CONFLICT (table_name) DO NOTHING;

-- Create trigger for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_analytics_request_metrics_updated_at
    BEFORE UPDATE ON analytics_request_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analytics_provider_utilization_updated_at
    BEFORE UPDATE ON analytics_provider_utilization
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analytics_config_updated_at
    BEFORE UPDATE ON analytics_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analytics_retention_policies_updated_at
    BEFORE UPDATE ON analytics_retention_policies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
