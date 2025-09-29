-- Admin Authentication and Authorization Tables
-- Tables for managing admin users, permissions, and access control

-- Admin users table
CREATE TABLE IF NOT EXISTS admin_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'readonly_admin',
    permissions JSONB DEFAULT '[]',
    mfa_secret VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES admin_users(id)
);

-- Admin IP whitelist
CREATE TABLE IF NOT EXISTS admin_ip_whitelist (
    id SERIAL PRIMARY KEY,
    admin_id UUID NOT NULL REFERENCES admin_users(id) ON DELETE CASCADE,
    ip_address INET NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES admin_users(id)
);

-- Admin access logs
CREATE TABLE IF NOT EXISTS admin_access_logs (
    id SERIAL PRIMARY KEY,
    admin_id UUID NOT NULL REFERENCES admin_users(id),
    method VARCHAR(10) NOT NULL,
    path VARCHAR(500) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admin action logs
CREATE TABLE IF NOT EXISTS admin_action_logs (
    id SERIAL PRIMARY KEY,
    admin_id UUID NOT NULL REFERENCES admin_users(id),
    action VARCHAR(100) NOT NULL,
    target_type VARCHAR(50),
    target_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admin sessions
CREATE TABLE IF NOT EXISTS admin_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    admin_id UUID NOT NULL REFERENCES admin_users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admin roles and permissions
CREATE TABLE IF NOT EXISTS admin_roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL,
    is_system_role BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default admin roles
INSERT INTO admin_roles (name, description, permissions, is_system_role) VALUES
('super_admin', 'Super Administrator with full access', '["system_mode_change", "emergency_stop", "user_management", "system_config", "financial_data", "audit_logs", "security_settings"]', true),
('system_admin', 'System Administrator with system management access', '["system_mode_change", "system_config", "user_management", "audit_logs"]', true),
('support_admin', 'Support Administrator with user management access', '["user_management", "audit_logs", "system_metrics"]', true),
('readonly_admin', 'Read-only Administrator with view-only access', '["system_metrics", "audit_logs"]', true)
ON CONFLICT (name) DO NOTHING;

-- Create default admin user (password: admin123)
INSERT INTO admin_users (email, password_hash, role, is_active) VALUES
('admin@quantdesk.com', '$2a$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 'super_admin', true)
ON CONFLICT (email) DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_admin_users_email ON admin_users(email);
CREATE INDEX IF NOT EXISTS idx_admin_users_role ON admin_users(role);
CREATE INDEX IF NOT EXISTS idx_admin_users_active ON admin_users(is_active);
CREATE INDEX IF NOT EXISTS idx_admin_ip_whitelist_admin_id ON admin_ip_whitelist(admin_id);
CREATE INDEX IF NOT EXISTS idx_admin_ip_whitelist_ip ON admin_ip_whitelist(ip_address);
CREATE INDEX IF NOT EXISTS idx_admin_access_logs_admin_id ON admin_access_logs(admin_id);
CREATE INDEX IF NOT EXISTS idx_admin_access_logs_timestamp ON admin_access_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_admin_action_logs_admin_id ON admin_action_logs(admin_id);
CREATE INDEX IF NOT EXISTS idx_admin_action_logs_timestamp ON admin_action_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_admin_sessions_admin_id ON admin_sessions(admin_id);
CREATE INDEX IF NOT EXISTS idx_admin_sessions_token_hash ON admin_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_admin_sessions_expires_at ON admin_sessions(expires_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_admin_users_updated_at BEFORE UPDATE ON admin_users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM admin_sessions WHERE expires_at < NOW();
END;
$$ language 'plpgsql';

-- Create function to log admin actions
CREATE OR REPLACE FUNCTION log_admin_action(
    p_admin_id UUID,
    p_action VARCHAR(100),
    p_target_type VARCHAR(50),
    p_target_id VARCHAR(255),
    p_old_values JSONB,
    p_new_values JSONB,
    p_ip_address INET,
    p_user_agent TEXT
)
RETURNS void AS $$
BEGIN
    INSERT INTO admin_action_logs (
        admin_id, action, target_type, target_id, 
        old_values, new_values, ip_address, user_agent
    ) VALUES (
        p_admin_id, p_action, p_target_type, p_target_id,
        p_old_values, p_new_values, p_ip_address, p_user_agent
    );
END;
$$ language 'plpgsql';

-- Create view for admin user summary
CREATE OR REPLACE VIEW admin_user_summary AS
SELECT 
    au.id,
    au.email,
    au.role,
    au.is_active,
    au.last_login,
    au.created_at,
    COUNT(aal.id) as total_logins,
    MAX(aal.timestamp) as last_activity
FROM admin_users au
LEFT JOIN admin_access_logs aal ON au.id = aal.admin_id
GROUP BY au.id, au.email, au.role, au.is_active, au.last_login, au.created_at;

-- Create view for admin activity summary
CREATE OR REPLACE VIEW admin_activity_summary AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_actions,
    COUNT(DISTINCT admin_id) as unique_admins,
    COUNT(CASE WHEN success = true THEN 1 END) as successful_actions,
    COUNT(CASE WHEN success = false THEN 1 END) as failed_actions
FROM admin_access_logs
GROUP BY DATE(timestamp)
ORDER BY date DESC;
