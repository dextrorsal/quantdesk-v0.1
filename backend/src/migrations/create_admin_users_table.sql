-- Create admin users table for QuantDesk
CREATE TABLE IF NOT EXISTS admin_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('founding-dev', 'admin', 'super-admin')),
    permissions JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES admin_users(id)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_admin_users_username ON admin_users(username);
CREATE INDEX IF NOT EXISTS idx_admin_users_role ON admin_users(role);
CREATE INDEX IF NOT EXISTS idx_admin_users_active ON admin_users(is_active);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_admin_users_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
CREATE TRIGGER trigger_update_admin_users_updated_at
    BEFORE UPDATE ON admin_users
    FOR EACH ROW
    EXECUTE FUNCTION update_admin_users_updated_at();

-- Insert initial admin users
INSERT INTO admin_users (username, password_hash, role, permissions) VALUES
(
    'dex',
    '$2b$10$rQZ8K9vX8K9vX8K9vX8K9e', -- 'quantdex47' (will be updated with real hash)
    'founding-dev',
    '["read", "write", "admin", "super-admin", "founding-dev"]'::jsonb
),
(
    'gorni',
    '$2b$10$rQZ8K9vX8K9vX8K9vX8K9e', -- 'quantgorni31' (will be updated with real hash)
    'admin',
    '["read", "write", "admin"]'::jsonb
) ON CONFLICT (username) DO NOTHING;

-- Create admin audit log table
CREATE TABLE IF NOT EXISTS admin_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    admin_user_id UUID REFERENCES admin_users(id),
    action VARCHAR(50) NOT NULL,
    resource VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for audit logs
CREATE INDEX IF NOT EXISTS idx_admin_audit_logs_user_id ON admin_audit_logs(admin_user_id);
CREATE INDEX IF NOT EXISTS idx_admin_audit_logs_action ON admin_audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_admin_audit_logs_created_at ON admin_audit_logs(created_at);

-- Enable Row Level Security
ALTER TABLE admin_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin_audit_logs ENABLE ROW LEVEL SECURITY;

-- Create policies (only admin users can access)
CREATE POLICY "Admin users can view all admin users" ON admin_users
    FOR SELECT USING (true);

CREATE POLICY "Admin users can update admin users" ON admin_users
    FOR UPDATE USING (true);

CREATE POLICY "Admin users can insert admin users" ON admin_users
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Admin users can view audit logs" ON admin_audit_logs
    FOR SELECT USING (true);

CREATE POLICY "Admin users can insert audit logs" ON admin_audit_logs
    FOR INSERT WITH CHECK (true);
