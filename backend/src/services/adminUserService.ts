// QuantDesk Admin User Service
import bcrypt from 'bcrypt';
import { databaseService } from './supabaseDatabase';
import { Logger } from '../utils/logger';

export interface AdminUser {
  id: string;
  username: string;
  role: 'founding-dev' | 'admin' | 'super-admin';
  permissions: string[];
  is_active: boolean;
  last_login?: Date;
  created_at: Date;
  updated_at: Date;
  created_by?: string;
}

export interface CreateAdminUserData {
  username: string;
  password: string;
  role: 'founding-dev' | 'admin' | 'super-admin';
  permissions?: string[];
  created_by: string;
}

export interface UpdateAdminUserData {
  username?: string;
  role?: 'founding-dev' | 'admin' | 'super-admin';
  permissions?: string[];
  is_active?: boolean;
}

export class AdminUserService {
  private static instance: AdminUserService;
  private supabase: any;
  private logger: Logger;

  private constructor() {
    this.supabase = databaseService.getClient();
    this.logger = new Logger();
  }

  public static getInstance(): AdminUserService {
    if (!AdminUserService.instance) {
      AdminUserService.instance = new AdminUserService();
    }
    return AdminUserService.instance;
  }

  // Get all admin users
  async getAllUsers(): Promise<AdminUser[]> {
    try {
      const { data, error } = await this.supabase
        .from('admin_users')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      this.logger.error('Error fetching admin users:', error);
      throw error;
    }
  }

  // Get user by username
  async getUserByUsername(username: string): Promise<AdminUser | null> {
    try {
      const { data, error } = await this.supabase
        .from('admin_users')
        .select('*')
        .eq('username', username)
        .eq('is_active', true)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data;
    } catch (error) {
      this.logger.error('Error fetching admin user by username:', error);
      throw error;
    }
  }

  // Get user by ID
  async getUserById(id: string): Promise<AdminUser | null> {
    try {
      const { data, error } = await this.supabase
        .from('admin_users')
        .select('*')
        .eq('id', id)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data;
    } catch (error) {
      this.logger.error('Error fetching admin user by ID:', error);
      throw error;
    }
  }

  // Create new admin user
  async createUser(userData: CreateAdminUserData): Promise<AdminUser> {
    try {
      // Check if username already exists
      const existingUser = await this.getUserByUsername(userData.username);
      if (existingUser) {
        throw new Error('Username already exists');
      }

      // Hash password
      const saltRounds = 10;
      const passwordHash = await bcrypt.hash(userData.password, saltRounds);

      // Set default permissions based on role
      const defaultPermissions = this.getDefaultPermissions(userData.role);
      const permissions = userData.permissions || defaultPermissions;

      const { data, error } = await this.supabase
        .from('admin_users')
        .insert({
          username: userData.username,
          password_hash: passwordHash,
          role: userData.role,
          permissions: permissions,
          created_by: userData.created_by
        })
        .select()
        .single();

      if (error) throw error;

      this.logger.info(`Admin user created: ${userData.username} by ${userData.created_by}`);
      return data;
    } catch (error) {
      this.logger.error('Error creating admin user:', error);
      throw error;
    }
  }

  // Update admin user
  async updateUser(id: string, userData: UpdateAdminUserData): Promise<AdminUser> {
    try {
      const updateData: any = { ...userData };
      
      // Remove undefined values
      Object.keys(updateData).forEach(key => {
        if (updateData[key] === undefined) {
          delete updateData[key];
        }
      });

      const { data, error } = await this.supabase
        .from('admin_users')
        .update(updateData)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;

      this.logger.info(`Admin user updated: ${id}`);
      return data;
    } catch (error) {
      this.logger.error('Error updating admin user:', error);
      throw error;
    }
  }

  // Delete admin user (soft delete)
  async deleteUser(id: string): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('admin_users')
        .update({ is_active: false })
        .eq('id', id);

      if (error) throw error;

      this.logger.info(`Admin user deactivated: ${id}`);
    } catch (error) {
      this.logger.error('Error deleting admin user:', error);
      throw error;
    }
  }

  // Verify password
  async verifyPassword(username: string, password: string): Promise<AdminUser | null> {
    try {
      this.logger.info(`Verifying password for user: ${username}`);
      
      const { data, error } = await this.supabase
        .from('admin_users')
        .select('*')
        .eq('username', username)
        .eq('is_active', true)
        .single();

      if (error) {
        this.logger.error(`Database error for user ${username}:`, error);
        if (error.code !== 'PGRST116') throw error;
        return null;
      }
      
      if (!data) {
        this.logger.warn(`User not found: ${username}`);
        return null;
      }

      this.logger.info(`User found: ${username}, checking password`);
      const isValidPassword = await bcrypt.compare(password, data.password_hash);
      if (!isValidPassword) {
        this.logger.warn(`Invalid password for user: ${username}`);
        return null;
      }

      this.logger.info(`Password valid for user: ${username}`);
      // Update last login
      await this.updateLastLogin(data.id);

      return data;
    } catch (error) {
      this.logger.error('Error verifying password:', error);
      throw error;
    }
  }

  // Update last login timestamp
  async updateLastLogin(id: string): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('admin_users')
        .update({ last_login: new Date().toISOString() })
        .eq('id', id);

      if (error) throw error;
    } catch (error) {
      this.logger.error('Error updating last login:', error);
      throw error;
    }
  }

  // Get default permissions for role
  private getDefaultPermissions(role: string): string[] {
    switch (role) {
      case 'founding-dev':
        return ['read', 'write', 'admin', 'super-admin', 'founding-dev'];
      case 'super-admin':
        return ['read', 'write', 'admin', 'super-admin'];
      case 'admin':
        return ['read', 'write', 'admin'];
      default:
        return ['read'];
    }
  }

  // Log admin action
  async logAction(
    adminUserId: string,
    action: string,
    resource?: string,
    details?: any,
    ipAddress?: string,
    userAgent?: string
  ): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('admin_audit_logs')
        .insert({
          admin_user_id: adminUserId,
          action,
          resource,
          details,
          ip_address: ipAddress,
          user_agent: userAgent
        });

      if (error) throw error;
    } catch (error) {
      this.logger.error('Error logging admin action:', error);
      // Don't throw error for logging failures
    }
  }

  // Get audit logs
  async getAuditLogs(limit: number = 100, offset: number = 0): Promise<any[]> {
    try {
      const { data, error } = await this.supabase
        .from('admin_audit_logs')
        .select(`
          *,
          admin_users!admin_audit_logs_admin_user_id_fkey (
            username,
            role
          )
        `)
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);

      if (error) throw error;
      return data || [];
    } catch (error) {
      this.logger.error('Error fetching audit logs:', error);
      throw error;
    }
  }
}
