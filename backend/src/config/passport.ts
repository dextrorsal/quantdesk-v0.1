import passport from 'passport';
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';
import { createClient } from '@supabase/supabase-js';

// Supabase client for user management - created lazily
let supabase: any = null;

function getSupabaseClient() {
  if (!supabase) {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
      console.warn('⚠️ Supabase credentials not configured. Google OAuth will not work.');
      return null;
    }
    supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );
  }
  return supabase;
}

// Configure Google OAuth Strategy (only if credentials are available)
if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
  passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: process.env.GOOGLE_CALLBACK_URL || "/api/auth/google/callback"
  }, async (accessToken, refreshToken, profile, done) => {
        try {
          console.log('🔐 Google OAuth profile:', profile.id, profile.emails?.[0]?.value);

          const supabaseClient = getSupabaseClient();
          if (!supabaseClient) {
            return done(new Error('Supabase not configured'), null);
          }

          // Check if user exists in database
          const { data: existingUser, error: fetchError } = await supabaseClient
            .from('admin_users')
            .select('*')
            .eq('google_id', profile.id)
            .single();

    if (fetchError && fetchError.code !== 'PGRST116') {
      console.error('❌ Error fetching user:', fetchError);
      return done(fetchError, null);
    }

    if (existingUser) {
      // User exists, update last login
      const { error: updateError } = await supabaseClient
        .from('admin_users')
        .update({ 
          last_login: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .eq('google_id', profile.id);

      if (updateError) {
        console.error('❌ Error updating user login:', updateError);
      }

      return done(null, existingUser);
    }

      // Check if user exists by email (for migration from password-based auth)
      const { data: userByEmail, error: emailError } = await supabaseClient
        .from('admin_users')
        .select('*')
        .eq('email', profile.emails?.[0]?.value)
        .single();

      if (emailError && emailError.code !== 'PGRST116') {
        console.error('❌ Error fetching user by email:', emailError);
        return done(emailError, null);
      }

      if (userByEmail) {
        // Migrate existing user to Google OAuth
        const { error: migrateError } = await supabaseClient
          .from('admin_users')
          .update({
            google_id: profile.id,
            auth_provider: 'google',
            last_login: new Date().toISOString(),
            updated_at: new Date().toISOString()
          })
          .eq('email', profile.emails?.[0]?.value);

        if (migrateError) {
          console.error('❌ Error migrating user:', migrateError);
          return done(migrateError, null);
        }

        return done(null, { ...userByEmail, google_id: profile.id, auth_provider: 'google' });
      }

      // Special case for configured admin account
      const adminEmail = process.env.ADMIN_EMAIL;
      const adminGoogleId = process.env.ADMIN_GOOGLE_ID;
      
      if (adminEmail && adminGoogleId && 
          (profile.emails?.[0]?.value === adminEmail || profile.id === adminGoogleId)) {
        const admin = {
          id: process.env.ADMIN_USER_ID || 'admin-user-id', // Should be set in env
          username: process.env.ADMIN_USERNAME || 'admin',
          email: profile.emails?.[0]?.value || adminEmail,
          google_id: profile.id,
          role: process.env.ADMIN_ROLE || 'founding-dev',
          permissions: (process.env.ADMIN_PERMISSIONS || 'read,write,admin,super-admin,founding-dev').split(','),
          is_active: true,
          auth_provider: 'google',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          last_login: new Date().toISOString()
        };
        console.log('✅ Admin login successful for:', admin.email, '(Google ID:', profile.id, ')');
        return done(null, admin);
      }

      // User doesn't exist - check if email is in allowed admin list
      const allowedEmails = process.env.ADMIN_ALLOWED_EMAILS?.split(',') || [];
      const userEmail = profile.emails?.[0]?.value;

      if (!allowedEmails.includes(userEmail)) {
        console.log('🚫 Unauthorized email attempt:', userEmail);
        return done(new Error('Email not authorized for admin access'), null);
      }

    // Create new admin user
    const newUser = {
      username: profile.displayName?.toLowerCase().replace(/\s+/g, '') || profile.id,
      email: userEmail,
      google_id: profile.id,
      role: 'admin', // Default role for new users
      permissions: ['read', 'write', 'admin'],
      is_active: true,
      auth_provider: 'google',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      last_login: new Date().toISOString()
    };

    const { data: createdUser, error: createError } = await supabaseClient
      .from('admin_users')
      .insert(newUser)
      .select()
      .single();

    if (createError) {
      console.error('❌ Error creating user:', createError);
      return done(createError, null);
    }

    console.log('✅ Created new admin user:', createdUser.username);
    return done(null, createdUser);

  } catch (error) {
    console.error('❌ Google OAuth error:', error);
    return done(error, null);
  }
}));
} else {
  console.warn('⚠️ Google OAuth credentials not configured. Google authentication will not be available.');
}

// Serialize user for session
passport.serializeUser((user: any, done) => {
  done(null, user.id);
});

// Deserialize user from session
passport.deserializeUser(async (id: string, done) => {
  try {
    const supabaseClient = getSupabaseClient();
    if (!supabaseClient) {
      return done(new Error('Supabase not configured'), null);
    }
    
    const { data: user, error } = await supabaseClient
      .from('admin_users')
      .select('*')
      .eq('id', id)
      .single();

    if (error) {
      console.error('❌ Error deserializing user:', error);
      return done(error, null);
    }

    done(null, user);
  } catch (error) {
    console.error('❌ Deserialize user error:', error);
    done(error, null);
  }
});

export default passport;
