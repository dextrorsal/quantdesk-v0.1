// QuantDesk Admin Authentication Context
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  username: string;
  role: 'admin' | 'super-admin';
  permissions: string[];
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (token: string) => void;
  logout: () => void;
  checkAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing token on mount
  useEffect(() => {
    // Check for OAuth token from main app
    const urlParams = new URLSearchParams(window.location.search);
    const oauthToken = urlParams.get('token');
    
    if (oauthToken) {
      // Store OAuth token and clean URL
      localStorage.setItem('quantdesk_admin_token', oauthToken);
      setToken(oauthToken);
      window.history.replaceState({}, document.title, window.location.pathname);
      checkAuth(oauthToken);
    } else {
      // Check for existing stored token
      const storedToken = localStorage.getItem('quantdesk_admin_token');
      if (storedToken) {
        setToken(storedToken);
        checkAuth(storedToken);
      } else {
        setIsLoading(false);
      }
    }
  }, []);

  const checkAuth = async (tokenToCheck?: string): Promise<boolean> => {
    const tokenToUse = tokenToCheck || token;
    if (!tokenToUse) {
      setIsLoading(false);
      return false;
    }

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
      const response = await fetch(`${apiUrl}/api/admin/verify`, {
        headers: {
          'Authorization': `Bearer ${tokenToUse}`
        }
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData.user);
        setIsLoading(false);
        return true;
      } else {
        // Token is invalid
        logout();
        return false;
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      logout();
      return false;
    }
  };

  const login = (newToken: string) => {
    setToken(newToken);
    localStorage.setItem('quantdesk_admin_token', newToken);
    checkAuth();
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('quantdesk_admin_token');
    setIsLoading(false);
  };

  const value: AuthContextType = {
    user,
    token,
    isAuthenticated: !!user && !!token,
    isLoading,
    login,
    logout,
    checkAuth
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
