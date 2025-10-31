/**
 * Centralized Navigation Utility
 * Provides consistent navigation patterns across the application
 */

import { NavigateFunction } from 'react-router-dom';

export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  requiresAuth?: boolean;
  isExternal?: boolean;
}

export interface NavigationState {
  currentPath: string;
  previousPath?: string;
  history: string[];
  breadcrumbs: NavigationItem[];
}

export class NavigationManager {
  private static instance: NavigationManager;
  private navigate: NavigateFunction | null = null;
  private state: NavigationState = {
    currentPath: '/',
    history: ['/'],
    breadcrumbs: []
  };

  private constructor() {}

  public static getInstance(): NavigationManager {
    if (!NavigationManager.instance) {
      NavigationManager.instance = new NavigationManager();
    }
    return NavigationManager.instance;
  }

  /**
   * Initialize navigation manager with React Router navigate function
   */
  public initialize(navigate: NavigateFunction): void {
    this.navigate = navigate;
  }

  /**
   * Navigate to a specific path
   */
  public navigateTo(path: string, options?: { replace?: boolean; state?: any }): void {
    if (!this.navigate) {
      console.error('NavigationManager not initialized');
      return;
    }

    const { replace = false, state } = options || {};

    // Update internal state
    this.updateNavigationState(path);

    // Perform navigation
    if (replace) {
      this.navigate(path, { replace: true, state });
    } else {
      this.navigate(path, { state });
    }

    console.log(`🧭 Navigated to: ${path}`);
  }

  /**
   * Navigate back in history
   */
  public goBack(): void {
    if (this.state.history.length > 1) {
      const previousPath = this.state.history[this.state.history.length - 2];
      this.navigateTo(previousPath, { replace: true });
    }
  }

  /**
   * Navigate forward in history
   */
  public goForward(): void {
    // This would require maintaining forward history
    // For now, just log that it's not implemented
    console.warn('Forward navigation not implemented');
  }

  /**
   * Update navigation state
   */
  private updateNavigationState(path: string): void {
    const previousPath = this.state.currentPath;
    
    // Don't add duplicate consecutive paths
    if (path !== this.state.currentPath) {
      this.state.previousPath = previousPath;
      this.state.currentPath = path;
      this.state.history.push(path);
      
      // Limit history size
      if (this.state.history.length > 50) {
        this.state.history = this.state.history.slice(-50);
      }
    }
  }

  /**
   * Get current navigation state
   */
  public getState(): NavigationState {
    return { ...this.state };
  }

  /**
   * Get navigation history
   */
  public getHistory(): string[] {
    return [...this.state.history];
  }

  /**
   * Check if a path is accessible
   */
  public canNavigateTo(path: string): boolean {
    // Add any access control logic here
    return true;
  }

  /**
   * Get breadcrumbs for current path
   */
  public getBreadcrumbs(): NavigationItem[] {
    return [...this.state.breadcrumbs];
  }

  /**
   * Set breadcrumbs for current path
   */
  public setBreadcrumbs(breadcrumbs: NavigationItem[]): void {
    this.state.breadcrumbs = breadcrumbs;
  }
}

/**
 * Main navigation items for the application
 */
export const MAIN_NAVIGATION_ITEMS: NavigationItem[] = [
  {
    id: 'overview',
    label: 'Overview',
    path: '/overview',
    icon: '📊',
    requiresAuth: false
  },
  {
    id: 'trading',
    label: 'Trading',
    path: '/trading',
    icon: '📈',
    requiresAuth: true
  },
  {
    id: 'portfolio',
    label: 'Portfolio',
    path: '/portfolio',
    icon: '💼',
    requiresAuth: true
  },
  {
    id: 'markets',
    label: 'Markets',
    path: '/markets',
    icon: '🏪',
    requiresAuth: false
  },
  {
    id: 'pro',
    label: 'Pro',
    path: '/pro',
    icon: '⚡',
    requiresAuth: true
  }
];

/**
 * Secondary navigation items
 */
export const SECONDARY_NAVIGATION_ITEMS: NavigationItem[] = [
  {
    id: 'chat',
    label: 'Chat',
    path: '/chat-integration',
    icon: '💬',
    requiresAuth: true
  },
  {
    id: 'dex-trading',
    label: 'DEX Trading',
    path: '/dex-trading',
    icon: '🔄',
    requiresAuth: true
  },
  {
    id: 'theme-demo',
    label: 'Theme Demo',
    path: '/theme-demo',
    icon: '🎨',
    requiresAuth: false
  }
];

/**
 * Navigation utility functions
 */
export class NavigationUtils {
  /**
   * Get navigation item by ID
   */
  static getNavigationItem(id: string): NavigationItem | undefined {
    return [...MAIN_NAVIGATION_ITEMS, ...SECONDARY_NAVIGATION_ITEMS].find(item => item.id === id);
  }

  /**
   * Get navigation item by path
   */
  static getNavigationItemByPath(path: string): NavigationItem | undefined {
    return [...MAIN_NAVIGATION_ITEMS, ...SECONDARY_NAVIGATION_ITEMS].find(item => item.path === path);
  }

  /**
   * Check if path requires authentication
   */
  static requiresAuth(path: string): boolean {
    const item = this.getNavigationItemByPath(path);
    return item?.requiresAuth || false;
  }

  /**
   * Get all navigation items
   */
  static getAllNavigationItems(): NavigationItem[] {
    return [...MAIN_NAVIGATION_ITEMS, ...SECONDARY_NAVIGATION_ITEMS];
  }

  /**
   * Filter navigation items by authentication requirement
   */
  static getNavigationItemsForUser(isAuthenticated: boolean): NavigationItem[] {
    return this.getAllNavigationItems().filter(item => 
      !item.requiresAuth || isAuthenticated
    );
  }

  /**
   * Generate breadcrumbs for a path
   */
  static generateBreadcrumbs(path: string): NavigationItem[] {
    const breadcrumbs: NavigationItem[] = [];
    const pathSegments = path.split('/').filter(segment => segment);

    // Add home
    breadcrumbs.push({
      id: 'home',
      label: 'Home',
      path: '/',
      icon: '🏠'
    });

    // Add path segments
    let currentPath = '';
    pathSegments.forEach(segment => {
      currentPath += `/${segment}`;
      const item = this.getNavigationItemByPath(currentPath);
      if (item) {
        breadcrumbs.push(item);
      } else {
        // Create breadcrumb for unknown path
        breadcrumbs.push({
          id: segment,
          label: segment.charAt(0).toUpperCase() + segment.slice(1),
          path: currentPath
        });
      }
    });

    return breadcrumbs;
  }

  /**
   * Validate navigation path
   */
  static validatePath(path: string): { isValid: boolean; error?: string } {
    if (!path || typeof path !== 'string') {
      return { isValid: false, error: 'Invalid path' };
    }

    if (!path.startsWith('/')) {
      return { isValid: false, error: 'Path must start with /' };
    }

    // Check for invalid characters
    if (!/^[a-zA-Z0-9\/\-_]*$/.test(path)) {
      return { isValid: false, error: 'Path contains invalid characters' };
    }

    return { isValid: true };
  }

  /**
   * Get navigation error message
   */
  static getNavigationErrorMessage(error: any): string {
    if (typeof error === 'string') {
      return error;
    }

    if (error?.message) {
      return error.message;
    }

    return 'Navigation error occurred';
  }
}

/**
 * Navigation hook for React components
 */
export function useNavigation() {
  const navigationManager = NavigationManager.getInstance();

  return {
    navigateTo: (path: string, options?: { replace?: boolean; state?: any }) => 
      navigationManager.navigateTo(path, options),
    goBack: () => navigationManager.goBack(),
    goForward: () => navigationManager.goForward(),
    getState: () => navigationManager.getState(),
    getHistory: () => navigationManager.getHistory(),
    canNavigateTo: (path: string) => navigationManager.canNavigateTo(path),
    getBreadcrumbs: () => navigationManager.getBreadcrumbs(),
    setBreadcrumbs: (breadcrumbs: NavigationItem[]) => navigationManager.setBreadcrumbs(breadcrumbs)
  };
}

export default NavigationManager;
