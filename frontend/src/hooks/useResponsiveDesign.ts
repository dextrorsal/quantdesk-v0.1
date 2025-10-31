import { useState, useEffect } from 'react';

export interface ScreenSize {
  width: number;
  height: number;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLargeDesktop: boolean;
  isUltraWide: boolean;
  deviceType: 'mobile' | 'tablet' | 'desktop' | 'large-desktop' | 'ultra-wide';
  chartHeight: number;
  sidebarWidth: number;
}

/**
 * Comprehensive responsive design hook
 * Handles multiple breakpoints and device types
 * Provides dynamic sizing for different screen configurations
 */
export const useResponsiveDesign = (): ScreenSize => {
  const [screenSize, setScreenSize] = useState<ScreenSize>(() => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    return calculateScreenSize(width, height);
  });

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      setScreenSize(calculateScreenSize(width, height));
    };

    // Throttle resize events for better performance
    let timeoutId: NodeJS.Timeout;
    const throttledResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(handleResize, 100);
    };

    window.addEventListener('resize', throttledResize);
    
    return () => {
      window.removeEventListener('resize', throttledResize);
      clearTimeout(timeoutId);
    };
  }, []);

  return screenSize;
};

/**
 * Calculate screen size and device type based on dimensions
 * Optimized for common monitor setups: 1080p and 1440p
 */
function calculateScreenSize(width: number, height: number): ScreenSize {
  // Device type detection
  const isMobile = width <= 768;
  const isTablet = width > 768 && width <= 1024;
  const isDesktop = width > 1024 && width <= 1920; // Covers both 1080p and 1440p
  const isLargeDesktop = width > 1920 && width <= 2560;
  const isUltraWide = width > 2560;

  // Determine device type
  let deviceType: ScreenSize['deviceType'] = 'desktop';
  if (isMobile) deviceType = 'mobile';
  else if (isTablet) deviceType = 'tablet';
  else if (isDesktop) deviceType = 'desktop';
  else if (isLargeDesktop) deviceType = 'large-desktop';
  else if (isUltraWide) deviceType = 'ultra-wide';

  // Dynamic chart height based on screen size
  const chartHeight = calculateChartHeight(width, height, deviceType);
  
  // Dynamic sidebar width
  const sidebarWidth = calculateSidebarWidth(width, deviceType);

  // Log common monitor setups for debugging
  if (width === 1920 && height === 1080) {
    console.log('🖥️ Detected 1080p monitor (1920×1080)');
  } else if (width === 2560 && height === 1440) {
    console.log('🖥️ Detected 1440p monitor (2560×1440)');
  } else if (width > 1920) {
    console.log(`🖥️ Detected high-res monitor (${width}×${height})`);
  }

  return {
    width,
    height,
    isMobile,
    isTablet,
    isDesktop,
    isLargeDesktop,
    isUltraWide,
    deviceType,
    chartHeight,
    sidebarWidth,
  };
}

/**
 * Calculate optimal chart height based on screen dimensions
 * Optimized for common monitor setups: 1080p (1920×1080) and 1440p (2560×1440)
 */
function calculateChartHeight(width: number, height: number, deviceType: ScreenSize['deviceType']): number {
  const availableHeight = height - 80; // Reduced from 120 to 80 to give MORE space to chart
  
  switch (deviceType) {
    case 'mobile':
      return Math.min(600, availableHeight * 0.9); // Much bigger
    case 'tablet':
      return Math.min(700, availableHeight * 0.95); // Much bigger
    case 'desktop':
      // Optimized for 1080p monitors (1920×1080)
      if (height <= 1080) {
        return Math.min(900, availableHeight * 0.95); // Much bigger
      }
      // Optimized for 1440p monitors (2560×1440)
      return Math.min(1000, availableHeight * 0.98); // Much bigger
    case 'large-desktop':
      // For larger monitors, use more space
      return Math.min(1100, availableHeight * 0.98); // Much bigger
    case 'ultra-wide':
      // Ultra-wide monitors can handle even larger charts
      return Math.min(1200, availableHeight * 0.99); // Much bigger
    default:
      return 900; // Much bigger default
  }
}

/**
 * Calculate optimal sidebar width based on screen size
 * Optimized for common monitor setups
 */
function calculateSidebarWidth(width: number, deviceType: ScreenSize['deviceType']): number {
  switch (deviceType) {
    case 'mobile':
      return width * 0.9; // Full width on mobile
    case 'tablet':
      return Math.min(400, width * 0.5);
    case 'desktop':
      // Optimized for 1080p and 1440p monitors
      if (width <= 1920) {
        return Math.min(320, width * 0.25); // Smaller sidebar for 1080p
      }
      return Math.min(380, width * 0.22); // Slightly larger for 1440p
    case 'large-desktop':
      return Math.min(420, width * 0.2);
    case 'ultra-wide':
      return Math.min(480, width * 0.18);
    default:
      return 350;
  }
}

/**
 * Get responsive classes for Tailwind CSS
 */
export const getResponsiveClasses = (screenSize: ScreenSize) => {
  return {
    container: {
      mobile: 'px-2 py-2',
      tablet: 'px-4 py-3',
      desktop: 'px-6 py-4',
      'large-desktop': 'px-8 py-5',
      'ultra-wide': 'px-10 py-6',
    },
    grid: {
      mobile: 'grid-cols-1',
      tablet: 'grid-cols-2',
      desktop: 'grid-cols-3',
      'large-desktop': 'grid-cols-4',
      'ultra-wide': 'grid-cols-5',
    },
    chart: {
      mobile: 'h-96',
      tablet: 'h-[500px]',
      desktop: 'h-[600px]',
      'large-desktop': 'h-[700px]',
      'ultra-wide': 'h-[800px]',
    },
  };
};
