import React from 'react';
import { useResponsiveDesign, ScreenSize } from '../hooks/useResponsiveDesign';

export interface ResponsiveContainerProps {
  children: React.ReactNode;
  className?: string;
  mobileClassName?: string;
  tabletClassName?: string;
  desktopClassName?: string;
  largeDesktopClassName?: string;
  ultraWideClassName?: string;
}

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({
  children,
  className = '',
  mobileClassName = '',
  tabletClassName = '',
  desktopClassName = '',
  largeDesktopClassName = '',
  ultraWideClassName = ''
}) => {
  const screenSize = useResponsiveDesign();

  const getResponsiveClasses = () => {
    let classes = className;

    switch (screenSize.deviceType) {
      case 'mobile':
        classes += ` ${mobileClassName}`;
        break;
      case 'tablet':
        classes += ` ${tabletClassName}`;
        break;
      case 'desktop':
        classes += ` ${desktopClassName}`;
        break;
      case 'large-desktop':
        classes += ` ${largeDesktopClassName}`;
        break;
      case 'ultra-wide':
        classes += ` ${ultraWideClassName}`;
        break;
    }

    return classes.trim();
  };

  return (
    <div className={getResponsiveClasses()}>
      {children}
    </div>
  );
};

export interface ResponsiveGridProps {
  children: React.ReactNode;
  className?: string;
  mobileCols?: number;
  tabletCols?: number;
  desktopCols?: number;
  largeDesktopCols?: number;
  ultraWideCols?: number;
  gap?: string;
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  children,
  className = '',
  mobileCols = 1,
  tabletCols = 2,
  desktopCols = 3,
  largeDesktopCols = 4,
  ultraWideCols = 5,
  gap = '4'
}) => {
  const screenSize = useResponsiveDesign();

  const getGridCols = () => {
    switch (screenSize.deviceType) {
      case 'mobile':
        return mobileCols;
      case 'tablet':
        return tabletCols;
      case 'desktop':
        return desktopCols;
      case 'large-desktop':
        return largeDesktopCols;
      case 'ultra-wide':
        return ultraWideCols;
      default:
        return desktopCols;
    }
  };

  const gridCols = getGridCols();

  return (
    <div className={`grid grid-cols-${gridCols} gap-${gap} ${className}`}>
      {children}
    </div>
  );
};

export interface ResponsiveTextProps {
  children: React.ReactNode;
  className?: string;
  mobileSize?: string;
  tabletSize?: string;
  desktopSize?: string;
  largeDesktopSize?: string;
  ultraWideSize?: string;
}

export const ResponsiveText: React.FC<ResponsiveTextProps> = ({
  children,
  className = '',
  mobileSize = 'text-sm',
  tabletSize = 'text-base',
  desktopSize = 'text-lg',
  largeDesktopSize = 'text-xl',
  ultraWideSize = 'text-2xl'
}) => {
  const screenSize = useResponsiveDesign();

  const getTextSize = () => {
    switch (screenSize.deviceType) {
      case 'mobile':
        return mobileSize;
      case 'tablet':
        return tabletSize;
      case 'desktop':
        return desktopSize;
      case 'large-desktop':
        return largeDesktopSize;
      case 'ultra-wide':
        return ultraWideSize;
      default:
        return desktopSize;
    }
  };

  return (
    <span className={`${getTextSize()} ${className}`}>
      {children}
    </span>
  );
};

export interface ResponsiveSpacingProps {
  children: React.ReactNode;
  className?: string;
  mobilePadding?: string;
  tabletPadding?: string;
  desktopPadding?: string;
  largeDesktopPadding?: string;
  ultraWidePadding?: string;
}

export const ResponsiveSpacing: React.FC<ResponsiveSpacingProps> = ({
  children,
  className = '',
  mobilePadding = 'p-2',
  tabletPadding = 'p-4',
  desktopPadding = 'p-6',
  largeDesktopPadding = 'p-8',
  ultraWidePadding = 'p-10'
}) => {
  const screenSize = useResponsiveDesign();

  const getPadding = () => {
    switch (screenSize.deviceType) {
      case 'mobile':
        return mobilePadding;
      case 'tablet':
        return tabletPadding;
      case 'desktop':
        return desktopPadding;
      case 'large-desktop':
        return largeDesktopPadding;
      case 'ultra-wide':
        return ultraWidePadding;
      default:
        return desktopPadding;
    }
  };

  return (
    <div className={`${getPadding()} ${className}`}>
      {children}
    </div>
  );
};

export interface ResponsiveChartProps {
  children: React.ReactNode;
  className?: string;
  height?: number;
}

export const ResponsiveChart: React.FC<ResponsiveChartProps> = ({
  children,
  className = '',
  height
}) => {
  const screenSize = useResponsiveDesign();
  const chartHeight = height || screenSize.chartHeight;

  return (
    <div 
      className={`${className}`}
      style={{ height: `${chartHeight}px` }}
    >
      {children}
    </div>
  );
};

export interface ResponsiveSidebarProps {
  children: React.ReactNode;
  className?: string;
  width?: number;
  isOpen?: boolean;
  onClose?: () => void;
}

export const ResponsiveSidebar: React.FC<ResponsiveSidebarProps> = ({
  children,
  className = '',
  width,
  isOpen = true,
  onClose
}) => {
  const screenSize = useResponsiveDesign();
  const sidebarWidth = width || screenSize.sidebarWidth;

  if (!isOpen) {
    return null;
  }

  return (
    <div 
      className={`bg-gray-800 border-r border-gray-700 ${className}`}
      style={{ width: `${sidebarWidth}px` }}
    >
      {screenSize.isMobile && onClose && (
        <div className="flex justify-end p-4">
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
      {children}
    </div>
  );
};

export interface ResponsiveModalProps {
  children: React.ReactNode;
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  className?: string;
}

export const ResponsiveModal: React.FC<ResponsiveModalProps> = ({
  children,
  isOpen,
  onClose,
  title,
  className = ''
}) => {
  const screenSize = useResponsiveDesign();

  if (!isOpen) {
    return null;
  }

  const getModalClasses = () => {
    if (screenSize.isMobile) {
      return 'fixed inset-0 z-50 flex items-end justify-center';
    }
    return 'fixed inset-0 z-50 flex items-center justify-center';
  };

  const getContentClasses = () => {
    if (screenSize.isMobile) {
      return 'w-full max-w-full rounded-t-lg';
    }
    return 'w-full max-w-md rounded-lg';
  };

  return (
    <div className={`${getModalClasses()} ${className}`}>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className={`bg-gray-800 ${getContentClasses()} relative z-10`}>
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        
        {/* Content */}
        <div className="p-4">
          {children}
        </div>
      </div>
    </div>
  );
};

export interface ResponsiveTableProps {
  children: React.ReactNode;
  className?: string;
  showOnMobile?: boolean;
  mobileFallback?: React.ReactNode;
}

export const ResponsiveTable: React.FC<ResponsiveTableProps> = ({
  children,
  className = '',
  showOnMobile = false,
  mobileFallback
}) => {
  const screenSize = useResponsiveDesign();

  if (screenSize.isMobile && !showOnMobile) {
    return mobileFallback || (
      <div className="text-center text-gray-400 p-4">
        Table not available on mobile. Please use desktop view.
      </div>
    );
  }

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="min-w-full">
        {children}
      </table>
    </div>
  );
};

export interface ResponsiveBreakpointProps {
  children: React.ReactNode;
  showOnMobile?: boolean;
  showOnTablet?: boolean;
  showOnDesktop?: boolean;
  showOnLargeDesktop?: boolean;
  showOnUltraWide?: boolean;
}

export const ResponsiveBreakpoint: React.FC<ResponsiveBreakpointProps> = ({
  children,
  showOnMobile = true,
  showOnTablet = true,
  showOnDesktop = true,
  showOnLargeDesktop = true,
  showOnUltraWide = true
}) => {
  const screenSize = useResponsiveDesign();

  const shouldShow = () => {
    switch (screenSize.deviceType) {
      case 'mobile':
        return showOnMobile;
      case 'tablet':
        return showOnTablet;
      case 'desktop':
        return showOnDesktop;
      case 'large-desktop':
        return showOnLargeDesktop;
      case 'ultra-wide':
        return showOnUltraWide;
      default:
        return true;
    }
  };

  if (!shouldShow()) {
    return null;
  }

  return <>{children}</>;
};

export default ResponsiveContainer;
