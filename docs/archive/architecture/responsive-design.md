# Responsive Design Architecture

## Overview
QuantDesk's responsive design system is optimized for the most common monitor setups used by traders and developers.

## Common Monitor Optimizations

### Primary Targets
- **1080p 27"**: 1920×1080 resolution
- **1440p 27"**: 2560×1440 resolution

### Chart Height Optimizations
- **1080p monitors**: 550px chart height (optimized for 1920×1080)
- **1440p monitors**: 650px chart height (optimized for 2560×1440)
- **Dynamic scaling**: Charts resize based on available viewport space

### Sidebar Width Optimizations
- **1080p monitors**: 320px sidebar (25% of width)
- **1440p monitors**: 380px sidebar (22% of width)
- **Responsive**: Automatically adjusts based on screen size

## Breakpoint System

### Device Types
```typescript
'mobile': ≤768px
'tablet': 769px-1024px
'desktop': 1025px-1920px (covers both 1080p and 1440p)
'large-desktop': 1921px-2560px
'ultra-wide': >2560px
```

### Tailwind Classes
```css
mobile:max-768px
tablet:min-769px:max-1024px
desktop:min-1025px:max-1920px
large-desktop:min-1921px:max-2560px
ultra-wide:min-2561px
```

## Implementation

### Hook Usage
```typescript
import { useResponsiveDesign } from '../hooks/useResponsiveDesign';

const screenSize = useResponsiveDesign();
// screenSize.chartHeight - Dynamic chart height
// screenSize.sidebarWidth - Dynamic sidebar width
// screenSize.deviceType - Current device type
```

### Dynamic Sizing
- **Chart Height**: Automatically calculated based on viewport
- **Sidebar Width**: Optimized for common monitor ratios
- **Throttled Resize**: 100ms throttle for smooth performance

## Monitor Detection

### Console Logging
The system logs common monitor setups:
- `🖥️ Detected 1080p monitor (1920×1080)`
- `🖥️ Detected 1440p monitor (2560×1440)`
- `🖥️ Detected high-res monitor (WIDTH×HEIGHT)`

### Debugging
Check browser console for:
- Device type detection
- Chart height calculations
- Sidebar width adjustments
- Resize event handling

## Performance Features

### Throttled Resize Events
- **Throttle**: 100ms delay
- **Jittered Backoff**: Prevents excessive calculations
- **Smooth Transitions**: CSS transitions for size changes

### Memory Optimization
- **Memoized Calculations**: Cached screen size calculations
- **Efficient Updates**: Only recalculates when necessary
- **Cleanup**: Proper event listener cleanup

## Testing Checklist

### Common Monitor Tests
- [ ] **1080p Monitor**: Chart height ~550px, sidebar ~320px
- [ ] **1440p Monitor**: Chart height ~650px, sidebar ~380px
- [ ] **Window Resize**: Smooth transitions between sizes
- [ ] **Multi-Monitor**: Consistent behavior across monitors

### Responsive Tests
- [ ] **Mobile**: ≤768px switches to mobile layout
- [ ] **Tablet**: 769px-1024px uses tablet layout
- [ ] **Desktop**: 1025px-1920px uses desktop layout
- [ ] **Large Desktop**: 1921px+ uses larger layouts

## Future Enhancements

### Planned Features
- **DPI Detection**: Support for high-DPI displays
- **Aspect Ratio**: Optimize for different aspect ratios
- **User Preferences**: Allow manual size overrides
- **Performance Metrics**: Track resize performance

### Customization
- **Chart Height Override**: User-defined chart heights
- **Sidebar Width Override**: User-defined sidebar widths
- **Layout Presets**: Save/load layout preferences
- **Theme Integration**: Responsive design with theme system

## Troubleshooting

### Common Issues
1. **Charts too small**: Check viewport height calculations
2. **Sidebar too wide**: Verify width percentage calculations
3. **Resize lag**: Check throttle settings
4. **Multi-monitor issues**: Verify screen size detection

### Debug Commands
```javascript
// Check current screen size
console.log(window.innerWidth, window.innerHeight);

// Check responsive hook values
const screenSize = useResponsiveDesign();
console.log(screenSize);
```

## Best Practices

### Development
- **Test on Common Monitors**: Always test on 1080p and 1440p
- **Use Dynamic Sizing**: Avoid fixed pixel values
- **Throttle Resize Events**: Prevent performance issues
- **Log Monitor Detection**: Help with debugging

### User Experience
- **Smooth Transitions**: Use CSS transitions for size changes
- **Consistent Behavior**: Same experience across monitors
- **Performance First**: Optimize for smooth resizing
- **Accessibility**: Ensure usability at all sizes
