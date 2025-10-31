import { ISeriesPrimitive, SeriesAttachedParameter, Time, MouseEventParams, CandlestickData } from 'lightweight-charts';

export interface TooltipPrimitiveOptions {
	lineColor?: string;
	tooltip?: {
		followMode?: 'tracking' | 'top';
	};
}

export class TooltipPrimitive implements ISeriesPrimitive<Time> {
	private _options: TooltipPrimitiveOptions;
	private _tooltipElement: HTMLElement | null = null;
	private _chart: any = null;
	private _series: any = null;

	constructor(options: TooltipPrimitiveOptions = {}) {
		this._options = {
			lineColor: 'rgba(0, 0, 0, 0.2)',
			tooltip: {
				followMode: 'top',
			},
			...options,
		};
	}

	attached(param: SeriesAttachedParameter<Time>): void {
		this._chart = param.chart;
		this._series = param.series;
		
		// Subscribe to crosshair movement
		this._chart.subscribeCrosshairMove(this._onMouseMove.bind(this));
		
		// Create tooltip element
		this._createTooltipElement();
	}

	detached(): void {
		if (this._chart) {
			this._chart.unsubscribeCrosshairMove(this._onMouseMove.bind(this));
		}
		if (this._tooltipElement && this._tooltipElement.parentNode) {
			this._tooltipElement.parentNode.removeChild(this._tooltipElement);
		}
	}

	paneViews() {
		return [];
	}

	applyOptions(options: Partial<TooltipPrimitiveOptions>): void {
		console.log('TooltipPrimitive: applyOptions called with:', options);
		console.log('TooltipPrimitive: current _options before update:', this._options);
		
		this._options = { ...this._options, ...options };
		
		console.log('TooltipPrimitive: new _options after update:', this._options);
	}

	private _createTooltipElement(): void {
		if (!this._chart) return;

		this._tooltipElement = document.createElement('div');
		this._tooltipElement.style.cssText = `
			position: absolute;
			background: rgba(26, 26, 26, 0.95);
			border: 1px solid #333333;
			border-radius: 6px;
			padding: 8px 12px;
			font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
			font-size: 12px;
			color: #ffffff;
			pointer-events: none;
			z-index: 1000;
			box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
			backdrop-filter: blur(8px);
			opacity: 0;
			transition: opacity 0.2s ease;
			min-width: 120px;
		`;

		const chartElement = this._chart.chartElement();
		chartElement.appendChild(this._tooltipElement);
	}

	private _onMouseMove(param: MouseEventParams): void {
		if (!this._tooltipElement || !this._chart || !this._series) return;

		const data = param.seriesData.get(this._series);
		if (!data || !param.point) {
			this._hideTooltip();
			return;
		}

		const candlestickData = data as CandlestickData;
		const price = candlestickData.close.toFixed(2);
		const time = param.time ? new Date(param.time * 1000).toLocaleString() : '';

		// Update tooltip content
		this._tooltipElement.innerHTML = `
			<div style="font-weight: 600; margin-bottom: 4px;">$${price}</div>
			<div style="color: #cccccc; font-size: 11px;">${time}</div>
		`;

		// Get chart container position
		const chartElement = this._chart.chartElement();
		const chartRect = chartElement.getBoundingClientRect();
		
		// Calculate position relative to chart container
		let left = param.point.x;
		let top = param.point.y;

		// Debug logging
		console.log('Tooltip Debug:', {
			followMode: this._options.tooltip?.followMode,
			pointX: param.point.x,
			pointY: param.point.y,
			chartWidth: chartRect.width,
			chartHeight: chartRect.height
		});

		// Handle different follow modes
		if (this._options.tooltip?.followMode === 'top') {
			top = 10; // Fixed at top
		} else {
			// Tracking mode - follow mouse cursor with offset
			top = param.point.y - 30; // Offset above cursor
		}

		// Adjust if tooltip would go off screen horizontally
		if (left + 120 > chartRect.width) {
			left = param.point.x - 120; // Move to left of cursor
		}
		
		// Adjust if tooltip would go off screen vertically (only in tracking mode)
		if (this._options.tooltip?.followMode !== 'top') {
			if (top < 0) {
				top = param.point.y + 20; // Show below cursor
			}
			if (top + 60 > chartRect.height) {
				top = param.point.y - 60; // Show above cursor
			}
		}

		this._tooltipElement.style.left = `${left}px`;
		this._tooltipElement.style.top = `${top}px`;
		this._tooltipElement.style.opacity = '1';
	}

	private _hideTooltip(): void {
		if (this._tooltipElement) {
			this._tooltipElement.style.opacity = '0';
		}
	}
}