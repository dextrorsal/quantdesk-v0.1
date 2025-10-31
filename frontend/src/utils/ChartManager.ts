import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType
} from 'lightweight-charts'

export interface SeriesData {
  time: number
  open?: number
  high?: number
  low?: number
  close?: number
  value?: number
  color?: string
}

export interface SeriesInfo {
  series: ISeriesApi<any>
  type: string
  data: SeriesData[]
  options: any
}

export class ChartManager {
  private chart: IChartApi | null = null
  private readonly series: Map<string, SeriesInfo> = new Map()
  private container: HTMLElement | null = null
  private readonly loadedBars = 500

  private readonly defaultOptions = {
    layout: {
      textColor: '#ffffff',
      background: { type: 'solid' as ColorType, color: '#000000' },
      panes: {
        separatorColor: 'rgba(96,96,96,0.3)',
      },
    },
    grid: {
      vertLines: { color: '#1f2937' },
      horzLines: { color: '#1f2937' },
    },
    autoSize: true,
    crosshair: {
      mode: 1,
    },
    rightPriceScale: {
      borderColor: '#1f2937',
      scaleMargins: {
        top: 0.1,
        bottom: 0.1,
      },
    },
    timeScale: {
      borderColor: '#1f2937',
      timeVisible: true,
      secondsVisible: false,
    },
  }

  private readonly timeScaleOptions = {
    timeVisible: true,
    secondsVisible: false,
  }

  private readonly seriesTypes = {
    line: 'LineSeries',
    area: 'AreaSeries', 
    bar: 'BarSeries',
    baseline: 'BaselineSeries',
    candlestick: 'CandlestickSeries',
    histogram: 'HistogramSeries',
  }

  constructor(options = {}) {
    this.defaultOptions = { ...this.defaultOptions, ...options }
  }

  init(containerElement: HTMLElement): boolean {
    this.container = containerElement

    try {
      this.chart = createChart(containerElement, this.defaultOptions)
      this.chart.timeScale().applyOptions(this.timeScaleOptions)
      return true
    } catch (error) {
      console.error('Failed to initialize chart:', error)
      return false
    }
  }

  addSeries(key: string, type: string, data: SeriesData[], seriesOptions: any = {}, paneIndex: number = 0): ISeriesApi<any> | null {
    if (!this.chart) {
      console.error('Chart not initialized')
      return null
    }

    if (this.series.has(key)) {
      this.removeSeries(key)
    }

    const newSeries = this.createSeries(type, seriesOptions, paneIndex)

    if (newSeries) {
      newSeries.setData(data)
      this.series.set(key, {
        series: newSeries,
        type,
        data: [...data],
        options: { ...seriesOptions },
      })
    }

    return newSeries
  }

  private createSeries(type: string, seriesOptions: any = {}, paneIndex: number): ISeriesApi<any> | null {
    if (!this.chart) {
      console.error('Chart not initialized')
      return null
    }

    try {
      let newSeries: ISeriesApi<any>
      
      switch (type) {
        case 'candlestick':
          newSeries = this.chart.addCandlestickSeries(seriesOptions)
          break
        case 'line':
          newSeries = this.chart.addLineSeries(seriesOptions)
          break
        case 'area':
          newSeries = this.chart.addAreaSeries(seriesOptions)
          break
        case 'bar':
          newSeries = this.chart.addBarSeries(seriesOptions)
          break
        case 'histogram':
          newSeries = this.chart.addHistogramSeries(seriesOptions)
          break
        case 'baseline':
          newSeries = this.chart.addBaselineSeries(seriesOptions)
          break
        default:
          console.error('Invalid series type:', type)
          return null
      }
      
      return newSeries
    } catch (error) {
      console.error('Failed to create series:', error)
      return null
    }
  }

  removeSeries(key: string): void {
    const seriesInfo = this.series.get(key)

    try {
      if (seriesInfo && this.chart) {
        this.chart.removeSeries(seriesInfo.series)
        this.series.delete(key)
      }
    } catch (error) {
      console.error(`Failed to remove series '${key}':`, error)
    }
  }

  updateSeriesOptions(key: string, newOptions: any): boolean {
    const seriesInfo = this.series.get(key)
    if (!seriesInfo) {
      console.error(`Series '${key}' not found`)
      return false
    }

    try {
      seriesInfo.series.applyOptions(newOptions)
      seriesInfo.options = { ...seriesInfo.options, ...newOptions }
      return true
    } catch (error) {
      console.error(`Failed to update options for series '${key}':`, error)
      return false
    }
  }

  async getPaneHtmlElement(paneIndex: number = 0): Promise<HTMLElement | null> {
    // Simplified implementation - return the chart container for now
    return this.container
  }

  subscribeCrosshairMove(callback: (param: any) => void): void {
    if (this.chart) {
      this.chart.subscribeCrosshairMove(callback)
    }
  }

  subscribeVisibleLogicalRangeChange(callback: (newVisibleLogicalRange: any) => void): void {
    if (this.chart) {
      this.chart.timeScale().subscribeVisibleLogicalRangeChange(callback)
    }
  }

  getSeries(): Map<string, SeriesInfo> {
    return this.series
  }

  getChart(): IChartApi | null {
    return this.chart
  }

  destroy(): void {
    if (this.chart) {
      this.series.forEach((_, key) => {
        this.removeSeries(key)
      })

      this.chart.remove()
      this.chart = null
      this.container = null
    }
  }
}
