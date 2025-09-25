import { ChartManager, SeriesData } from './ChartManager'

export interface IndicatorInfo {
  name: string
  type: string
  outputs: Record<string, {
    type: string
    plotOptions?: any
  }>
  parameters: Record<string, any>
}

export interface IndicatorData {
  time: number
  [key: string]: any
}

export interface Indicator {
  id: string
  info: IndicatorInfo
  data: IndicatorData[]
  paneIndex: number
  paneHtmlElement: HTMLElement | null
}

export class IndicatorManager {
  private readonly chartManager: ChartManager
  private readonly indicators: Map<string, Indicator> = new Map()
  private indicatorCounter = 0

  constructor(chartManager: ChartManager) {
    this.chartManager = chartManager
  }

  addIndicator(info: IndicatorInfo, data: IndicatorData[], paneIndex: number = 0): string {
    const id = `indicator_${++this.indicatorCounter}`
    
    const indicator: Indicator = {
      id,
      info,
      data,
      paneIndex,
      paneHtmlElement: null
    }

    this.indicators.set(id, indicator)
    this.addIndicatorSeries(id)
    
    return id
  }

  updateIndicatorData(id: string, data: IndicatorData[]): void {
    const indicator = this.indicators.get(id)
    if (!indicator) return

    indicator.data = data
    this.addIndicatorSeries(id)
  }

  private async addIndicatorSeries(id: string): Promise<void> {
    const indicator = this.indicators.get(id)
    if (!indicator) return

    const { info, data, paneIndex } = indicator

    if (!data.length) return

    // Get pane HTML element for UI overlays
    if (!indicator.paneHtmlElement) {
      indicator.paneHtmlElement = await this.chartManager.getPaneHtmlElement(paneIndex)
    }

    for (const outputKey in info.outputs) {
      if (outputKey === 'timestamp') continue

      const outputInfo = info.outputs[outputKey]
      const transformedData = this.transformIndicatorData(data, outputKey)

      if (transformedData.length > 0) {
        let seriesOptions = { ...outputInfo.plotOptions || {} }
        const seriesKey = `${id}_${outputKey}`

        this.chartManager.addSeries(
          seriesKey,
          outputInfo.type,
          transformedData,
          seriesOptions,
          paneIndex
        )
      }
    }
  }

  private transformIndicatorData(data: IndicatorData[], outputKey: string): SeriesData[] {
    return data.map(item => ({
      time: Math.floor(new Date(item.time * 1000).getTime() / 1000),
      value: item[outputKey],
      color: item.color
    })).filter(item => item.value !== undefined && item.value !== null)
  }

  updateIndicatorStyles(id: string, outputKey: string, newStyles: any): void {
    const seriesKey = `${id}_${outputKey}`
    this.chartManager.updateSeriesOptions(seriesKey, newStyles)
  }

  removeIndicator(id: string): void {
    this.removeIndicatorSeries(id)
    this.indicators.delete(id)
  }

  private removeIndicatorSeries(id: string): void {
    const indicator = this.indicators.get(id)
    if (!indicator) return

    for (const indicatorOutput in indicator.info.outputs) {
      if (indicatorOutput === 'timestamp') continue

      const seriesKey = `${id}_${indicatorOutput}`
      this.chartManager.removeSeries(seriesKey)
    }
  }

  getIndicator(id: string): Indicator | undefined {
    return this.indicators.get(id)
  }

  getAllIndicators(): Indicator[] {
    return Array.from(this.indicators.values())
  }

  destroy(): void {
    const allIndicatorIds = Array.from(this.indicators.keys())
    
    for (const id of allIndicatorIds) {
      this.removeIndicator(id)
    }
    
    this.indicators.clear()
  }
}
