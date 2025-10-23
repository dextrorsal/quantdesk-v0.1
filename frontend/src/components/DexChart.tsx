import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useMarkets } from '../contexts/MarketContext';

interface ChartDataPoint {
  timestamp: number;
  price: number;
  volume?: number;
  time: string;
}

interface DexChartProps {
  symbol?: string;
  height?: number;
  showVolume?: boolean;
  chartType?: 'line' | 'area';
}

/**
 * DEX-style lightweight chart component
 * Inspired by Uniswap, DexScreener, and other clean DEX interfaces
 */
const DexChart: React.FC<DexChartProps> = ({ 
  symbol = 'BTC-PERP', 
  height = 300, 
  showVolume = false,
  chartType = 'area'
}) => {
  const { markets, selectedMarket } = useMarkets();
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Get current market data
  const currentMarket = markets.find(m => m.symbol === symbol) || selectedMarket;
  const currentPrice = currentMarket?.price || 0;

  // Generate mock chart data (in production, this would come from your API)
  const generateMockData = useMemo(() => {
    const data: ChartDataPoint[] = [];
    const now = Date.now();
    const basePrice = currentPrice || 50000; // Default BTC price
    
    // Generate 24 hours of hourly data
    for (let i = 23; i >= 0; i--) {
      const timestamp = now - (i * 60 * 60 * 1000);
      const time = new Date(timestamp);
      
      // Add some realistic price movement
      const volatility = 0.02; // 2% volatility
      const randomChange = (Math.random() - 0.5) * volatility;
      const price = basePrice * (1 + randomChange);
      
      data.push({
        timestamp,
        price: Number(price.toFixed(2)),
        volume: Math.random() * 1000000 + 500000, // Random volume
        time: time.toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          hour12: false 
        })
      });
    }
    
    return data;
  }, [currentPrice]);

  useEffect(() => {
    setIsLoading(true);
    
    // Simulate API call delay
    const timer = setTimeout(() => {
      setChartData(generateMockData);
      setIsLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, [generateMockData]);

  // Chart colors based on price movement
  const isPositive = chartData.length > 1 && 
    chartData[chartData.length - 1].price > chartData[0].price;
  
  const lineColor = isPositive ? '#10b981' : '#ef4444'; // green or red

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload?.length) {
      return (
        <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
          <p className="text-gray-300 text-sm">{`Time: ${label}`}</p>
          <p className="text-white font-semibold">
            {`Price: $${payload[0].value.toLocaleString()}`}
          </p>
          {showVolume && payload[1] && (
            <p className="text-gray-400 text-sm">
              {`Volume: ${payload[1].value.toLocaleString()}`}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  if (isLoading) {
    return (
      <div 
        className="flex items-center justify-center bg-gray-900 rounded-lg border border-gray-700"
        style={{ height }}
      >
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-gray-400 text-sm">Loading chart...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">
            {currentMarket?.symbol || symbol}
          </h3>
          <p className="text-sm text-gray-400">
            {currentMarket?.baseAsset}/{currentMarket?.quoteAsset || 'USD'}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xl font-bold text-white">
            ${currentPrice.toLocaleString()}
          </p>
          <p className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {isPositive ? '+' : ''}{((chartData[chartData.length - 1]?.price - chartData[0]?.price) / chartData[0]?.price * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height: height - 80 }}>
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'area' ? (
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={lineColor} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={lineColor} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="time" 
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                domain={['dataMin - 100', 'dataMax + 100']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="price"
                stroke={lineColor}
                strokeWidth={2}
                fill="url(#colorArea)"
              />
            </AreaChart>
          ) : (
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="time" 
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                domain={['dataMin - 100', 'dataMax + 100']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="price"
                stroke={lineColor}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: lineColor }}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Chart Footer */}
      <div className="flex items-center justify-between mt-4 text-xs text-gray-500">
        <div className="flex space-x-4">
          <span>24h High: ${Math.max(...chartData.map(d => d.price)).toLocaleString()}</span>
          <span>24h Low: ${Math.min(...chartData.map(d => d.price)).toLocaleString()}</span>
        </div>
        <div>
          <span>24h Volume: {chartData.reduce((sum, d) => sum + (d.volume || 0), 0).toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
};

export default DexChart;
