// Trade Form Component - Market/Limit Orders with Stop Loss/Take Profit
// Adapted from https://github.com/Hamz-06/crypto-position-calculator
// Styled for QuantDesk terminal aesthetic

import React, { useState, useEffect } from 'react';

interface TradeFormProps {
  symbol: string;
  currentPrice: number | null;
  onTradeSubmit: (trade: any) => void;
}

export const TradeForm: React.FC<TradeFormProps> = ({ 
  symbol, 
  currentPrice,
  onTradeSubmit 
}) => {
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [positionType, setPositionType] = useState<'long' | 'short'>('long');
  const [limitPrice, setLimitPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [takeProfit, setTakeProfit] = useState('');
  const [clickOnChart, setClickOnChart] = useState(false);
  const [errors, setErrors] = useState<Record<string, boolean>>({});

  const handleSubmit = () => {
    // Validate inputs
    const newErrors: Record<string, boolean> = {};
    
    if (orderType === 'limit' && !limitPrice && !clickOnChart) {
      newErrors.limitPrice = true;
    }
    if (!stopLoss) newErrors.stopLoss = true;
    if (!takeProfit) newErrors.takeProfit = true;
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    const entryPrice = orderType === 'market' 
      ? currentPrice 
      : parseFloat(limitPrice);

    // Calculate stop loss and take profit prices
    const stopLossPrice = positionType === 'long'
      ? entryPrice! * (1 - parseFloat(stopLoss) / 100)
      : entryPrice! * (1 + parseFloat(stopLoss) / 100);
    
    const takeProfitPrice = positionType === 'long'
      ? entryPrice! * (1 + parseFloat(takeProfit) / 100)
      : entryPrice! * (1 - parseFloat(takeProfit) / 100);

    onTradeSubmit({
      symbol,
      orderType,
      positionType,
      entryPrice,
      stopLossPercent: parseFloat(stopLoss),
      takeProfitPercent: parseFloat(takeProfit),
      stopLossPrice,
      takeProfitPrice,
    });

    // Clear form
    setLimitPrice('');
    setStopLoss('');
    setTakeProfit('');
    setErrors({});
  };

  return (
    <div style={{
      padding: '16px',
      backgroundColor: '#000',
      borderTop: '1px solid #333',
      fontFamily: 'JetBrains Mono, monospace'
    }}>
      {/* Order Type Tabs */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '16px'
      }}>
        <button
          onClick={() => setOrderType('market')}
          style={{
            flex: 1,
            padding: '8px',
            backgroundColor: orderType === 'market' ? '#3b82f6' : '#1a1a1a',
            border: '1px solid #333',
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '12px',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          Market Order
        </button>
        <button
          onClick={() => setOrderType('limit')}
          style={{
            flex: 1,
            padding: '8px',
            backgroundColor: orderType === 'limit' ? '#3b82f6' : '#1a1a1a',
            border: '1px solid #333',
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '12px',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          Limit Order
        </button>
      </div>

      {/* Limit Price Input (only for limit orders) */}
      {orderType === 'limit' && (
        <div style={{ marginBottom: '12px' }}>
          <label style={{
            display: 'block',
            fontSize: '11px',
            color: '#999',
            marginBottom: '4px'
          }}>
            Limit Price
          </label>
          <input
            type="number"
            value={limitPrice}
            onChange={(e) => setLimitPrice(e.target.value)}
            placeholder={currentPrice ? `Current: $${currentPrice.toFixed(2)}` : 'Enter price'}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#1a1a1a',
              border: `1px solid ${errors.limitPrice ? '#ff4d4f' : '#333'}`,
              color: '#fff',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '13px'
            }}
          />
          {orderType === 'limit' && (
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginTop: '8px',
              fontSize: '11px',
              color: '#999',
              cursor: 'pointer'
            }}>
              <input
                type="checkbox"
                checked={clickOnChart}
                onChange={(e) => setClickOnChart(e.target.checked)}
              />
              Click on chart to set price
            </label>
          )}
        </div>
      )}

      {/* Stop Loss */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{
          display: 'block',
          fontSize: '11px',
          color: '#999',
          marginBottom: '4px'
        }}>
          Stop Loss (%) <span style={{ color: '#ff4d4f' }}>*</span>
        </label>
        <input
          type="number"
          value={stopLoss}
          onChange={(e) => setStopLoss(e.target.value)}
          placeholder="e.g. 2"
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#1a1a1a',
            border: `1px solid ${errors.stopLoss ? '#ff4d4f' : '#333'}`,
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '13px'
          }}
        />
      </div>

      {/* Take Profit */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{
          display: 'block',
          fontSize: '11px',
          color: '#999',
          marginBottom: '4px'
        }}>
          Take Profit (%) <span style={{ color: '#52c41a' }}>*</span>
        </label>
        <input
          type="number"
          value={takeProfit}
          onChange={(e) => setTakeProfit(e.target.value)}
          placeholder="e.g. 5"
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#1a1a1a',
            border: `1px solid ${errors.takeProfit ? '#ff4d4f' : '#333'}`,
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '13px'
          }}
        />
      </div>

      {/* Position Type (Long/Short) */}
      <div style={{ 
        display: 'flex',
        gap: '8px',
        marginBottom: '16px'
      }}>
        <button
          onClick={() => setPositionType('long')}
          style={{
            flex: 1,
            padding: '8px',
            backgroundColor: positionType === 'long' ? '#52c41a' : '#1a1a1a',
            border: '1px solid #333',
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '12px',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          Long
        </button>
        <button
          onClick={() => setPositionType('short')}
          style={{
            flex: 1,
            padding: '8px',
            backgroundColor: positionType === 'short' ? '#ff4d4f' : '#1a1a1a',
            border: '1px solid #333',
            color: '#fff',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '12px',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          Short
        </button>
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        style={{
          width: '100%',
          padding: '12px',
          backgroundColor: '#3b82f6',
          border: 'none',
          color: '#fff',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '13px',
          fontWeight: 'bold',
          cursor: 'pointer',
          transition: 'background-color 0.2s'
        }}
        onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#2563eb'}
        onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#3b82f6'}
      >
        Place Order
      </button>
    </div>
  );
};

export default TradeForm;

