import React, { useState } from 'react';

type Props = {
  symbol: string;
  currentPrice?: number;
};

const ConditionalOrderForm: React.FC<Props> = ({ symbol, currentPrice }) => {
  const [type, setType] = useState<'stop_loss' | 'take_profit'>('stop_loss');
  const [side, setSide] = useState<'long' | 'short'>('long');
  const [size, setSize] = useState<string>('0.01');
  const [triggerPrice, setTriggerPrice] = useState<string>('');
  const [mode, setMode] = useState<'price' | 'percent'>('price');
  const [percent, setPercent] = useState<string>('');
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setMessage(null);
    try {
      const url = type === 'stop_loss'
        ? '/api/advanced-orders/stop-loss'
        : '/api/advanced-orders/take-profit';

      let trigger = Number(triggerPrice);
      if (mode === 'percent') {
        const pct = Number(percent) / 100;
        if (!currentPrice || !isFinite(currentPrice) || !isFinite(pct)) throw new Error('Invalid percent/current price');
        // Compute trigger from percent relative to current price, direction-aware
        if (type === 'stop_loss') {
          trigger = side === 'long' ? currentPrice * (1 - pct) : currentPrice * (1 + pct);
        } else {
          // take_profit
          trigger = side === 'long' ? currentPrice * (1 + pct) : currentPrice * (1 - pct);
        }
      }
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          market_id: symbol,
          size: Number(size),
          direction: side === 'long' ? 'long' : 'short',
          trigger_price: Number(trigger.toFixed(8))
        })
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setMessage(`Order created: ${data.order_id || 'OK'}`);
    } catch (err: any) {
      setMessage(`Failed: ${err.message || 'unknown error'} (are you logged in?)`);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">Conditional Orders</h3>
      <form onSubmit={submit} className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <label className="text-sm text-gray-300">Type
            <select
              className="mt-1 w-full bg-gray-900 text-white border border-gray-700 rounded px-2 py-1"
              value={type}
              onChange={e => setType(e.target.value as any)}
            >
              <option value="stop_loss">Stop-Loss</option>
              <option value="take_profit">Take-Profit</option>
            </select>
          </label>
          <label className="text-sm text-gray-300">Side
            <select
              className="mt-1 w-full bg-gray-900 text-white border border-gray-700 rounded px-2 py-1"
              value={side}
              onChange={e => setSide(e.target.value as any)}
            >
              <option value="long">Long</option>
              <option value="short">Short</option>
            </select>
          </label>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-sm text-gray-300">Size
            <input
              type="number"
              step="0.0001"
              min="0"
              className="mt-1 w-full bg-gray-900 text-white border border-gray-700 rounded px-2 py-1"
              value={size}
              onChange={e => setSize(e.target.value)}
            />
          </label>
          {mode === 'price' ? (
            <label className="text-sm text-gray-300">Trigger Price
              <input
                type="number"
                step="0.0001"
                min="0"
                className="mt-1 w-full bg-gray-900 text-white border border-gray-700 rounded px-2 py-1"
                value={triggerPrice}
                onChange={e => setTriggerPrice(e.target.value)}
              />
            </label>
          ) : (
            <label className="text-sm text-gray-300">Trigger %
              <div className="mt-1 flex items-center gap-2">
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  className="w-full bg-gray-900 text-white border border-gray-700 rounded px-2 py-1"
                  value={percent}
                  onChange={e => setPercent(e.target.value)}
                />
                <span className="text-gray-400">%</span>
              </div>
            </label>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-300">
          <span>Mode:</span>
          <button type="button" onClick={()=> setMode('price')} className={`px-2 py-1 rounded ${mode==='price'?'bg-blue-600 text-white':'bg-gray-900 border border-gray-700'}`}>Price</button>
          <button type="button" onClick={()=> setMode('percent')} className={`px-2 py-1 rounded ${mode==='percent'?'bg-blue-600 text-white':'bg-gray-900 border border-gray-700'}`}>%</button>
          {mode==='percent' && currentPrice && (
            <span className="ml-auto text-gray-400">Ref: ${currentPrice.toFixed(2)}</span>
          )}
        </div>
        <button
          type="submit"
          disabled={submitting}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded py-2"
        >
          {submitting ? 'Placing…' : 'Place Conditional Order'}
        </button>
      </form>
      {message && (
        <p className="mt-3 text-sm text-gray-300">{message}</p>
      )}
    </div>
  );
};

export default ConditionalOrderForm;


