import React, { useEffect, useState } from 'react';

type OrderRow = {
  id?: string;
  market_id?: string;
  order_type?: string;
  side?: string;
  size?: number;
  status?: string;
  trigger_price?: number;
  created_at?: string;
};

const OrdersStatusPanel: React.FC = () => {
  const [orders, setOrders] = useState<OrderRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setError(null);
        // Auth required; expect 401 if not logged in
        const res = await fetch('/api/orders', { credentials: 'include' });
        if (!res.ok) {
          setError(`Orders unavailable (${res.status}). Connect wallet / login to view.`);
          return;
        }
        const data = await res.json();
        if (!cancelled) setOrders(data?.orders || data || []);
      } catch (e: any) {
        if (!cancelled) setError(e.message || 'failed to load orders');
      }
    }
    load();
    const t = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  if (error) {
    return <div className="bg-gray-800 border border-gray-700 rounded p-4 text-gray-300">{error}</div>;
  }
  if (!orders) {
    return <div className="bg-gray-800 border border-gray-700 rounded p-4 text-gray-300">Loading orders…</div>;
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-4">
      <h3 className="text-white font-semibold mb-3">My Orders</h3>
      <div className="overflow-auto max-h-96">
        <table className="w-full text-sm">
          <thead className="text-gray-400">
            <tr>
              <th className="text-left py-1 pr-3">Market</th>
              <th className="text-left py-1 pr-3">Type</th>
              <th className="text-left py-1 pr-3">Side</th>
              <th className="text-left py-1 pr-3">Size</th>
              <th className="text-left py-1 pr-3">Trigger</th>
              <th className="text-left py-1 pr-3">Status</th>
              <th className="text-left py-1 pr-3">Created</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((o, i) => (
              <tr key={o.id || i} className="text-gray-200 border-t border-gray-700">
                <td className="py-1 pr-3">{o.market_id || '-'}</td>
                <td className="py-1 pr-3">{o.order_type || '-'}</td>
                <td className="py-1 pr-3">{o.side || '-'}</td>
                <td className="py-1 pr-3">{o.size ?? '-'}</td>
                <td className="py-1 pr-3">{o.trigger_price ?? '-'}</td>
                <td className="py-1 pr-3">{o.status || '-'}</td>
                <td className="py-1 pr-3">{o.created_at ? new Date(o.created_at).toLocaleString() : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default OrdersStatusPanel;


