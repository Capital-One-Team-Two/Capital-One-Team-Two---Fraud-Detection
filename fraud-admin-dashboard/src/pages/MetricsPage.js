import React, { useEffect, useState } from 'react';
import { getMetrics } from '../api';

export default function MetricsPage() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    async function load() {
      const res = await getMetrics();
      setMetrics(res);
    }
    load();
  }, []);

  if (!metrics) return <p>Loading metrics…</p>;

  return (
    <div>
      <h1>System Metrics</h1>
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <div className="card">
          <h3>Total transactions</h3>
          <p>{metrics.total_transactions}</p>
        </div>
        <div className="card">
          <h3>Flagged transactions</h3>
          <p>{metrics.flagged_transactions}</p>
        </div>
        <div className="card">
          <h3>Fraud rate</h3>
          <p>{(metrics.fraud_rate * 100).toFixed(2)}%</p>
        </div>
        <div className="card">
          <h3>SMS sent</h3>
          <p>{metrics.sms_sent}</p>
        </div>
      </div>
    </div>
  );
}
