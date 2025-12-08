import React, { useEffect, useState } from 'react';
import { listTransactions, getTransaction, resendNotification } from '../api';

export default function TransactionsPage() {
  const [transactions, setTransactions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    const res = await listTransactions();
    const list = res.transactions || res.Items || [];
    // optional: sort by timestamp descending if you like
    setTransactions(list);
  }

  async function openDetail(id) {
    try {
      setSelectedId(id);
      setLoadingDetail(true);
      const tx = await getTransaction(id);
      setSelected(tx);
    } catch (e) {
      console.error(e);
      alert('Failed to load transaction details');
    } finally {
      setLoadingDetail(false);
    }
  }

  async function resend(id) {
    try {
      await resendNotification(id);
      alert(`Notification re-sent for transaction ${id}`);
    } catch (e) {
      console.error(e);
      alert('Failed to resend SMS');
    }
  }

  function closeModal() {
    setSelected(null);
    setSelectedId(null);
  }

  return (
    <div>
      <h1>Transactions</h1>

      <div className="card">
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 60 }}>#</th>
                <th style={{ minWidth: 170 }}>Transaction ID</th>
                <th style={{ minWidth: 220 }}>User ID</th>
                <th style={{ width: 100 }}>Amount</th>
                <th style={{ width: 120 }}>Decision</th>
                <th style={{ width: 110 }}>SMS Sent</th>
                <th style={{ width: 140 }}>User Decision</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((t, idx) => (
                <tr key={t.transaction_id || idx}>
                  {/* count column */}
                  <td>{idx + 1}</td>
                  <td>
                    <span
                      onClick={() => openDetail(t.transaction_id)}
                      className="tx-id-link"
                      title={t.transaction_id}
                    >
                      {t.transaction_id}…
                    </span>
                  </td>

                  <td>{t.user_id}</td>
                  <td>{t.amt}</td>
                  <td>{t.decision}</td>
                  <td>{t.notificationSent ? 'Yes' : 'No'}</td>
                  <td>{t.userDecision || 'Pending'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal overlay for transaction details */}
      {selected && (
        <div className="modal-backdrop" onClick={closeModal}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h2>Transaction details</h2>
            {loadingDetail ? (
              <p>Loading…</p>
            ) : (
              <div className="tx-detail-grid">
                <div>
                  <strong>ID</strong>
                  <div>{selected.transaction_id}</div>
                </div>
                <div>
                  <strong>User ID</strong>
                  <div>{selected.user_id}</div>
                </div>
                <div>
                  <strong>Amount</strong>
                  <div>{selected.amt}</div>
                </div>
                <div>
                  <strong>Decision</strong>
                  <div>{selected.decision}</div>
                </div>
                <div>
                  <strong>Score</strong>
                  <div>{selected.p_raw}</div>
                </div>
                <div>
                  <strong>SMS Sent</strong>
                  <div>{selected.notificationSent ? 'Yes' : 'No'}</div>
                </div>
                <div>
                  <strong>User Decision</strong>
                  <div>{selected.userDecision || 'Pending'}</div>
                </div>
                <div>
                  <strong>Timestamp</strong>
                  <div>{selected.timestamp}</div>
                </div>
              </div>
            )}
            <div style={{ marginTop: 20, textAlign: 'right' }}>
              <button className="btn secondary" onClick={closeModal}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
