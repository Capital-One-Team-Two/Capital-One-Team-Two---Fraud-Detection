// src/Dashboard.js
import React, { useState } from 'react';
import TransactionsPage from './pages/TransactionsPage';
import AccountsPage from './pages/AccountsPage';
import MetricsPage from './pages/MetricsPage';

export default function Dashboard({ user, signOut }) {
  const [page, setPage] = useState('transactions');

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside className="sidebar">
        <h2>Admin Page</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
          Signed in as<br /> <strong>{user?.username}</strong>
        </p>

        <button className="btn secondary" onClick={() => setPage('transactions')}>
          Transactions
        </button>
        <button className="btn secondary" onClick={() => setPage('accounts')}>
          Accounts
        </button>
        <button className="btn secondary" onClick={() => setPage('metrics')}>
          Metrics
        </button>

        <div style={{ marginTop: 'auto' }}>
          <button className="btn secondary" onClick={signOut}>
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="main">
        {page === 'transactions' && <TransactionsPage />}
        {page === 'accounts' && <AccountsPage />}
        {page === 'metrics' && <MetricsPage />}
      </main>
    </div>
  );
}
