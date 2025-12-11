// src/pages/AccountsPage.js
import React, { useEffect, useState } from 'react';
import { listAccounts, getAccount, updateAccount } from '../api';

export default function AccountsPage() {
  const [accounts, setAccounts] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [form, setForm] = useState(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingEdit, setLoadingEdit] = useState(false);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    try {
      setLoadingList(true);
      const res = await listAccounts();
      setAccounts(res.accounts || res.Items || []);
    } catch (e) {
      console.error('Failed to load accounts', e);
      alert('Failed to load accounts – check console for details.');
    } finally {
      setLoadingList(false);
    }
  }

  async function edit(userId) {
    try {
      setLoadingEdit(true);
      const acc = await getAccount(userId);   // should return a single account object
      setSelectedId(userId);
      setForm({
        name: acc.name || '',
        phone_number: acc.phone_number || '',
      });
    } catch (e) {
      console.error('Failed to load account', e);
      alert('Failed to load account details – check CloudWatch/API logs.');
      setSelectedId(null);
      setForm(null);
    } finally {
      setLoadingEdit(false);
    }
  }

  async function save() {
    try {
      await updateAccount(selectedId, form);  // body: { name, phone_number }
      alert('Account updated');
      setSelectedId(null);
      setForm(null);
      await load();                           // refresh the table
    } catch (e) {
      console.error('Failed to update account', e);
      alert('Failed to update account – check CloudWatch/API logs.');
    }
  }

  function closeModal() {
    setSelectedId(null);
    setForm(null);
  }

  return (
    <div>
      <h1>Accounts</h1>

      <div className="card">
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 60 }}>#</th>
                <th style={{ minWidth: 220 }}>User ID</th>
                <th style={{ minWidth: 200 }}>Name</th>
                <th style={{ minWidth: 160 }}>Phone</th>
                <th style={{ minWidth: 180 }}>Created At</th>
                <th style={{ minWidth: 140 }}>Status</th>
                <th style={{ width: 100 }}></th>
              </tr>
            </thead>
            <tbody>
              {accounts.map((a, idx) => (
                <tr key={a.user_id || idx}>
                  <td>{idx + 1}</td>
                  <td className="accounts-table-userid" title={a.user_id}>{a.user_id}</td>           {/* plain text, not clickable */}
                  <td className="accounts-table-name" title={a.name}>{a.name}</td>
                  <td>{a.phone_number}</td>
                  <td className="accounts-table-created">{a.created_at}</td>
                  <td>{a.account_status}</td>
                  <td>
                    <button
                      className="btn secondary"
                      onClick={() => edit(a.user_id)}
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
              {!loadingList && accounts.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '16px 0' }}>
                    No accounts found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal popup for editing name + phone */}
      {form && (
        <div className="modal-backdrop" onClick={closeModal}>
          <div className="modal-card" onClick={e => e.stopPropagation()}>
            <h2>Edit account&nbsp;{selectedId}</h2>

            {loadingEdit ? (
              <p>Loading…</p>
            ) : (
              <>
                <label>
                  Name<br />
                  <input
                    value={form.name}
                    onChange={e => setForm({ ...form, name: e.target.value })}
                    style={{
                      width: '100%',
                      marginTop: 6,
                      marginBottom: 12,
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid rgba(255,255,255,0.25)',
                      background: 'rgba(3,10,20,0.9)',
                      color: 'var(--text)'
                    }}
                  />
                </label>
                <br />
                <label>
                  Phone<br />
                  <input
                    value={form.phone_number}
                    onChange={e => setForm({ ...form, phone_number: e.target.value })}
                    style={{
                      width: '100%',
                      marginTop: 6,
                      marginBottom: 12,
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid rgba(255,255,255,0.25)',
                      background: 'rgba(3,10,20,0.9)',
                      color: 'var(--text)'
                    }}
                  />
                </label>

                <div style={{ marginTop: 16, textAlign: 'right' }}>
                  <button className="btn primary" onClick={save}>
                    Save
                  </button>{' '}
                  <button className="btn secondary" onClick={closeModal}>
                    Cancel
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
