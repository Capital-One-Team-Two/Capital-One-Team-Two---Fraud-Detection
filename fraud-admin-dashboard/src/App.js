// src/App.js
import React, { useState } from 'react';
import Dashboard from './Dashboard';
import './theme.css';

const ADMIN_PASSWORD = process.env.REACT_APP_ADMIN_PASSWORD;

function App() {
  const [authed, setAuthed] = useState(false);
  const [pw, setPw] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (pw === ADMIN_PASSWORD) {
      setAuthed(true);
      setError('');
    } else {
      setError('Incorrect password');
    }
  };

  if (!authed) {
    return (
      <div className="login-page">
        <div className="login-card">
          <h1 className="login-title">Admin Login</h1>
          <p className="login-subtitle">
            Enter the internal admin password to access the dashboard.
          </p>

          <form onSubmit={handleSubmit}>
            <label className="login-label">
              Password
              <input
                type="password"
                value={pw}
                onChange={(e) => setPw(e.target.value)}
                className="login-input"
              />
            </label>

            {error && <p className="login-error">{error}</p>}

            <button type="submit" className="btn primary login-button">
              Sign in
            </button>
          </form>
        </div>
      </div>
    );
  }

  return <Dashboard user={{ username: 'admin' }} signOut={() => setAuthed(false)} />;
}

export default App;

