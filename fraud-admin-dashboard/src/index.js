import React from 'react';
import ReactDOM from 'react-dom/client';
import { Amplify } from 'aws-amplify';
import App from './App';
import './theme.css';
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
const REGION = process.env.REACT_APP_API_REGION;

Amplify.configure({
  API: {
    REST: {
      AdminAPI: {
        endpoint: API_BASE_URL,
        region: REGION,
      },
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
