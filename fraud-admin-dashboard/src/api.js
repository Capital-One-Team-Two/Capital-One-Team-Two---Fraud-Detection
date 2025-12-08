// src/api.js
import { get, post } from 'aws-amplify/api';

// Helper to unwrap the { body }.response pattern
async function parseResponse(operation) {
  const { body } = await operation.response;
  // API Gateway + Lambda typically return JSON
  return body ? body.json() : null;
}

// ---- Transactions ----

export async function listTransactions() {
  return parseResponse(
    get({
      apiName: 'AdminAPI',
      path: '/admin/transactions',
      options: {}
    })
  );
}

export async function getTransaction(transactionId) {
  return parseResponse(
    get({
      apiName: 'AdminAPI',
      path: `/admin/transactions/${transactionId}`,
      options: {}
    })
  );
}

// IMPORTANT: adapt this to the path you actually created
// Option A: POST /admin/transactions/{transactionId}/notify
export async function resendNotification(transactionId) {
  return parseResponse(
    post({
      apiName: 'AdminAPI',
      path: `/admin/notify`,
      options: {
        body: { transaction_id: transactionId }
      }          // no body for this path
    })
  );
}

// If you instead used POST /admin/notify with body, use this version:
/*
export async function resendNotification(transactionId) {
  return parseResponse(
    post({
      apiName: 'AdminAPI',
      path: '/admin/notify',
      options: {
        body: { transaction_id: transactionId },
      }
    })
  );
}
*/

// ---- Accounts ----

export async function listAccounts() {
  return parseResponse(
    get({
      apiName: 'AdminAPI',
      path: '/admin/accounts',
      options: {}
    })
  );
}

export async function getAccount(userId) {
  return parseResponse(
    get({
      apiName: 'AdminAPI',
      path: `/admin/accounts/${userId}`,
      options: {}
    })
  );
}

export async function updateAccount(userId, payload) {
  return parseResponse(
    post({
      apiName: 'AdminAPI',
      path: `/admin/accounts/${userId}`,
      options: {
        body: payload
      }
    })
  );
}

// ---- Metrics ----

export async function getMetrics() {
  return parseResponse(
    get({
      apiName: 'AdminAPI',
      path: '/admin/metrics',
      options: {}
    })
  );
}
