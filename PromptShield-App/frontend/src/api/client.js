/**
 * client.js
 * ---------
 * Thin wrapper around fetch() for the PromptShield API.
 * Automatically injects the Bearer token from localStorage when present.
 */

const BASE = '';  // Proxied by Vite in dev

function getToken() {
  return localStorage.getItem('ps_token');
}

async function request(method, path, body = null) {
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) {
    headers['Authorization'] = 'Bearer ' + token;
  }

  const opts = { method, headers };
  if (body !== null) {
    opts.body = JSON.stringify(body);
  }

  const resp = await fetch(BASE + path, opts);
  const data = await resp.json().catch(() => null);

  if (!resp.ok) {
    const msg = data?.detail || resp.statusText || 'Request failed';
    const err = new Error(msg);
    err.status = resp.status;
    throw err;
  }
  return data;
}

// -- Auth --

export async function signup(email, password) {
  return request('POST', '/auth/signup', { email, password });
}

export async function login(email, password) {
  const data = await request('POST', '/auth/login', { email, password });
  localStorage.setItem('ps_token', data.access_token);
  return data;
}

export async function getMe() {
  return request('GET', '/auth/me');
}

export function logout() {
  localStorage.removeItem('ps_token');
}

export function isLoggedIn() {
  return !!getToken();
}

// -- Predict --

export async function predict(prompt) {
  return request('POST', '/predict', { prompt });
}

export async function batchPredict(prompts) {
  return request('POST', '/predict/batch', { prompts });
}

// -- History --

export async function getHistory(page = 1, pageSize = 20) {
  return request('GET', '/history?page=' + page + '&page_size=' + pageSize);
}

// -- Health --

export async function getHealth() {
  return request('GET', '/health');
}
