// api.js — 所有后端请求的封装
// 会话身份走 HttpOnly Cookie，前端不再携带 API Key。

const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function errorMessage(res) {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail) && data.detail.length) {
      return data.detail[0].msg || `请求失败（${res.status}）`;
    }
  } catch {
    /* 非 JSON 响应 */
  }
  return `请求失败（${res.status}）`;
}

export async function request(path, options = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...options });
  if (!res.ok) {
    const err = new Error(await errorMessage(res));
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null;
  const type = res.headers.get('content-type') || '';
  return type.includes('application/json') ? res.json() : res.text();
}

export const getMe = () => request('/api/me');

export const getAuthConfig = () => request('/api/auth-config');

export const login = (username, password) =>
  request('/api/login', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ username, password }),
  });

export const register = ({ username, password, inviteCode }) =>
  request('/api/register', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      username,
      password,
      invite_code: inviteCode || null,
    }),
  });

export const logout = () => request('/api/logout', { method: 'POST' });

/**
 * 流式对话。每收到一个 SSE 事件就调用一次 onEvent(evt)。
 * 服务端事件类型：thinking / sources / text / done / error
 */
export async function streamChat({ message, history, conversationId, signal, onEvent }) {
  const res = await fetch('/chat', {
    method: 'POST',
    credentials: 'same-origin',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      message,
      history: history || [],
      conversation_id: conversationId || null,
    }),
    signal,
  });

  if (!res.ok) {
    const err = new Error(await errorMessage(res));
    err.status = res.status;
    throw err;
  }
  if (!res.body) throw new Error('当前浏览器不支持流式响应');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const handleLine = (line) => {
    if (!line.startsWith('data: ')) return;
    const payload = line.slice(6);
    if (!payload) return;
    try {
      onEvent(JSON.parse(payload));
    } catch {
      /* 非 JSON 事件，忽略 */
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const part of parts) {
      for (const line of part.split('\n')) handleLine(line);
    }
  }
  if (buffer.trim()) {
    for (const line of buffer.split('\n')) handleLine(line);
  }
}
