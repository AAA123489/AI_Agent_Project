// store.js — 会话历史的本地存储（按账号隔离）
// 键：rag_chat:v2:u<id>:conversations

const PREFIX = 'rag_chat:v2';
const LEGACY_CHATS_KEY = 'rag_chat_conversations';
const LEGACY_CLEAR_KEYS = ['rag_chat_api_key', 'rag_chat_user_id'];

let userId = null;

export function initStore(id) {
  userId = String(id);
}

function conversationsKey() {
  return `${PREFIX}:u${userId}:conversations`;
}

function importedKey() {
  return `${PREFIX}:u${userId}:legacy_imported`;
}

function normalize(list) {
  return list
    .filter((item) => item && typeof item === 'object' && Array.isArray(item.messages))
    .map((item) => ({
      id: String(item.id || makeId()),
      title: String(item.title || '新对话'),
      messages: item.messages,
      createdAt: Number(item.createdAt) || Date.now(),
      updatedAt: Number(item.updatedAt) || Number(item.createdAt) || Date.now(),
    }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

export function makeId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

export function createConversation(title = '新对话') {
  const now = Date.now();
  return { id: makeId(), title, messages: [], createdAt: now, updatedAt: now };
}

export function loadConversations() {
  try {
    const raw = localStorage.getItem(conversationsKey());
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? normalize(list) : [];
  } catch {
    return [];
  }
}

export function saveConversations(list) {
  if (!userId) return;
  try {
    localStorage.setItem(conversationsKey(), JSON.stringify(list));
  } catch (err) {
    console.warn('本地保存对话失败', err);
  }
}

// ── 旧版本数据 ──

export function clearLegacyKeys() {
  for (const key of LEGACY_CLEAR_KEYS) {
    try {
      localStorage.removeItem(key);
    } catch {
      /* 忽略 */
    }
  }
}

export function legacyConversationCount() {
  try {
    const raw = localStorage.getItem(LEGACY_CHATS_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.length : 0;
  } catch {
    return 0;
  }
}

export function legacyAlreadyHandled() {
  try {
    return localStorage.getItem(importedKey()) === '1';
  } catch {
    return true;
  }
}

export function markLegacyHandled() {
  try {
    localStorage.setItem(importedKey(), '1');
  } catch {
    /* 忽略 */
  }
}

/** 把旧版（未分账号）的对话导入当前账号；返回导入条数。 */
export function importLegacyConversations() {
  let imported = [];
  try {
    const raw = localStorage.getItem(LEGACY_CHATS_KEY);
    const list = raw ? JSON.parse(raw) : [];
    imported = normalize(Array.isArray(list) ? list : []);
  } catch {
    imported = [];
  }
  if (!imported.length) return 0;

  const existing = loadConversations();
  const known = new Set(existing.map((item) => item.id));
  const merged = [...existing, ...imported.filter((item) => !known.has(item.id))];
  merged.sort((a, b) => b.updatedAt - a.updatedAt);
  saveConversations(merged);
  markLegacyHandled();
  return imported.length;
}
