// chat.js — 对话状态与 SSE 流式渲染

import { streamChat } from './api.js';
import {
  renderMarkdown,
  renderMessageList,
  scrollToBottom,
  setContent,
  setError,
  setSources,
  setThinking,
  truncate,
} from './render.js';
import { createConversation, saveConversations } from './store.js';

let els = {};
let callbacks = {};
let conversations = [];
let currentId = null;
let streaming = false;

export function initChat(options) {
  els = {
    messages: options.messagesEl,
    form: options.formEl,
    input: options.inputEl,
    sendBtn: options.sendBtn,
    title: options.titleEl,
  };
  callbacks = {
    onUpdate: options.onUpdate,
    onQuota: options.onQuota,
  };
  conversations = options.conversations || [];
  currentId = conversations[0] ? conversations[0].id : null;

  els.form.addEventListener('submit', handleSubmit);
  els.input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  });
  els.input.addEventListener('input', resizeInput);

  renderCurrent();
  notify();
}

export const getConversations = () => conversations;
export const getCurrentId = () => currentId;
export const isBusy = () => streaming;

function currentConversation() {
  return conversations.find((conv) => conv.id === currentId) || null;
}

function touch(conv) {
  conv.updatedAt = Date.now();
  conversations = [conv, ...conversations.filter((item) => item.id !== conv.id)];
}

function notify() {
  const conv = currentConversation();
  els.title.textContent = conv ? conv.title : '新对话';
  callbacks.onUpdate?.(conversations, currentId);
}

function renderCurrent() {
  renderMessageList(els.messages, currentConversation());
}

function resizeInput() {
  els.input.style.height = 'auto';
  els.input.style.height = `${Math.min(160, els.input.scrollHeight)}px`;
}

function setStreaming(value) {
  streaming = value;
  els.sendBtn.disabled = value;
  els.input.disabled = value;
}

function ensureConversation() {
  let conv = currentConversation();
  if (conv) return conv;
  conv = createConversation();
  conversations.unshift(conv);
  currentId = conv.id;
  return conv;
}

// ── 对外操作 ──

export function newChat() {
  if (streaming) return false;
  const conv = currentConversation();
  if (conv && conv.messages.length === 0) {
    scrollToBottom(els.messages);
    return true;
  }

  const fresh = createConversation();
  conversations.unshift(fresh);
  currentId = fresh.id;
  saveConversations(conversations);
  renderCurrent();
  notify();
  els.input.focus();
  return true;
}

export function openConversation(id) {
  if (streaming || id === currentId) return;
  if (!conversations.some((conv) => conv.id === id)) return;
  currentId = id;
  renderCurrent();
  notify();
}

export function deleteConversation(id) {
  if (streaming && id === currentId) return;
  conversations = conversations.filter((conv) => conv.id !== id);
  if (currentId === id) {
    currentId = conversations[0] ? conversations[0].id : null;
  }
  saveConversations(conversations);
  renderCurrent();
  notify();
}

// ── 发送 ──

async function handleSubmit(event) {
  event?.preventDefault();
  if (streaming) return;

  const message = els.input.value.trim();
  if (!message) return;

  const conv = ensureConversation();
  // 服务端限制：最多 20 轮、单条 2000 字；超出直接 422，这里主动截断
  const history = conv.messages
    .filter((item) => item.content && item.role)
    .slice(-20)
    .map((item) => ({ role: item.role, content: item.content.slice(0, 2000) }));

  if (conv.messages.length === 0) conv.title = truncate(message, 30);

  conv.messages.push({ role: 'user', content: message });
  touch(conv);

  els.input.value = '';
  resizeInput();

  const assistant = { role: 'assistant', content: '', thinking: [], sources: null };
  conv.messages.push(assistant);
  saveConversations(conversations);

  renderMessageList(els.messages, conv);
  const assistantEl = els.messages.lastElementChild;
  setContent(assistantEl, '');
  setStreaming(true);
  notify();

  let acc = '';
  let steps = [];
  let sources = null;

  try {
    await streamChat({
      message,
      history,
      conversationId: conv.id,
      onEvent: (evt) => {
        if (evt.type === 'thinking') {
          steps.push(evt.step);
          setThinking(assistantEl, steps, { open: true });
        } else if (evt.type === 'sources') {
          sources = evt.docs || [];
          setSources(assistantEl, sources);
        } else if (evt.type === 'text') {
          acc += evt.content || '';
          setContent(assistantEl, renderMarkdown(acc));
          scrollToBottom(els.messages);
        } else if (evt.type === 'done') {
          if (Array.isArray(evt.thinking) && evt.thinking.length) steps = evt.thinking;
          if (Array.isArray(evt.sources) && evt.sources.length) sources = evt.sources;
        } else if (evt.type === 'error') {
          assistant.error = evt.message || '服务端返回错误';
          setError(assistantEl, assistant.error);
        }
      },
    });
  } catch (err) {
    if (err.status === 401) {
      window.location.replace('/login?next=/');
      return;
    }
    assistant.error =
      err.status === 429 ? err.message : `请求失败：${err.message}`;
    setError(assistantEl, assistant.error);
  } finally {
    setStreaming(false);
  }

  assistant.content = acc;
  assistant.thinking = steps;
  assistant.sources = sources;
  if (!acc && !assistant.error) assistant.error = '没有收到回答，请稍后重试';

  touch(conv);
  saveConversations(conversations);
  renderMessageList(els.messages, conv);
  notify();
  callbacks.onQuota?.();
  els.input.focus();
}
