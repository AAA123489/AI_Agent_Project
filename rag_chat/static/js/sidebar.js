// sidebar.js — 侧边栏：新对话、历史分组、搜索、用户区

import { escapeHtml, groupConversations, refreshIcons } from './render.js';

let els = {};
let handlers = {};
let conversations = [];
let currentId = null;
let query = '';
const collapsedGroups = new Set();

export function initSidebar(callbacks) {
  handlers = callbacks || {};
  els = {
    history: document.getElementById('historyList'),
    newChat: document.getElementById('newChatBtn'),
    searchToggle: document.getElementById('searchToggle'),
    searchBox: document.getElementById('searchBox'),
    searchInput: document.getElementById('searchInput'),
    avatar: document.getElementById('userAvatar'),
    userName: document.getElementById('userName'),
    quota: document.getElementById('quotaHint'),
    logout: document.getElementById('logoutBtn'),
    close: document.getElementById('sidebarClose'),
  };

  els.newChat.addEventListener('click', () => handlers.onNewChat?.());
  els.logout.addEventListener('click', () => handlers.onLogout?.());
  els.close.addEventListener('click', () => handlers.onToggle?.(false));

  els.searchToggle.addEventListener('click', () => {
    const opening = els.searchBox.classList.contains('hidden');
    els.searchBox.classList.toggle('hidden', !opening);
    if (opening) {
      els.searchInput.focus();
    } else {
      els.searchInput.value = '';
      query = '';
      render();
    }
  });

  els.searchInput.addEventListener('input', () => {
    query = els.searchInput.value.trim().toLowerCase();
    render();
  });
}

export function setUser(user) {
  if (!user) return;
  const name = user.username || '';
  els.userName.textContent = name;
  els.avatar.textContent = name.slice(0, 1).toUpperCase();
}

export function setQuota(quota) {
  if (!els.quota) return;
  if (!quota) {
    els.quota.textContent = '';
    return;
  }
  if (!quota.limit) {
    els.quota.textContent = '今日不限次数';
    return;
  }
  els.quota.textContent = `今日剩余 ${quota.remaining} / ${quota.limit} 次`;
}

export function renderSidebar(list, activeId) {
  conversations = Array.isArray(list) ? list : [];
  currentId = activeId || null;
  render();
}

function render() {
  if (!els.history) return;
  els.history.innerHTML = '';

  const keyword = query;
  const filtered = keyword
    ? conversations.filter((conv) => conv.title.toLowerCase().includes(keyword))
    : conversations;

  if (!filtered.length) {
    const empty = document.createElement('div');
    empty.className = 'px-3 py-8 text-center text-[12px] text-neutral-400';
    empty.textContent = keyword ? '没有匹配的对话' : '暂无对话历史';
    els.history.appendChild(empty);
    refreshIcons();
    return;
  }

  for (const group of groupConversations(filtered)) {
    const section = document.createElement('section');
    section.className = 'mt-3';

    const header = document.createElement('div');
    header.className = 'flex items-center justify-between px-2 py-1';

    const label = document.createElement('span');
    label.className = 'text-[11px] font-medium text-neutral-400';
    label.textContent = group.label;
    header.appendChild(label);

    const more = document.createElement('button');
    more.type = 'button';
    more.title = '收起 / 展开';
    more.className =
      'flex h-6 w-6 items-center justify-center rounded text-neutral-300 transition hover:bg-neutral-200/60 hover:text-neutral-500';
    more.innerHTML = '<i data-lucide="more-horizontal" class="h-4 w-4"></i>';
    more.addEventListener('click', () => {
      if (collapsedGroups.has(group.label)) collapsedGroups.delete(group.label);
      else collapsedGroups.add(group.label);
      render();
    });
    header.appendChild(more);
    section.appendChild(header);

    if (!collapsedGroups.has(group.label)) {
      for (const conv of group.items) section.appendChild(renderItem(conv));
    }
    els.history.appendChild(section);
  }

  refreshIcons();
}

function renderItem(conv) {
  const active = conv.id === currentId;
  const item = document.createElement('div');
  item.className = [
    'group flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-[14px] transition',
    active
      ? 'bg-neutral-200/70 font-medium text-neutral-900'
      : 'text-neutral-700 hover:bg-neutral-200/50',
  ].join(' ');
  item.setAttribute('role', 'button');
  item.tabIndex = 0;
  item.title = conv.title;

  const title = document.createElement('span');
  title.className = 'min-w-0 flex-1 truncate';
  title.textContent = conv.title;
  item.appendChild(title);

  const del = document.createElement('button');
  del.type = 'button';
  del.title = '删除对话';
  del.className =
    'hidden h-5 w-5 shrink-0 items-center justify-center rounded text-neutral-400 hover:bg-neutral-300/60 hover:text-red-500 group-hover:flex';
  del.innerHTML = '<i data-lucide="x" class="h-3.5 w-3.5"></i>';
  del.addEventListener('click', (event) => {
    event.stopPropagation();
    if (window.confirm(`删除对话「${conv.title}」？此操作不可撤销。`)) {
      handlers.onDelete?.(conv.id);
    }
  });
  item.appendChild(del);

  item.addEventListener('click', () => handlers.onSelect?.(conv.id));
  item.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handlers.onSelect?.(conv.id);
    }
  });

  return item;
}

export function resetSearch() {
  query = '';
  if (els.searchInput) els.searchInput.value = '';
  els.searchBox?.classList.add('hidden');
}
