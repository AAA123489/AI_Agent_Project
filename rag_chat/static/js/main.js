// main.js — 聊天页启动：鉴权、侧边栏、对话控制器

import { getMe, logout } from './api.js';
import { deleteConversation, initChat, newChat, openConversation } from './chat.js';
import { refreshIcons } from './render.js';
import { initSidebar, renderSidebar, setQuota, setUser } from './sidebar.js';
import {
  importLegacyConversations,
  initStore,
  legacyAlreadyHandled,
  legacyConversationCount,
  loadConversations,
  markLegacyHandled,
} from './store.js';

const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('sidebarOverlay');
const statusBadge = document.getElementById('statusBadge');

const isMobile = () => window.matchMedia('(max-width: 767px)').matches;

function setSidebarOpen(open) {
  if (isMobile()) {
    sidebar.classList.toggle('-translate-x-full', !open);
    overlay.classList.toggle('hidden', !open);
    document.body.classList.toggle('overflow-hidden', open);
  } else {
    sidebar.classList.toggle('md:hidden', !open);
    overlay.classList.add('hidden');
  }
}

document.getElementById('sidebarOpen').addEventListener('click', () => setSidebarOpen(true));
overlay.addEventListener('click', () => setSidebarOpen(false));
window.addEventListener('resize', () => {
  if (!isMobile()) {
    sidebar.classList.remove('-translate-x-full');
    overlay.classList.add('hidden');
  }
});

async function refreshQuota() {
  try {
    const me = await getMe();
    setQuota(me.quota);
  } catch {
    /* 额度显示失败不影响对话 */
  }
}

async function doLogout() {
  if (!window.confirm('确定退出登录吗？')) return;
  try {
    await logout();
  } finally {
    window.location.replace('/login');
  }
}

async function checkHealth() {
  try {
    const res = await fetch('/health');
    statusBadge.textContent = res.ok ? '服务正常' : '服务异常';
  } catch {
    statusBadge.textContent = '无法连接';
  }
}

async function boot() {
  let me;
  try {
    me = await getMe();
  } catch (err) {
    if (err.status === 401) {
      window.location.replace('/login?next=/');
      return;
    }
    document.getElementById('messages').innerHTML =
      '<p class="mt-10 text-center text-[13px] text-red-500">加载失败，请刷新页面重试</p>';
    return;
  }

  initStore(me.user.id);
  refreshIcons();
  initSidebar({
    onNewChat: () => {
      newChat();
      if (isMobile()) setSidebarOpen(false);
    },
    onSelect: (id) => {
      openConversation(id);
      if (isMobile()) setSidebarOpen(false);
    },
    onDelete: (id) => deleteConversation(id),
    onLogout: doLogout,
    onToggle: setSidebarOpen,
  });
  setUser(me.user);
  setQuota(me.quota);

  let conversations = loadConversations();

  // 旧版（不分账号）记录：只询问一次，明确同意才导入
  if (!legacyAlreadyHandled()) {
    const count = legacyConversationCount();
    if (count > 0) {
      const ok = window.confirm(
        `检测到旧版本（未分账号）的 ${count} 条对话记录。\n` +
          `是否导入当前账号「${me.user.username}」？\n\n` +
          '仅在本次确认后导入，不会自动同步到其他账号。'
      );
      if (ok) {
        importLegacyConversations();
        conversations = loadConversations();
      }
    }
    markLegacyHandled();
  }

  initChat({
    messagesEl: document.getElementById('messages'),
    formEl: document.getElementById('inputForm'),
    inputEl: document.getElementById('input'),
    sendBtn: document.getElementById('sendBtn'),
    titleEl: document.getElementById('chatTitle'),
    conversations,
    onUpdate: (list, currentId) => renderSidebar(list, currentId),
    onQuota: refreshQuota,
  });

  checkHealth();
}

boot();
