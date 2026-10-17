// login.js — 登录 / 注册同页切换

import { getAuthConfig, login, register } from './api.js';

const form = document.getElementById('authForm');
const usernameInput = document.getElementById('username');
const passwordInput = document.getElementById('password');
const inviteRow = document.getElementById('inviteRow');
const inviteInput = document.getElementById('inviteCode');
const errorMsg = document.getElementById('errorMsg');
const submitBtn = document.getElementById('submitBtn');
const formTitle = document.getElementById('formTitle');
const switchHint = document.getElementById('switchHint');
const switchLink = document.getElementById('switchLink');

let mode = 'login';
let config = { invite_required: false, min_password_len: 8, max_password_len: 64 };

function safeNext() {
  const value = new URLSearchParams(window.location.search).get('next');
  if (!value) return '/';
  // 只允许站内路径，防止开放重定向
  if (!value.startsWith('/') || value.startsWith('//')) return '/';
  return value;
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.classList.remove('hidden');
}

function hideError() {
  errorMsg.classList.add('hidden');
  errorMsg.textContent = '';
}

function setMode(nextMode) {
  mode = nextMode;
  const isRegister = mode === 'register';

  formTitle.textContent = isRegister ? '注册' : '登录';
  submitBtn.textContent = isRegister ? '注册并登录' : '登录';
  switchHint.textContent = isRegister ? '已经有账号？' : '还没有账号？';
  switchLink.textContent = isRegister ? '去登录' : '立即注册';
  passwordInput.autocomplete = isRegister ? 'new-password' : 'current-password';
  passwordInput.placeholder = `至少 ${config.min_password_len} 位`;
  inviteRow.classList.toggle('hidden', !(isRegister && config.invite_required));
  hideError();
  usernameInput.focus();
}

function setBusy(busy) {
  submitBtn.disabled = busy;
  usernameInput.disabled = busy;
  passwordInput.disabled = busy;
  inviteInput.disabled = busy;
  submitBtn.textContent = busy
    ? mode === 'register'
      ? '注册中…'
      : '登录中…'
    : mode === 'register'
      ? '注册并登录'
      : '登录';
}

function validate() {
  const username = usernameInput.value.trim();
  const password = passwordInput.value;

  if (!/^[A-Za-z]{2,10}$/.test(username)) {
    return '用户名必须为 2-10 个英文字母';
  }
  if (password.length < config.min_password_len) {
    return `密码至少 ${config.min_password_len} 个字符`;
  }
  if (password.length > config.max_password_len) {
    return `密码最多 ${config.max_password_len} 个字符`;
  }
  if (mode === 'register' && config.invite_required && !inviteInput.value.trim()) {
    return '请输入邀请码';
  }
  return null;
}

switchLink.addEventListener('click', (event) => {
  event.preventDefault();
  setMode(mode === 'login' ? 'register' : 'login');
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  hideError();

  const error = validate();
  if (error) {
    showError(error);
    return;
  }

  const username = usernameInput.value.trim();
  const password = passwordInput.value;

  setBusy(true);
  try {
    if (mode === 'login') {
      await login(username, password);
    } else {
      await register({
        username,
        password,
        inviteCode: inviteInput.value.trim(),
      });
    }
    window.location.replace(safeNext());
  } catch (err) {
    showError(err.message || '请求失败，请稍后重试');
    setBusy(false);
  }
});

getAuthConfig()
  .then((cfg) => {
    config = { ...config, ...cfg };
    setMode('login');
    if (cfg.invite_required) inviteRow.classList.toggle('hidden', true);
  })
  .catch(() => {
    setMode('login');
  });

setMode('login');
