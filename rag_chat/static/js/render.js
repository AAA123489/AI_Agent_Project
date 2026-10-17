// render.js — 消息与历史列表的 DOM 渲染

export function refreshIcons() {
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

export function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

export function renderMarkdown(text) {
  const source = text || '';
  if (window.marked && typeof window.marked.parse === 'function') {
    try {
      return window.marked.parse(source);
    } catch {
      /* 退回纯文本 */
    }
  }
  return escapeHtml(source).replace(/\n/g, '<br>');
}

export function truncate(text, max = 30) {
  const value = (text || '').trim();
  return value.length > max ? value.slice(0, max) + '…' : value;
}

// ── 历史列表分组 ──

const DAY_MS = 86400000;

export function groupConversations(conversations) {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const weekStart = todayStart - 6 * DAY_MS;
  const monthStart = todayStart - 29 * DAY_MS;

  const buckets = new Map([
    ['今天', []],
    ['7 天内', []],
    ['30 天内', []],
    ['更早', []],
  ]);

  for (const conv of conversations) {
    const time = conv.updatedAt || conv.createdAt || 0;
    if (time >= todayStart) buckets.get('今天').push(conv);
    else if (time >= weekStart) buckets.get('7 天内').push(conv);
    else if (time >= monthStart) buckets.get('30 天内').push(conv);
    else buckets.get('更早').push(conv);
  }

  return [...buckets.entries()]
    .filter(([, items]) => items.length)
    .map(([label, items]) => ({ label, items }));
}

// ── 消息 ──

export function createUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'flex w-full justify-end';

  const bubble = document.createElement('div');
  bubble.className =
    'max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-neutral-100 px-4 py-2.5 text-[15px] leading-7 text-neutral-900';
  bubble.textContent = text;
  row.appendChild(bubble);
  return row;
}

export function createAssistantMessage() {
  const row = document.createElement('div');
  row.className = 'flex w-full gap-3';
  row.innerHTML = `
    <div class="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-50 text-blue-600">
      <i data-lucide="sparkles" class="h-4 w-4"></i>
    </div>
    <div class="min-w-0 flex-1">
      <div class="msg-content prose-chat prose prose-sm max-w-none text-[15px] leading-7 text-neutral-800"></div>
      <div class="thinking-slot"></div>
      <div class="sources-slot"></div>
    </div>`;
  return row;
}

export function setContent(el, html) {
  const content = el.querySelector('.msg-content');
  if (!content) return;
  if (html) content.innerHTML = html;
  else content.textContent = '思考中…';
}

export function setThinking(el, steps, { open = false } = {}) {
  const slot = el.querySelector('.thinking-slot');
  if (!slot) return;
  if (!steps || !steps.length) {
    slot.innerHTML = '';
    return;
  }

  let details = slot.querySelector('[data-thinking]');
  if (!details) {
    slot.innerHTML = `
      <details data-thinking class="mt-2 rounded-xl border border-neutral-100 bg-neutral-50/70 px-3 py-2">
        <summary class="flex cursor-pointer list-none items-center gap-1.5 text-[12px] text-neutral-400 select-none hover:text-neutral-600">
          <span>💭 思考过程</span>
          <i data-lucide="chevron-down" class="h-3.5 w-3.5"></i>
        </summary>
        <div class="thinking-body mt-2 max-h-56 overflow-y-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-neutral-500"></div>
      </details>`;
    details = slot.querySelector('[data-thinking]');
    refreshIcons();
  }
  details.querySelector('.thinking-body').textContent = steps.join('\n');
  if (open) details.open = true;
}

function sourceItem(doc) {
  const meta = [doc.date, doc.year, doc.source || doc.category].filter(Boolean).join(' · ');
  const similarity =
    doc.similarity != null ? `相似度 ${(doc.similarity * 100).toFixed(1)}%` : '';
  const metaLine = [meta, similarity].filter(Boolean).join(' · ');
  const url = doc.url
    ? `<div class="mt-0.5"><a class="break-all text-[12px] text-blue-600 underline-offset-2 hover:underline" href="${escapeHtml(doc.url)}" target="_blank" rel="noopener">查看原文</a></div>`
    : '';
  const text = (doc.text || '').slice(0, 120);

  return `
    <div class="border-t border-neutral-100 pt-2 first:border-t-0 first:pt-0">
      ${metaLine ? `<div class="text-[11px] text-neutral-400">📅 ${escapeHtml(metaLine)}</div>` : ''}
      ${url}
      <div class="mt-0.5 line-clamp-3 text-[11px] leading-relaxed text-neutral-500">${escapeHtml(text)}…</div>
    </div>`;
}

export function setSources(el, docs) {
  const slot = el.querySelector('.sources-slot');
  if (!slot) return;
  if (!docs || !docs.length) {
    slot.innerHTML = '';
    return;
  }

  slot.innerHTML = `
    <details class="mt-2 rounded-xl border border-neutral-100 bg-neutral-50/70 px-3 py-2">
      <summary class="flex cursor-pointer list-none items-center gap-1.5 text-[12px] text-neutral-400 select-none hover:text-neutral-600">
        <span>📎 参考来源 (${docs.length})</span>
        <i data-lucide="chevron-down" class="h-3.5 w-3.5"></i>
      </summary>
      <div class="mt-2 space-y-2">${docs.map(sourceItem).join('')}</div>
    </details>`;
  refreshIcons();
}

export function setError(el, message) {
  const wrap = document.createElement('div');
  wrap.className = 'mt-2 rounded-lg bg-red-50 px-3 py-2 text-[12px] leading-relaxed text-red-600';
  wrap.textContent = `⚠️ ${message}`;
  el.appendChild(wrap);
}

export function renderEmptyState(container) {
  container.innerHTML = `
    <div class="mx-auto mt-12 max-w-xl text-center md:mt-20">
      <div class="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
        <i data-lucide="sparkles" class="h-6 w-6"></i>
      </div>
      <h2 class="text-[20px] font-semibold text-neutral-900">有什么可以帮你？</h2>
      <p class="mt-2 text-[13px] leading-relaxed text-neutral-400">
        问我关于河南工学院的招生、专业、校园生活等公开信息。<br>回答会标注参考来源，方便你核对原文。
      </p>
    </div>`;
  refreshIcons();
}

export function renderMessageList(container, conv) {
  container.innerHTML = '';
  if (!conv || !conv.messages.length) {
    renderEmptyState(container);
    return;
  }

  for (const message of conv.messages) {
    if (message.role === 'user') {
      container.appendChild(createUserMessage(message.content));
      continue;
    }

    const el = createAssistantMessage();
    if (message.content) setContent(el, renderMarkdown(message.content));
    if (message.thinking && message.thinking.length) setThinking(el, message.thinking);
    if (message.sources && message.sources.length) setSources(el, message.sources);
    if (message.error) setError(el, message.error);
    container.appendChild(el);
  }

  refreshIcons();
  scrollToBottom(container);
}

export function scrollToBottom(container) {
  container.scrollTop = container.scrollHeight;
}
