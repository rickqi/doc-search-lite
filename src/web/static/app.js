/* doc-search Web — application logic */
(function () {
  'use strict';

  // ── State ──────────────────────────────────────────
  const state = {
    sessions: [],
    activeSessionId: null,
    eventSource: null,
    isProcessing: false,
    token: localStorage.getItem('doc_search_token') || '',
  };

  // ── Auth ───────────────────────────────────────────
  function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (state.token && state.token.length > 0) {
      headers['Authorization'] = 'Bearer ' + state.token;
    }
    return headers;
  }

  function updateAuthUI() {
    const input = document.getElementById('api-key-input');
    const badge = document.getElementById('auth-status');
    if (input) input.value = state.token;
    if (badge) {
      badge.textContent = state.token ? '🔒' : '🔓';
      badge.title = state.token ? '已认证' : '未认证 — 输入 API Token';
    }
  }

  document.getElementById('api-key-input').addEventListener('input', function(e) {
    state.token = e.target.value.trim();
    localStorage.setItem('doc_search_token', state.token);
    updateAuthUI();
    if (state.token) loadSessions();
  });

  // ── API wrapper with token ──────────────────────────
  async function apiFetch(url, options) {
    options = options || {};
    const h = new Headers();
    h.set('Content-Type', 'application/json');
    if (state.token) {
      var cleanToken = state.token.replace(/[^\x20-\x7E]/g, '');
      if (cleanToken) {
        h.set('Authorization', 'Bearer ' + cleanToken);
      }
    }
    if (options.headers) {
      delete options.headers;
    }
    options.headers = h;
    return fetch(url, options);
  }

  // ── DOM refs ───────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    sidebar: $('#sidebar'),
    sessionList: $('#session-list'),
    messageList: $('#message-list'),
    welcome: $('#welcome'),
    promptInput: $('#prompt-input'),
    sendBtn: $('#send-btn'),
    abortBtn: $('#abort-btn'),
    newSessionBtn: $('#new-session-btn'),
    toggleSidebar: $('#toggle-sidebar'),
    themeToggle: $('#theme-toggle'),
    statusDot: $('#status-dot'),
    indexPath: $('#index-path'),
    rawDir: $('#raw-dir'),
  };

  // ── Version ────────────────────────────────────────
  apiFetch('/health').then(r => r.json()).then(d => {
    var el = document.getElementById('app-version');
    if (el && d.version) el.textContent = 'v' + d.version;
  }).catch(() => {});

  // ── API Client ─────────────────────────────────────
  const api = {
    async listSessions() {
      const r = await apiFetch('/api/sessions');
      if (!r.ok) {
        if (r.status === 401) throw new Error('API 认证失败，请在右上角输入 Token');
        throw new Error('加载会话失败: ' + r.status);
      }
      return (await r.json()).sessions || [];
    },

    async createSession(indexPath, rawDir) {
      const params = new URLSearchParams({ index_path: indexPath });
      if (rawDir) params.set('raw_dir', rawDir);
      const r = await apiFetch('/api/sessions?' + params, { method: 'POST' });
      if (!r.ok) {
        if (r.status === 401) throw new Error('API 认证失败，请在右上角输入 Token');
        throw new Error('创建会话失败: ' + r.status);
      }
      return await r.json();
    },

    async getSession(sid) {
      const r = await apiFetch('/api/sessions/' + sid);
      if (!r.ok) {
        if (r.status === 404) return null; // expired
        throw new Error('加载会话失败: ' + r.status);
      }
      return await r.json();
    },

    async deleteSession(sid) {
      await apiFetch('/api/sessions/' + sid, { method: 'DELETE' });
    },

    async sendPrompt(sid, prompt, skill, mode) {
      const params = new URLSearchParams({ prompt: prompt });
      if (skill) params.set('skill', skill);
      if (mode && mode !== 'agent') params.set('mode', mode);
      const r = await apiFetch('/api/sessions/' + sid + '/prompt?' + params, { method: 'POST' });
      const data = await r.json();
      return data;  // now includes { status, session_id, mode }
    },

    async abortSession(sid) {
      await apiFetch('/api/sessions/' + sid + '/abort', { method: 'POST' });
    },

    connectEvents(sid, handlers) {
      const tokenParam = state.token ? '?token=' + encodeURIComponent(state.token) : '';
      const es = new EventSource('/api/sessions/' + sid + '/events' + tokenParam);

      es.onerror = function() {
        if (es.readyState === EventSource.CLOSED) {
          console.error('SSE connection closed');
        }
      };

      es.addEventListener('session_start', (e) => {
        handlers.onSessionStart?.(JSON.parse(e.data));
      });
      es.addEventListener('thinking', (e) => {
        handlers.onThinking?.(JSON.parse(e.data));
      });
      es.addEventListener('tool_call', (e) => {
        handlers.onToolCall?.(JSON.parse(e.data));
      });
      es.addEventListener('tool_result', (e) => {
        handlers.onToolResult?.(JSON.parse(e.data));
      });
      es.addEventListener('search_result', (e) => {
        handlers.onSearchResult?.(JSON.parse(e.data));
      });
      es.addEventListener('strategy_info', (e) => {
        handlers.onStrategyInfo?.(JSON.parse(e.data));
      });
      es.addEventListener('sufficiency_check', (e) => {
        handlers.onSufficiencyCheck?.(JSON.parse(e.data));
      });
      es.addEventListener('answer_chunk', (e) => {
        handlers.onAnswerChunk?.(JSON.parse(e.data));
      });
      es.addEventListener('answer_complete', (e) => {
        handlers.onAnswerComplete?.(JSON.parse(e.data));
        es.close();
      });
      es.addEventListener('error', (e) => {
        handlers.onError?.(JSON.parse(e.data || '{}'));
        es.close();
      });
      es.addEventListener('aborted', () => {
        handlers.onAbort?.();
        es.close();
      });

      es.onerror = () => {
        // SSE connection lost — will auto-reconnect unless explicitly closed
        if (es.readyState === EventSource.CLOSED) {
          handlers.onDisconnect?.();
        }
      };

      return es;
    },
  };

  // ── UI Helpers ─────────────────────────────────────
  function scrollToBottom() {
    dom.messageList.scrollTop = dom.messageList.scrollHeight;
  }

  function setProcessing(active) {
    state.isProcessing = active;
    // Show both abort and send — user can abort current or submit new
    dom.abortBtn.classList.toggle('hidden', !active);
    if (active) {
      dom.sendBtn.textContent = '重新提交';
      dom.sendBtn.classList.remove('hidden');
    } else {
      dom.sendBtn.textContent = '发送';
    }
    dom.statusDot.className = 'status-dot ' + (active ? 'thinking' : 'idle');
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ── Message Rendering ──────────────────────────────
  function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = 'message msg-' + type;
    const inner = document.createElement('div');
    inner.className = 'msg-content';
    inner.innerHTML = content;
    div.appendChild(inner);
    // Add copy button
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.title = '复制';
    btn.textContent = '复制';
    btn.addEventListener('click', () => {
      const text = inner.textContent;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = '已复制';
        setTimeout(() => btn.textContent = '复制', 1500);
      }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.left = '-9999px';
        document.body.appendChild(ta); ta.select();
        document.execCommand('copy'); document.body.removeChild(ta);
        btn.textContent = '已复制';
        setTimeout(() => btn.textContent = '复制', 1500);
      });
    });
    div.appendChild(btn);
    dom.messageList.appendChild(div);
    scrollToBottom();
    return div;
  }

  let _currentAnswer = null;
  let _toolCounts = { search: 0, read: 0, grep: 0, bash: 0, rerank: 0 };
  let _sufficiencyCount = 0;
  let _lastCoverage = null;

  function ensureAnswerElement() {
    if (!_currentAnswer) {
      _currentAnswer = addMessage('assistant', '');
    }
    return _currentAnswer.querySelector('.msg-content');
  }

  function appendAnswerChunk(chunk) {
    const el = ensureAnswerElement();
    el.innerHTML += escapeHtml(chunk).replace(/\n/g, '<br>');
    scrollToBottom();
  }

  function tryRenderReviewTable(el, answer) {
    // Detect if answer contains compliance review JSON with "results" array
    let json = null;
    try {
      // Try direct parse
      const parsed = JSON.parse(answer);
      if (parsed.results && Array.isArray(parsed.results)) json = parsed;
    } catch (e) {
      // Try extracting from markdown code blocks
      const m = answer.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
      if (m) {
        try { const p = JSON.parse(m[1]); if (p.results) json = p; } catch (e2) {}
      }
    }
    if (!json || !json.results) return;

    // Build review table
    const table = document.createElement('div');
    table.className = 'review-table';

    let html = '<table><thead><tr>' +
      '<th>#</th><th>Pass</th><th>违规原文</th><th>触发词</th><th>规则</th><th>建议</th>' +
      '</tr></thead><tbody>';

    json.results.forEach((r, i) => {
      const passIcon = r.pass === 'N' || r.pass === 'n'
        ? '<span class="review-fail">N</span>'
        : '<span class="review-pass">Y</span>';
      const hitCtx = (r.hit_context || '').substring(0, 120);
      const hitWord = (r.hit_word || '').substring(0, 40);
      const rule = (r.rule_display || r.rule_id || '').substring(0, 80);
      const sugg = (r.suggestion || '').substring(0, 80);

      html += '<tr class="' + (r.pass === 'N' ? 'row-fail' : 'row-pass') + '">' +
        '<td>' + (i + 1) + '</td>' +
        '<td>' + passIcon + '</td>' +
        '<td><code>' + escapeHtml(hitCtx) + '</code></td>' +
        '<td>' + escapeHtml(hitWord) + '</td>' +
        '<td class="rule-col">' + escapeHtml(rule) + '</td>' +
        '<td>' + escapeHtml(sugg) + '</td>' +
        '</tr>';
    });

    html += '</tbody></table>';

    // Summary
    const failCount = json.results.filter(r => r.pass === 'N' || r.pass === 'n').length;
    html += '<div class="review-summary">' +
      '审查结果：共 ' + json.results.length + ' 条规则，' +
      '<span class="review-fail">' + failCount + ' 条违规</span>，' +
      '<span class="review-pass">' + (json.results.length - failCount) + ' 条合规</span>' +
      '</div>';

    table.innerHTML = html;
    el.appendChild(table);
  }

  function finalizeAnswer(data) {
    var t = window.__i18n ? window.__i18n.t : function(k) { return k; };
    const el = ensureAnswerElement();

    // Try to detect and render compliance review JSON as a styled table
    tryRenderReviewTable(el, data.answer || '');

    // ── Pipeline Summary ──────────────────────────
    var toolParts = [];
    var toolOrder = ['search', 'read', 'grep', 'bash', 'rerank'];
    for (var ti = 0; ti < toolOrder.length; ti++) {
      var tn = toolOrder[ti];
      var tc = _toolCounts[tn] || 0;
      if (tc > 0) {
        toolParts.push(t('pipeline.' + tn) + '\u00d7' + tc);
      }
    }
    var hasPipeline = toolParts.length > 0 || _sufficiencyCount > 0 || data.draft_verified;
    if (hasPipeline) {
      var pipelineHtml = '<span class="pipeline-label">\uD83D\uDCCB ' + t('pipeline.label') + ':</span> ';
      pipelineHtml += toolParts.map(function(p) {
        return '<span class="pipeline-tag">' + p + '</span>';
      }).join('<span class="pipeline-sep">|</span>');

      if (_sufficiencyCount > 0) {
        pipelineHtml += '<span class="pipeline-sep">|</span>';
        pipelineHtml += '<span class="pipeline-tag">' +
          t('pipeline.sufficiency') + '\u00d7' + _sufficiencyCount +
          (_lastCoverage !== null ? ' <span class="pipeline-coverage">(' + t('pipeline.coverage') + '=' + _lastCoverage.toFixed(2) + ')</span>' : '') +
          '</span>';
      }

      if (data.draft_verified) {
        pipelineHtml += '<span class="pipeline-sep">|</span>';
        pipelineHtml += '<span class="pipeline-tag">' + t('pipeline.draft') +
          (_lastCoverage !== null ? ' <span class="pipeline-coverage">(' + t('pipeline.coverage') + '=' + _lastCoverage.toFixed(2) + ')</span>' : '') +
          '</span>';
      }

      var pipelineDiv = document.createElement('div');
      pipelineDiv.className = 'msg-pipeline';
      pipelineDiv.innerHTML = pipelineHtml;
      el.appendChild(pipelineDiv);
    }

    // ── Source Citations ──────────────────────────
    var MAX_SHOW_HITS = 5;
    var hits = data.search_hits || [];
    if (hits.length > 0) {
      var srcDiv = document.createElement('div');
      srcDiv.className = 'msg-sources';
      var label = hits.length === 1 ? '引用文档' : '引用文档（共 ' + hits.length + ' 个）';
      var showCount = Math.min(hits.length, MAX_SHOW_HITS);

      var hitsHtml = '<div class="src-label">' + label + '：</div>';
      for (var hi = 0; hi < showCount; hi++) {
        var h = hits[hi];
        var scoreBadge = h.score ? ' <span class="src-score">' + h.score.toFixed(0) + '</span>' : '';
        var snippet = h.snippet
          ? '<div class="src-snippet-quote">' + escapeHtml(h.snippet.slice(0, 150)) + '</div>'
          : '';
        var fbKey = 'fb-' + hi + '-' + (h.doc_id || h.source_path || hi);
        var fbButtons =
          '<span class="src-feedback" data-fb-key="' + escapeHtml(fbKey) + '">' +
          '<button class="fb-btn fb-up" data-rating="1" data-doc-id="' + escapeHtml(h.doc_id || '') + '" ' +
          'data-doc-title="' + escapeHtml(h.title || h.source_path || '') + '" title="Good result">\uD83D\uDC4D</button>' +
          '<button class="fb-btn fb-down" data-rating="-1" data-doc-id="' + escapeHtml(h.doc_id || '') + '" ' +
          'data-doc-title="' + escapeHtml(h.title || h.source_path || '') + '" title="Bad result">\uD83D\uDC4E</button>' +
          '</span>';
        hitsHtml += '<div class="src-item hit">' +
          '<span class="src-num">#' + (hi + 1) + scoreBadge + '</span> ' +
          '<div class="src-body">' +
          '<span class="src-title">' + escapeHtml(h.title || h.source_path || '') + '</span>' +
          '<span class="src-path">' + escapeHtml(h.source_path || h.doc_id || '') + '</span>' +
          snippet +
          '</div>' +
          fbButtons +
          '</div>';
      }
      // "还有 N 个" indicator
      if (hits.length > MAX_SHOW_HITS) {
        var moreText = t('sources.more').replace('{n}', hits.length - MAX_SHOW_HITS);
        hitsHtml += '<div class="src-more">' + escapeHtml(moreText) + '</div>';
      }
      srcDiv.innerHTML = hitsHtml;
      el.appendChild(srcDiv);
    } else if (data.sources && data.sources.length > 0) {
      // Fallback: plain string sources (backward compat)
      const srcDiv = document.createElement('div');
      srcDiv.className = 'msg-sources';
      srcDiv.innerHTML =
        '<div class="src-label">引用文档（共 ' + data.sources.length + ' 个）：</div>' +
        data.sources.map((s, i) =>
          '<div class="src-item">' +
          '<span class="src-num">#' + (i + 1) + '</span> ' +
          escapeHtml(s) +
          '</div>'
        ).join('');
      el.appendChild(srcDiv);
    } else {
      const noSrc = document.createElement('div');
      noSrc.className = 'msg-sources empty';
      noSrc.textContent = '(无引用文档 — 此为直接审查结果)';
      el.appendChild(noSrc);
    }

    // Stats line
    var statsParts = [
      (data.tokens_used || 0).toLocaleString() + ' tokens',
      (data.processing_time || 0).toFixed(1) + 's',
    ];
    if (data.complexity) {
      var compLabels = { simple: t('strategy.simple'), medium: t('strategy.medium'), complex: t('strategy.complex') };
      statsParts.push(compLabels[data.complexity] || data.complexity);
    }
    if (data.feedback_rounds > 0) {
      statsParts.push(t('stats.feedback') + data.feedback_rounds + t('stats.rounds'));
    }
    if (data.draft_verified) {
      statsParts.push('\u2705 ' + t('stats.draft_verified'));
    }
    const stats = document.createElement('div');
    stats.className = 'msg-stats';
    stats.textContent = statsParts.join('  \u00b7  ');
    el.appendChild(stats);

    // Show export button
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
      _lastResult = data;
      exportBtn.style.display = 'inline-block';
      exportBtn.textContent = '导出';
      exportBtn.onclick = () => exportResult(data);
    }

    _currentAnswer = null;
    // Reset pipeline tracking
    _toolCounts = { search: 0, read: 0, grep: 0, bash: 0, rerank: 0 };
    _sufficiencyCount = 0;
    _lastCoverage = null;
    setProcessing(false);
  }

  let _pendingTools = new Map();  // tool_call_id -> DOM element

  /**
   * Get concise argument summary for a tool call.
   */
  function getToolArgSummary(tool, args) {
    if (!args || typeof args !== 'object') return '';
    switch (tool) {
      case 'search':
        return args.query ? '"' + String(args.query).slice(0, 60) + '"' : '';
      case 'read':
        return args.doc_id ? String(args.doc_id).slice(0, 16) + '...'
          : args.source_path ? String(args.source_path).slice(0, 40) : '';
      case 'grep':
        return args.pattern ? '"' + String(args.pattern).slice(0, 60) + '"' : '';
      case 'bash':
        return args.command ? String(args.command).slice(0, 50) : '';
      case 'rerank':
        if (args.documents) return '(' + (Array.isArray(args.documents) ? args.documents.length : 0) + ' docs)';
        if (args.query && args.documents_count) return '(' + args.documents_count + ' docs)';
        return args.query ? '"' + String(args.query).slice(0, 40) + '"' : '';
      default:
        // Show first meaningful string/number value
        for (var k in args) {
          if (args[k] && (typeof args[k] === 'string' || typeof args[k] === 'number')) {
            return String(args[k]).slice(0, 50);
          }
        }
        return '';
    }
  }

  /**
   * Get tool icon emoji.
   */
  function getToolIcon(tool) {
    switch (tool) {
      case 'search': return '\uD83D\uDD0D';
      case 'read': return '\uD83D\uDCD6';
      case 'grep': return '\uD83D\uDD0E';
      case 'bash': return '\uD83D\uDCBB';
      case 'rerank': return '\uD83D\uDCCA';
      default: return '\uD83D\uDD27';
    }
  }

  function addToolCall(data) {
    // Hide welcome if visible
    if (dom.welcome.style.display !== 'none') dom.welcome.style.display = 'none';

    // Track tool counts for pipeline summary
    var toolName = data.tool || 'unknown';
    if (_toolCounts.hasOwnProperty(toolName)) {
      _toolCounts[toolName]++;
    }

    var t = window.__i18n ? window.__i18n.t : function(k) { return k; };
    var labelKey = 'tool.' + toolName;
    var toolLabel = t(labelKey) !== labelKey ? t(labelKey) : t('tool.default');
    var toolIcon = getToolIcon(toolName);
    var argSummary = getToolArgSummary(toolName, data.arguments || {});

    const div = document.createElement('div');
    div.className = 'message msg-tool';
    div.innerHTML = '<div class="tool-header">' +
      '<span class="tool-icon">' + toolIcon + '</span>' +
      '<span class="tool-label">' + escapeHtml(toolLabel) + '</span>' +
      '<span class="tool-args-summary">' + escapeHtml(argSummary) + '</span>' +
      '<span class="tool-status"><span class="spinner"></span></span>' +
      '</div>' +
      '<div class="tool-result"></div>';
    div.querySelector('.tool-header').addEventListener('click', () => {
      div.classList.toggle('expanded');
    });

    dom.messageList.appendChild(div);
    scrollToBottom();

    // Track by tool name + timestamp for result matching
    const key = data.tool + '_' + (data.timestamp || Date.now());
    _pendingTools.set(key, div);
    return { div, key };
  }

  function updateToolResult(data) {
    // Find most recent pending tool of this name
    let found = null;
    for (const [key, div] of _pendingTools) {
      if (key.startsWith(data.tool + '_')) {
        found = div;
        _pendingTools.delete(key);
        break;
      }
    }
    if (!found) return;

    const statusEl = found.querySelector('.tool-status');
    if (data.success) {
      statusEl.className = 'tool-status success';
      statusEl.textContent = '\u2705';
    } else {
      statusEl.className = 'tool-status error';
      statusEl.textContent = '\u274c';
    }

    const resultEl = found.querySelector('.tool-result');
    if (data.content_preview) {
      resultEl.textContent = data.content_preview;
      found.classList.add('expanded');
    }
  }

  // ── Agentic RAG Pipeline Events ────────────────

  function addStrategyBadge(data) {
    var t = window.__i18n ? window.__i18n.t : function(k) { return k; };
    var labels = { simple: t('strategy.simple'), medium: t('strategy.medium'), complex: t('strategy.complex') };
    var label = labels[data.complexity] || data.complexity;
    var toolCount = data.tool_calls_count || 0;

    var div = document.createElement('div');
    div.className = 'message msg-strategy';
    div.innerHTML = '<span class="strategy-badge strategy-' + (data.complexity || 'simple') + '">' +
      t('strategy.label') + label + ' (' + toolCount + ' ' + t('strategy.rounds') + ')' +
      '</span>';
    dom.messageList.appendChild(div);
    scrollToBottom();
  }

  function addSufficiencyMessage(data) {
    var t = window.__i18n ? window.__i18n.t : function(k) { return k; };
    var score = typeof data.coverage_score === 'number' ? data.coverage_score.toFixed(2) : '--';
    var statusIcon = data.sufficient ? '\u2705' : '\u23f3';
    var missing = (data.missing_aspects || []).length;
    var missingText = missing > 0 ? ' \u00b7 ' + missing + ' ' + t('sufficiency.missing') : '';

    // Track for pipeline summary
    _sufficiencyCount++;
    _lastCoverage = data.coverage_score;

    var div = document.createElement('div');
    div.className = 'message msg-sufficiency';
    div.innerHTML = '<span class="sufficiency-badge">' +
      statusIcon + ' ' + t('sufficiency.label') + data.round + ' \u00b7 ' +
      t('sufficiency.coverage') + score + missingText +
      '</span>';
    dom.messageList.appendChild(div);
    scrollToBottom();
  }

  // ── Session Management ─────────────────────────────
  function renderSessions() {
    dom.sessionList.innerHTML = '';
    state.sessions.forEach(s => {
      const div = document.createElement('div');
      div.className = 'session-item' + (s.id === state.activeSessionId ? ' active' : '');
      div.innerHTML =
        '<span class="si-delete" data-sid="' + s.id + '">&times;</span>' +
        '<div class="si-title">' + (escapeHtml(s.prompt) || '新会话') + '</div>' +
        '<div class="si-meta">' + s.messages_count + ' 条消息</div>';

      div.addEventListener('click', (e) => {
        if (e.target.classList.contains('si-delete')) {
          e.stopPropagation();
          deleteSession(s.id);
          return;
        }
        selectSession(s.id);
      });

      dom.sessionList.appendChild(div);
    });
  }

  async function loadSessions() {
    try {
      state.sessions = await api.listSessions();
      renderSessions();
    } catch (e) {
      console.error('Failed to load sessions:', e);
      // Don't alert on auto-load — user hasn't entered token yet
    }
  }

  async function createSession() {
    const indexPath = dom.indexPath.value.trim();
    if (!indexPath) {
      alert('请输入索引路径');
      return;
    }

    try {
      const { session_id } = await api.createSession(indexPath, dom.rawDir.value.trim() || undefined);
      await loadSessions();
      selectSession(session_id);
    } catch (e) {
      alert('创建会话失败: ' + e.message);
    }
  }

  function selectSession(sid) {
    state.activeSessionId = sid;

    // Clear chat
    dom.messageList.innerHTML = '';
    dom.welcome.style.display = 'none';
    _currentAnswer = null;
    _pendingTools.clear();

    renderSessions();

    // Load history
    api.getSession(sid).then(data => {
      if (!data) {
        // Session expired — clear and create new one
        state.activeSessionId = null;
        renderSessions();
        return;
      }
      if (data.messages && data.messages.length > 0) {
        data.messages.forEach(m => {
          if (m.role === 'user') {
            addMessage('user', escapeHtml(m.content));
          } else if (m.role === 'assistant') {
            addMessage('assistant', escapeHtml(m.content));
          }
        });
      } else {
        dom.welcome.style.display = '';
      }
    }).catch(console.error);
  }

  async function deleteSession(sid) {
    await api.deleteSession(sid);
    if (state.activeSessionId === sid) {
      state.activeSessionId = null;
      dom.messageList.innerHTML = '';
      dom.welcome.style.display = '';
    }
    if (state.eventSource && state.eventSource.url.includes(sid)) {
      state.eventSource.close();
      state.eventSource = null;
    }
    await loadSessions();
  }

  // ── Direct Search (non-agent modes) ──────────────

  async function directSearch(prompt, mode) {
    const indexPath = dom.indexPath.value.trim();
    const rawDir = dom.rawDir.value.trim();
    if (!indexPath && mode !== 'review' && mode !== 'direct') {
      addMessage('assistant', '<span style="color:var(--danger)">请先配置索引路径</span>');
      return;
    }

    const t0 = Date.now();

    try {
      var resp, data;
      if (mode === 'review' || mode === 'direct') {
        // review/direct modes: use /query/agent with mode param
        resp = await apiFetch('/query/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: prompt, mode: mode,
            index_path: indexPath || 'D:/docs/raw/制度/index',
            limit: 10,
          }),
        });
      } else if (mode === 'bm25') {
        resp = await apiFetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: prompt, index_path: indexPath, limit: 20 }),
        });
      } else {
        // grep / hybrid / tag — need raw_dir
        var endpoint = mode === 'grep' ? '/api/search/grep'
          : mode === 'hybrid' ? '/api/search/hybrid'
          : '/api/search/tag';
        resp = await apiFetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: prompt,
            index_path: indexPath,
            raw_dir: rawDir || indexPath.replace(/\\index$/, ''),
            limit: 20,
          }),
        });
      }

      if (!resp.ok) {
        var errText = await resp.text().catch(function() { return ''; });
        addMessage('assistant', '<span style="color:var(--danger)">搜索失败 (' + resp.status + '): ' + escapeHtml(errText.slice(0, 200)) + '</span>');
        return;
      }

      data = await resp.json();

      // review/direct modes return {answer, execution_mode, tokens_used}
      if (mode === 'review' || mode === 'direct') {
        var answer = data.answer || '无回答';
        var modeLabel = mode === 'review' ? '合规审查' : '联网搜索';
        var stats = (data.tokens_used || 0).toLocaleString() + ' tokens · ' + (data.processing_time || 0).toFixed(1) + 's';
        addMessage('assistant', '<div class="msg-mode-badge">' + modeLabel + '</div>' + escapeHtml(answer) + '<div class="msg-stats">' + stats + '</div>');
        return;
      }

      var results = data.results || [];
      var elapsed = ((Date.now() - t0) / 1000).toFixed(2);
      renderDirectResults(prompt, mode, results, elapsed);
    } catch (e) {
      addMessage('assistant', '<span style="color:var(--danger)">搜索错误: ' + escapeHtml(e.message) + '</span>');
    }
  }

  async function analyzeSearch(prompt, analyzeMode) {
    const indexPath = dom.indexPath.value.trim();
    const rawDir = dom.rawDir.value.trim();
    if (!indexPath) {
      addMessage('assistant', '<span style="color:var(--danger)">请先配置索引路径</span>');
      return;
    }

    var modeLabels = { compare: '📊 对比分析', extract: '🔍 信息提取', summarize: '📝 文档摘要', table: '📋 表格提取' };

    try {
      var resp = await apiFetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: prompt,
          index_path: indexPath,
          raw_dir: rawDir || indexPath.replace(/\\index$/, ''),
          mode: analyzeMode,
        }),
      });

      if (!resp.ok) {
        var errText = await resp.text().catch(function() { return ''; });
        addMessage('assistant', '<span style="color:var(--danger)">分析失败 (' + resp.status + '): ' + escapeHtml(errText.slice(0, 200)) + '</span>');
        return;
      }

      var data = await resp.json();
      var inner = document.createElement('div');
      inner.className = 'msg-content';

      var html = '<div class="direct-search-header">' +
        '<span class="direct-mode-badge">' + (modeLabels[analyzeMode] || analyzeMode) + '</span>' +
        '<span class="direct-query"> — "' + escapeHtml(prompt) + '"</span>' +
        '</div>';

      if (data.success) {
        html += '<div class="msg-assistant-text">' + (window.marked ? window.marked(data.answer) : escapeHtml(data.answer)) + '</div>';
        if (data.sources && data.sources.length) {
          html += '<div class="msg-sources"><strong>来源:</strong> ' + data.sources.map(function(s) { return escapeHtml(s); }).join(', ') + '</div>';
        }
        var t = data.processing_time ? data.processing_time.toFixed(1) : '?';
        var tk = data.tokens_used || 0;
        html += '<div class="msg-stats">' + t + 's · ' + tk + ' tokens</div>';
      } else {
        html += '<div style="color:var(--danger)">' + escapeHtml(data.error || '分析失败') + '</div>';
      }

      inner.innerHTML = html;
      var div = document.createElement('div');
      div.className = 'message msg-assistant';
      div.appendChild(inner);
      dom.messageList.appendChild(div);
      scrollToBottom();
    } catch (e) {
      addMessage('assistant', '<span style="color:var(--danger)">分析错误: ' + escapeHtml(e.message) + '</span>');
    }
  }

  function renderDirectResults(query, mode, results, elapsed) {
    var modeLabels = { bm25: 'BM25 搜索', grep: 'Grep 搜索', hybrid: '混合搜索', tag: '标签搜索' };
    var div = document.createElement('div');
    div.className = 'message msg-assistant';
    var inner = document.createElement('div');
    inner.className = 'msg-content';

    // Mode header
    var html = '<div class="direct-search-header">' +
      '<span class="direct-mode-badge">' + escapeHtml(modeLabels[mode] || mode) + '</span> ' +
      '<span class="direct-count">' + results.length + ' 条结果</span>' +
      '<span class="direct-query"> — "' + escapeHtml(query) + '"</span>' +
      '</div>';

    if (results.length === 0) {
      html += '<div class="direct-empty">未找到匹配文档</div>';
    } else {
      html += '<div class="direct-results">';
      results.forEach(function(r, i) {
        var score = typeof r.score === 'number' ? ' <span class="src-score">' + r.score.toFixed(0) + '</span>' : '';
        var title = r.title || r.source_path || 'Untitled';
        var source = r.source_path || r.doc_id || '';
        var snippet = r.snippet ? escapeHtml(r.snippet.slice(0, 200)) : '';
        html += '<div class="direct-item">' +
          '<span class="src-num">#' + (i + 1) + score + '</span>' +
          '<div class="src-body">' +
          '<span class="src-title">' + escapeHtml(title) + '</span>' +
          (source ? '<span class="src-path">' + escapeHtml(source) + '</span>' : '') +
          (snippet ? '<div class="src-snippet-quote">' + snippet + '</div>' : '') +
          '</div></div>';
      });
      html += '</div>';
    }

    // Stats
    html += '<div class="msg-stats">' + elapsed + 's · ' + results.length + ' results · ' + mode + '</div>';

    inner.innerHTML = html;
    div.appendChild(inner);
    dom.messageList.appendChild(div);
    scrollToBottom();
  }

  // ── Query Execution ────────────────────────────────
  async function sendPrompt(prompt) {
    if (!prompt.trim()) return;

    // If already processing, abort current query first
    if (state.isProcessing) {
      if (state.activeSessionId) await api.abortSession(state.activeSessionId);
      if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
      setProcessing(false);
    }

    // Hide welcome (keep existing messages)
    dom.welcome.style.display = 'none';

    // Show user message (always append, never clear)
    addMessage('user', escapeHtml(prompt));
    _lastQuery = prompt;
    _lastIndexPath = dom.indexPath.value.trim();
    dom.promptInput.value = '';

    // Determine mode
    var mode = document.getElementById('search-mode-select')?.value || 'agent';

    // ── Analyze modes: auto-search + analysis ──
    if (mode.startsWith('analyze-')) {
      setProcessing(true);
      await analyzeSearch(prompt, mode.replace('analyze-', ''));
      setProcessing(false);
      return;
    }

    // ── Non-agent modes: direct search (no session needed) ──
    if (mode !== 'agent') {
      setProcessing(true);
      await directSearch(prompt, mode);
      setProcessing(false);
      return;
    }

    // ── Agent mode: session + SSE stream ──
    if (!state.activeSessionId) {
      await createSession();
      if (!state.activeSessionId) return;
    }

    var sid = state.activeSessionId;

    setProcessing(true);
    _currentAnswer = null;
    _pendingTools.clear();
    _toolCounts = { search: 0, read: 0, grep: 0, bash: 0, rerank: 0 };
    _sufficiencyCount = 0;
    _lastCoverage = null;

    try {
      const skill = document.getElementById('skill-select')?.value || '';
      const resp = await api.sendPrompt(sid, prompt, skill, 'agent');
      const detectedMode = resp.mode || 'tool_loop';

      // Show mode badge
      const modeLabel = detectedMode === 'review'
        ? '合规审查模式（直接LLM）'
        : '文档搜索模式（Agent）';
      dom.statusDot.title = modeLabel;

      // Connect SSE
      if (state.eventSource) state.eventSource.close();
      clearTrace();
      state.eventSource = api.connectEvents(sid, {
        onSessionStart(data) {
          console.log('Session started:', data.session_id);
          trace.startTime = performance.now();
        },
        onThinking(data) {
          const div = document.createElement('div');
          div.className = 'message msg-thinking';
          div.textContent = data.message || '正在分析...';
          dom.messageList.appendChild(div);
          scrollToBottom();
        },
        onToolCall(data) {
          addToolCall(data);
          trace.onToolCall(data);
        },
        onToolResult(data) {
          updateToolResult(data);
          trace.onToolResult(data);
        },
        onSearchResult(data) {
          trace.onSearchResult(data);
        },
        onStrategyInfo(data) {
          addStrategyBadge(data);
        },
        onSufficiencyCheck(data) {
          addSufficiencyMessage(data);
        },
        onAnswerChunk(data) {
          appendAnswerChunk(data.content || '');
        },
        onAnswerComplete(data) {
          finalizeAnswer(data);
          trace.onAnswerComplete(data);
          state.eventSource = null;
          loadSessions();
        },
        onError(data) {
          addMessage('assistant', '<span style="color:var(--danger)">' +
            escapeHtml(data.message || '执行出错') + '</span>');
          setProcessing(false);
          state.eventSource = null;
        },
        onAbort() {
          setProcessing(false);
          state.eventSource = null;
        },
        onDisconnect() {
          setProcessing(false);
          state.eventSource = null;
        },
      });
    } catch (e) {
      addMessage('assistant', '<span style="color:var(--danger)">发送失败: ' +
        escapeHtml(e.message) + '</span>');
      setProcessing(false);
    }
  }

  async function abortCurrent() {
    if (!state.activeSessionId) return;
    await api.abortSession(state.activeSessionId);
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    setProcessing(false);
  }

  // ── Event Handlers ─────────────────────────────────
  dom.sendBtn.addEventListener('click', () => {
    sendPrompt(dom.promptInput.value);
  });

  dom.abortBtn.addEventListener('click', abortCurrent);

  dom.promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt(dom.promptInput.value);
    }
    if (e.key === 'Escape') {
      abortCurrent();
    }
  });

  dom.newSessionBtn.addEventListener('click', createSession);

  // Advanced search toggle
  const advancedToggle = document.getElementById('advanced-toggle');
  const advancedOptions = document.getElementById('advanced-options');
  if (advancedToggle && advancedOptions) {
    advancedToggle.addEventListener('click', () => {
      advancedOptions.classList.toggle('hidden');
      advancedToggle.classList.toggle('active');
    });
    // Update agent badge when mode changes
    const modeSelect = document.getElementById('search-mode-select');
    const agentBadge = document.getElementById('agent-mode-badge');
    if (modeSelect && agentBadge) {
      modeSelect.addEventListener('change', () => {
        var mode = modeSelect.value;
        if (mode === 'agent') {
          agentBadge.textContent = 'Agent';
          agentBadge.className = '';
          agentBadge.id = 'agent-mode-badge';
        } else {
          var labels = { bm25: 'BM25', grep: 'Grep', hybrid: 'Hybrid', tag: 'Tag' };
          agentBadge.textContent = labels[mode] || mode;
          agentBadge.className = 'mode-override';
          agentBadge.id = 'agent-mode-badge';
        }
      });
    }
  }

  dom.toggleSidebar.addEventListener('click', () => {
    dom.sidebar.classList.toggle('collapsed');
  });

  dom.themeToggle.addEventListener('click', () => {
    const html = document.documentElement;
    const next = html.dataset.theme === 'light' ? 'dark' : 'light';
    html.dataset.theme = next;
    dom.themeToggle.textContent = next === 'light' ? '\u2600' : '\u263e';
    localStorage.setItem('doc-search-theme', next);
  });

  // ── Quick send (from chips) ─────────────────────────
  window.sendQuick = function (query) {
    dom.promptInput.value = query;
    sendPrompt(query);
  };

  // ── DB Panel ──────────────────────────────────────
  const dbDom = {
    panel: $('#panel-database'),
    panelSessions: $('#panel-sessions'),
    total: $('#db-total'), success: $('#db-success'),
    failed: $('#db-failed'), pending: $('#db-pending'),
    tokenSummary: $('#db-token-summary'),
    statusFilter: $('#db-status-filter'),
    filesList: $('#db-files-list'),
    refreshBtn: $('#db-refresh-btn'),
  };

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      dbDom.panelSessions.classList.toggle('active', tab === 'sessions');
      dbDom.panel.classList.toggle('active', tab === 'database');
      if (tab === 'database') loadDbData();
    });
  });

  async function loadDbData() {
    const rawDir = dom.rawDir.value.trim() || dom.indexPath.value.trim().replace(/\\index$/, '');
    if (!rawDir) {
      dbDom.filesList.innerHTML = '<div style="padding:8px;color:var(--text-muted);">请先在下方配置 Raw 目录</div>';
      return;
    }
    const enc = encodeURIComponent(rawDir);
    try {
      // Load stats
      const statsResp = await apiFetch('/api/db/' + enc + '/stats');
      const stats = await statsResp.json();
      dbDom.total.textContent = stats.file_total || '--';
      dbDom.success.textContent = stats.by_status?.success || '--';
      dbDom.failed.textContent = stats.by_status?.failed || '--';
      dbDom.pending.textContent = stats.by_status?.pending || '--';

      // Load token summary
      try {
        const tokenResp = await apiFetch('/api/db/' + enc + '/token/summary?days=7');
        const tokenData = await tokenResp.json();
        const total = tokenData.total || {};
        const cost = (total.cost_millicents || 0) / 100000;
        dbDom.tokenSummary.innerHTML =
          '7天 Token: <b>' + (total.total_tokens || 0).toLocaleString() + '</b>' +
          (cost > 0 ? ' | ¥' + cost.toFixed(4) : '');

        // Load daily chart data
        if (total.total_tokens > 0) {
          loadTokenChart(enc);
        }
      } catch (e) { dbDom.tokenSummary.innerHTML = ''; }

      // Load files
      const status = dbDom.statusFilter.value;
      const params = new URLSearchParams({ limit: 100 });
      if (status) params.set('status', status);
      const filesResp = await apiFetch('/api/db/' + enc + '/files?' + params);
      const filesData = await filesResp.json();
      renderFileList(filesData.files || []);
    } catch (e) {
      dbDom.filesList.innerHTML = '<div style="padding:8px;color:var(--danger);">加载失败: ' + e.message + '</div>';
    }
  }

  function renderFileList(files) {
    if (!files.length) {
      dbDom.filesList.innerHTML = '<div style="padding:8px;color:var(--text-muted);">无匹配文件</div>';
      return;
    }
    dbDom.filesList.innerHTML = files.map(f => {
      const err = f.last_error ? ' title="' + escapeHtml(f.last_error) + '"' : '';
      const size = f.file_size ? (f.file_size / 1024).toFixed(0) + 'KB' : '';
      return '<div class="db-file-row"' + err + '>' +
        '<span class="fname">' + escapeHtml(f.filename) + '</span>' +
        '<span style="font-size:10px;color:var(--text-muted)">' + size + '</span>' +
        '<span class="fstatus ' + f.status + '">' + f.status + '</span>' +
        '</div>';
    }).join('');
  }

  dbDom.statusFilter.addEventListener('change', loadDbData);
  dbDom.refreshBtn.addEventListener('click', loadDbData);

  // ── Token Usage Chart ───────────────────────────

  let _tokenChart = null;

  async function loadTokenChart(encRawDir) {
    try {
      const resp = await apiFetch('/api/db/' + encRawDir + '/token/daily?days=7&format=json');
      const data = await resp.json();
      const days = data.days_data || [];
      if (days.length < 2) return;

      const labels = days.map(d => d.date?.slice(5) || '');
      const tokens = days.map(d => d.total_tokens || 0);
      const inputTokens = days.map(d => d.input_tokens || 0);
      const outputTokens = days.map(d => d.output_tokens || 0);

      const container = document.getElementById('db-token-chart-container');
      const canvas = document.getElementById('db-token-chart');
      container.style.display = 'block';

      if (_tokenChart) _tokenChart.destroy();
      _tokenChart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            { label: '输入', data: inputTokens, backgroundColor: '#4fc3f7', borderRadius: 2 },
            { label: '输出', data: outputTokens, backgroundColor: '#81c784', borderRadius: 2 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: true, position: 'top', labels: { boxWidth: 10, font: { size: 9 }, color: '#888' } },
          },
          scales: {
            x: { ticks: { font: { size: 8 }, color: '#666' }, grid: { display: false } },
            y: { ticks: { font: { size: 8 }, color: '#666', callback: v => v >= 1000 ? (v/1000).toFixed(0)+'k' : v }, grid: { color: '#333' } },
          },
        },
      });
    } catch (e) {
      // Chart load failure is non-critical; summary text is sufficient.
    }
  }


  // ── Upload ───────────────────────────────────────  // ── Init ───────────────────────────────────────────
  function init() {
    // Restore theme
    const saved = localStorage.getItem('doc-search-theme');
    if (saved) {
      document.documentElement.dataset.theme = saved;
      dom.themeToggle.textContent = saved === 'light' ? '\u2600' : '\u263e';
    }

    // Pre-fill index path from URL query params (passed by CLI --web)
    const params = new URLSearchParams(window.location.search);
    const qIndex = params.get('index_path');
    const qRaw = params.get('raw_dir');
    if (qIndex) {
      dom.indexPath.value = qIndex;
      // Also save to localStorage for persistence across refreshes
      localStorage.setItem('doc-search-index-path', qIndex);
    } else {
      // Fallback: restore from localStorage
      const savedIndex = localStorage.getItem('doc-search-index-path');
      if (savedIndex) dom.indexPath.value = savedIndex;
    }
    if (qRaw) {
      dom.rawDir.value = qRaw;
      localStorage.setItem('doc-search-raw-dir', qRaw);
    } else {
      const savedRaw = localStorage.getItem('doc-search-raw-dir');
      if (savedRaw) dom.rawDir.value = savedRaw;
    }

    // Load sessions
    loadSessions();

    // Periodic refresh
    setInterval(loadSessions, 30000);

    // ── Trace panel toggle ────────────────────────────
    const traceToggle = document.getElementById('trace-toggle-btn');
    const tracePanel = document.getElementById('trace-panel');
    if (traceToggle && tracePanel) {
      traceToggle.addEventListener('click', function(e) {
        e.stopPropagation();
        tracePanel.classList.toggle('collapsed');
      });
      tracePanel.querySelector('.trace-panel-header')?.addEventListener('click', function(e) {
        if (e.target.closest('.trace-panel-actions')) return;
        tracePanel.classList.toggle('collapsed');
      });
    }

    // ── Upload handlers ──────────────────────────────
    initUpload();

    // Restore auth state
    updateAuthUI();
  }

  function initUpload() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('upload-input');
    const progressDiv = document.getElementById('upload-progress');
    const stageEl = document.getElementById('upload-stage');
    const barEl = document.getElementById('upload-bar');
    const statusEl = document.getElementById('upload-status-text');

    if (!dropzone || !fileInput) return;

    // File input change
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        uploadFiles(fileInput.files);
      }
    });

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer.files.length > 0) {
        uploadFiles(e.dataTransfer.files);
      }
    });

    async function uploadFiles(fileList) {
      const rawDir = localStorage.getItem('docsearch_raw_dir') || '';
      const idxDir = localStorage.getItem('docsearch_index_path') || '';
      if (!rawDir) {
        alert('请先在左侧配置 Raw 目录路径');
        return;
      }

      const formData = new FormData();
      for (const f of fileList) {
        formData.append('files', f);
      }
      formData.append('raw_dir', rawDir);
      formData.append('index_dir', idxDir || rawDir + '/index');

      // Show progress
      dropzone.classList.add('hidden');
      progressDiv.classList.remove('hidden');
      stageEl.textContent = '上传中...';
      barEl.value = 0;
      statusEl.textContent = `0/${fileList.length}`;

      try {
        const resp = await apiFetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.job_id) {
          listenUploadProgress(data.job_id, fileList.length);
        }
      } catch (err) {
        stageEl.textContent = '上传失败: ' + err.message;
        resetUpload();
      }
    }

    function listenUploadProgress(jobId, totalFiles) {
      const es = new EventSource(`/api/upload/${jobId}`);

      es.addEventListener('progress', (e) => {
        const p = JSON.parse(e.data);
        stageEl.textContent = p.stage === 'converting' ? `转换中: ${p.current_file}` :
                              p.stage === 'indexing' ? '建立索引中...' : p.stage;
        barEl.value = p.stage === 'converting' ? Math.round(p.current / p.total * 70) :
                      p.stage === 'indexing' ? 85 : 0;
        statusEl.textContent = `${p.success_count || 0} 成功 / ${p.failed_count || 0} 失败`;
      });

      es.addEventListener('complete', () => {
        stageEl.textContent = '✓ 完成';
        barEl.value = 100;
        statusEl.textContent = '刷新查看...';
        es.close();
        setTimeout(resetUpload, 3000);
        // Refresh DB stats and files
        if (typeof loadDBStats === 'function') loadDBStats();
        if (typeof loadDBFiles === 'function') loadDBFiles();
      });

      es.addEventListener('error', (e) => {
        stageEl.textContent = '上传失败';
        es.close();
        setTimeout(resetUpload, 3000);
      });
    }

    function resetUpload() {
      dropzone.classList.remove('hidden');
      progressDiv.classList.add('hidden');
      fileInput.value = '';
    }
  }

  // ── Export ────────────────────────────────────────

  let _lastResult = null;
  let _lastQuery = '';
  let _lastIndexPath = '';

  function exportResult(data) {
    if (!data || !data.answer) return;

    var parts = ['# ' + new Date().toISOString().slice(0, 10) + ' Query Result\n'];

    // Answer
    parts.push(data.answer || '');
    parts.push('');

    // Pipeline summary
    var toolOrder = ['search', 'read', 'grep', 'bash', 'rerank'];
    var toolParts = [];
    for (var i = 0; i < toolOrder.length; i++) {
      var tn = toolOrder[i];
      var tc = _toolCounts[tn] || 0;
      if (tc > 0) toolParts.push(tn + '\u00d7' + tc);
    }
    if (toolParts.length > 0 || data.draft_verified) {
      var pipelineLine = '\uD83D\uDCCB Pipeline: ' + toolParts.join(' | ');
      if (data.draft_verified) pipelineLine += ' | draft verified';
      parts.push(pipelineLine);
      parts.push('');
    }

    // Source citations
    var hits = data.search_hits || [];
    if (hits.length > 0) {
      parts.push('---');
      parts.push('');
      parts.push('\uD83D\uDCCE Sources (' + hits.length + '):');
      for (var hi = 0; hi < hits.length; hi++) {
        var h = hits[hi];
        var scoreStr = h.score ? ' ' + h.score.toFixed(3) : '';
        parts.push('  ' + (hi + 1) + '. **' + (h.title || h.source_path || '') + '**' + scoreStr);
        if (h.snippet) parts.push('     > ' + h.snippet.slice(0, 150));
        if (h.source_path) parts.push('     ' + h.source_path);
      }
      parts.push('');
    }

    // Stats
    parts.push('---');
    parts.push('Tokens: ' + (data.tokens_used || 0) + ' | Time: ' + (data.processing_time || 0).toFixed(1) + 's');
    if (data.complexity) parts.push('Complexity: ' + data.complexity);
    if (data.feedback_rounds > 0) parts.push('Feedback rounds: ' + data.feedback_rounds);

    const text = parts.join('\n');
    const blob = new Blob([text], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'doc-search-result.md';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ── Search Feedback ──────────────────────────────

  function showFeedbackToast(msg) {
    var toast = document.createElement('div');
    toast.className = 'fb-toast';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(function() { toast.classList.add('fb-toast-show'); }, 10);
    setTimeout(function() {
      toast.classList.remove('fb-toast-show');
      setTimeout(function() { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 300);
    }, 1500);
  }

  async function submitFeedback(btn) {
    var rating = parseInt(btn.getAttribute('data-rating'), 10);
    var docId = btn.getAttribute('data-doc-id') || '';
    var docTitle = btn.getAttribute('data-doc-title') || '';
    var wrapper = btn.closest('.src-feedback');
    if (wrapper && wrapper.classList.contains('fb-voted')) return;

    if (wrapper) {
      wrapper.classList.add('fb-voted');
      wrapper.querySelectorAll('.fb-btn').forEach(function(b) { b.disabled = true; });
      btn.classList.add('fb-selected');
    }

    try {
      await apiFetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: _lastQuery,
          rating: rating,
          doc_id: docId || undefined,
          doc_title: docTitle || undefined,
          index_path: _lastIndexPath || undefined,
        }),
      });
    } catch (e) { /* fire-and-forget */ }

    showFeedbackToast(rating > 0 ? 'Thanks for the feedback!' : 'Thanks — we\'ll improve this result.');
  }

  // ── Trace Collector ─────────────────────────────────────────
  const trace = {
    steps: [],
    searchResults: [],
    startTime: null,
    toolStartTimes: {},

    clear() {
      this.steps = [];
      this.searchResults = [];
      this.startTime = performance.now();
      this.toolStartTimes = {};
      const body = document.getElementById('trace-panel-body');
      if (body) {
        body.innerHTML = '<div class="trace-empty" data-i18n="trace.waiting">\u68c0\u7d22\u4e2d...</div>';
      }
      const stats = document.getElementById('trace-stats');
      if (stats) stats.textContent = '';
    },

    onToolCall(data) {
      const tool = data.tool || data.name || '?';
      this.toolStartTimes[tool + '_' + this.steps.length] = performance.now();
      const row = this._addRow(tool, '\u2026', 'running', data.arguments?.query || '');
      this.steps.push({ tool, status: 'running', row, startTime: performance.now() });
    },

    onToolResult(data) {
      const tool = data.tool || data.name || '?';
      for (let i = this.steps.length - 1; i >= 0; i--) {
        const s = this.steps[i];
        if (s.tool === tool && s.status === 'running') {
          s.status = 'done';
          const dur = performance.now() - s.startTime;
          const detail = data.content_preview
            ? data.content_preview.substring(0, 80)
            : (data.success !== false ? '\u2713' : '\u2717');
          this._updateRow(s.row, dur, detail);
          break;
        }
      }
      const panel = document.getElementById('trace-panel');
      if (panel && panel.classList.contains('collapsed')) {
        panel.classList.remove('collapsed');
      }
    },

    onSearchResult(data) {
      if (data.results) {
        const titles = data.results.slice(0, 3).map(r => r.title || r.doc_id || '?').join(', ');
        this._addRow('\u637c\u7d22', '\u2713', 0, titles + (data.results.length > 3 ? ' (+' + (data.results.length - 3) + ')' : ''));
      }
    },

    onAnswerComplete(data) {
      const body = document.getElementById('trace-panel-body');
      if (!body) return;
      const empty = body.querySelector('.trace-empty');
      if (empty) empty.remove();

      const total = data.processing_time || data.step_timings?.total_ms || 0;
      const hits = data.search_hits || [];
      const tools = data.tool_calls || [];
      const tokens = data.tokens_used || 0;

      const summary = document.createElement('div');
      summary.className = 'trace-summary';
      summary.innerHTML =
        '<span>\u23f1 <span class="stat-value">' + (total > 0 ? (total / 1000).toFixed(1) + 's' : '\u2014') + '</span></span>' +
        '<span>\ud83d\udd27 <span class="stat-value">' + tools.length + '</span> \u6b21\u8c03\u7528</span>' +
        (hits.length ? '<span>\ud83d\udcc4 <span class="stat-value">' + hits.length + '</span> \u7bc7\u6587\u6863</span>' : '') +
        (tokens ? '<span>\ud83d\udcca <span class="stat-value">' + (tokens / 1000).toFixed(1) + 'K</span> tokens</span>' : '');
      body.appendChild(summary);
    },

    _addRow(tool, duration, status, detail) {
      const body = document.getElementById('trace-panel-body');
      if (!body) return null;
      const empty = body.querySelector('.trace-empty');
      if (empty) empty.remove();

      const row = document.createElement('div');
      row.className = 'trace-row' + (status === 'running' ? ' trace-running' : '');
      const iconMap = { search: '\ud83d\udd0d', read: '\ud83d\udcd6', grep: '\ud83d\udd0e', rerank: '\ud83d\udcca',
        summarize: '\ud83d\udcdd', bash: '\ud83d\udcbb', think: '\ud83d\udcad', _llm_call: '\ud83e\udd16', chat: '\ud83e\udd16' };
      const icon = iconMap[tool] || '\u2699';

      const durText = status === 'running' ? '\u2026' : (typeof duration === 'number' ? duration.toFixed(0) + 'ms' : duration);
      row.innerHTML =
        '<span class="trace-icon">' + icon + '</span>' +
        '<span class="trace-tool">' + this._esc(tool) + '</span>' +
        '<span class="trace-duration">' + durText + '</span>' +
        '<span class="trace-detail' + (status === 'running' ? '' : ' highlight') + '">' + this._esc(detail || '') + '</span>';
      body.appendChild(row);
      body.scrollTop = body.scrollHeight;
      return row;
    },

    _updateRow(row, duration, detail) {
      if (!row) return;
      row.className = 'trace-row';
      const durCell = row.querySelector('.trace-duration');
      if (durCell) durCell.textContent = duration.toFixed(0) + 'ms';
      const detCell = row.querySelector('.trace-detail');
      if (detCell) {
        detCell.textContent = this._esc(detail || '').substring(0, 120);
        detCell.className = 'trace-detail highlight';
      }
    },

    _esc(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }
  };

  function clearTrace() {
    trace.clear();
  }

  document.addEventListener('click', function(e) {
    if (e.target && e.target.classList && e.target.classList.contains('fb-btn')) {
      e.preventDefault();
      e.stopPropagation();
      submitFeedback(e.target);
    }
  });

  init();
})();
