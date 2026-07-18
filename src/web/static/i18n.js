/* doc-search i18n — lightweight UI string internationalization.

   Usage:
     <span data-i18n="key">fallback</span>     → auto-translated
     t('key')                                   → JS string

   Default locale: zh-CN. Add ?lang=en to URL for English.
*/

(function () {
  'use strict';

  const STRINGS = {
    'zh-CN': {
      // Tab headers
      'tab.sessions': '会话',
      'tab.database': '数据库',
      // Sidebar
      'sidebar.new_session': '+ 新建',
      'sidebar.config': '配置',
      'sidebar.index_path': '索引路径',
      'sidebar.raw_dir': 'Raw 目录',
      // Chat
      'chat.welcome': 'doc-search Web',
      'chat.welcome_subtitle': '输入问题搜索文档知识库。Agent 会自动搜索、阅读文档并回答。',
      'chat.hint': '示例查询：',
      'chat.placeholder': '输入您的问题...（可编辑、复制、重新提交）',
      'chat.send': '发送',
      'chat.abort': '中止',
      // DB Panel
      'db.total_files': '总文件',
      'db.success': '成功',
      'db.failed': '失败',
      'db.pending': '待处理',
      'db.all_status': '全部状态',
      'db.refresh': '刷新',
      'db.loading': '请先在下方配置 Raw 目录',
      'db.token_label': '7天 Token:',
      'db.no_files': '无匹配文件',
      'db.load_failed': '加载失败',
      // Upload
      'upload.drop_text': '拖拽文件到此处上传',
      'upload.hint': '支持 PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT, 图片, ZIP',
      'upload.select': '选择文件',
      'upload.preparing': '准备上传...',
      // Session
      'session.title': '会话',
      'session.delete_title': '删除会话',
      // Status
      'status.idle': '就绪',
      'status.processing': '处理中',
      'status.error': '错误',
      // Strategy
      'strategy.label': '策略: ',
      'strategy.simple': '简单查询',
      'strategy.medium': '中等查询',
      'strategy.complex': '复杂查询',
      'strategy.rounds': '次调用',
      // Sufficiency
      'sufficiency.label': '充足性检查 第',
      'sufficiency.coverage': '覆盖率 ',
      'sufficiency.missing': '个缺失方面',
      // Stats extras
      'stats.feedback': '反馈补充 ',
      'stats.rounds': '轮',
      'stats.draft_verified': '草稿验证通过',
      // Pipeline
      'pipeline.label': '管线',
      'pipeline.search': '搜索',
      'pipeline.read': '读取',
      'pipeline.grep': '正则',
      'pipeline.bash': 'Shell',
      'pipeline.rerank': '重排序',
      'pipeline.sufficiency': '充足性',
      'pipeline.draft': '草稿验证',
      'pipeline.coverage': '覆盖率',
      // Trace panel
      'trace.title': '检索过程',
      'trace.toggle': '展开/折叠',
      'trace.empty': '等待查询...',
      'trace.waiting': '检索中...',
      // Sources
      'sources.more': '还有 {n} 个',
      // Tool labels
      'tool.search': '搜索',
      'tool.read': '读取',
      'tool.grep': '正则搜索',
      'tool.bash': 'Shell',
      'tool.rerank': '重排序',
      'tool.default': '工具',
      // Search mode
      'search.advanced_toggle': '高级',
      // Skill labels
      'skill.none': '无技能',
      'skill.summarize': '摘要',
      'skill.compare': '对比',
      'skill.extract_table': '提取表格',
      'skill.detailed': '详细分析',
      'skill.timeline': '时间线',
      'skill.action_items': '行动项',
      'skill.review': '消保审查',
    },
    'en': {
      'tab.sessions': 'Sessions',
      'tab.database': 'Database',
      'sidebar.new_session': '+ New',
      'sidebar.config': 'Config',
      'sidebar.index_path': 'Index Path',
      'sidebar.raw_dir': 'Raw Dir',
      'chat.welcome': 'doc-search Web',
      'chat.welcome_subtitle': 'Enter a question to search your document knowledge base. The Agent will search, read, and answer automatically.',
      'chat.hint': 'Examples:',
      'chat.placeholder': 'Enter your question... (editable, copy, resubmit)',
      'chat.send': 'Send',
      'chat.abort': 'Abort',
      'db.total_files': 'Total',
      'db.success': 'Success',
      'db.failed': 'Failed',
      'db.pending': 'Pending',
      'db.all_status': 'All',
      'db.refresh': 'Refresh',
      'db.loading': 'Configure Raw directory below',
      'db.token_label': '7-day Tokens:',
      'db.no_files': 'No matching files',
      'db.load_failed': 'Load failed',
      'upload.drop_text': 'Drop files here to upload',
      'upload.hint': 'Supports PDF, DOCX, XLSX, PPTX, HTML, CSV, TXT, Images, ZIP',
      'upload.select': 'Select Files',
      'upload.preparing': 'Preparing...',
      'session.title': 'Sessions',
      'session.delete_title': 'Delete Session',
      'status.idle': 'Idle',
      'status.processing': 'Processing',
      'status.error': 'Error',
      // Strategy
      'strategy.label': 'Strategy: ',
      'strategy.simple': 'Simple',
      'strategy.medium': 'Medium',
      'strategy.complex': 'Complex',
      'strategy.rounds': 'calls',
      // Sufficiency
      'sufficiency.label': 'Sufficiency check #',
      'sufficiency.coverage': 'coverage ',
      'sufficiency.missing': 'missing aspects',
      // Stats extras
      'stats.feedback': 'Feedback ',
      'stats.rounds': ' rounds',
      'stats.draft_verified': 'Draft verified',
      // Pipeline
      'pipeline.label': 'Pipeline',
      'pipeline.search': 'Search',
      'pipeline.read': 'Read',
      'pipeline.grep': 'Grep',
      'pipeline.bash': 'Shell',
      'pipeline.rerank': 'Rerank',
      'pipeline.sufficiency': 'Sufficiency',
      'pipeline.draft': 'Draft verified',
      'pipeline.coverage': 'coverage',
      // Trace panel
      'trace.title': 'Trace',
      'trace.toggle': 'Expand/Collapse',
      'trace.empty': 'Waiting for query...',
      'trace.waiting': 'Searching...',
      // Sources
      'sources.more': '+{n} more',
      // Tool labels
      'tool.search': 'Search',
      'tool.read': 'Read',
      'tool.grep': 'Grep Search',
      'tool.bash': 'Shell',
      'tool.rerank': 'Rerank',
      'tool.default': 'Tool',
      // Search mode
      'search.advanced_toggle': 'Advanced',
      // Skill labels
      'skill.none': 'No skill',
      'skill.summarize': 'Summarize',
      'skill.compare': 'Compare',
      'skill.extract_table': 'Extract Table',
      'skill.detailed': 'Detailed',
      'skill.timeline': 'Timeline',
      'skill.action_items': 'Action Items',
      'skill.review': 'Compliance Review',
    },
  };

  // Detect locale
  const urlParams = new URLSearchParams(window.location.search);
  const langParam = urlParams.get('lang');
  const htmlLang = document.documentElement.lang || '';
  let locale = langParam || htmlLang || 'zh-CN';
  if (!STRINGS[locale]) locale = 'zh-CN';

  window.__i18n = {
    locale: locale,
    /** Get translated string by key */
    t: function (key) {
      const dict = STRINGS[locale] || STRINGS['zh-CN'];
      return dict[key] || key;
    },
  };

  // Apply translations to elements with data-i18n attribute
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      var key = el.getAttribute('data-i18n');
      var text = window.__i18n.t(key);
      // Only change if tag has no children (avoid overwriting inner content)
      if (el.children.length === 0 && text) {
        el.textContent = text;
      }
    });
    // Placeholder attributes
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      var key = el.getAttribute('data-i18n-placeholder');
      var text = window.__i18n.t(key);
      if (text) el.placeholder = text;
    });
  });
})();
