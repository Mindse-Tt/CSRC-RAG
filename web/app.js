const sampleQuestions = [
  "根据这段违规行为推荐处罚方式，并说明法条依据。",
  "上市公司董事利用内幕信息买入股票获利，历史上通常对应哪些案例？",
  "信息披露违规案件常见的法规依据和处罚方式是什么？",
  "近五年内幕交易相关案件的处罚是否更严？",
];

const STORAGE_KEY = "csrc-rag-sessions";

const queryInputEl = document.getElementById("queryInput");
const sendBtnEl = document.getElementById("sendBtn");
const sampleRowEl = document.getElementById("sampleRow");
const chatContainerEl = document.getElementById("chatContainer");
const sessionListEl = document.getElementById("sessionList");
const newChatBtnEl = document.getElementById("newChatBtn");
const clearSessionBtnEl = document.getElementById("clearSessionBtn");
const toggleSidebarBtnEl = document.getElementById("toggleSidebarBtn");
const sidebarEl = document.getElementById("sidebar");
const sidebarStatusEl = document.getElementById("sidebarStatus");
const retrievalBadgeEl = document.getElementById("retrievalBadge");
const intentBadgeEl = document.getElementById("intentBadge");
const replyBadgeEl = document.getElementById("replyBadge");
const chatWrapEl = document.getElementById("chatWrap");

let sessions = loadSessions();
let activeSessionId = sessions[0]?.id || null;

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function createSession(title = "新对话") {
  return {
    id: uid(),
    title,
    createdAt: Date.now(),
    messages: [],
  };
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [createSession()];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.length ? parsed : [createSession()];
  } catch (_error) {
    return [createSession()];
  }
}

function saveSessions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function getActiveSession() {
  let session = sessions.find((item) => item.id === activeSessionId);
  if (!session) {
    session = createSession();
    sessions.unshift(session);
    activeSessionId = session.id;
    saveSessions();
  }
  return session;
}

function summarizeTitle(text) {
  return text.length > 22 ? `${text.slice(0, 22)}...` : text;
}

function renderSamples() {
  sampleRowEl.innerHTML = "";
  sampleQuestions.forEach((question) => {
    const button = document.createElement("button");
    button.className = "sample-btn";
    button.textContent = question;
    button.addEventListener("click", () => {
      queryInputEl.value = question;
      autosizeTextarea();
      queryInputEl.focus();
    });
    sampleRowEl.appendChild(button);
  });
}

function renderSessions() {
  sessionListEl.innerHTML = "";
  sessions.forEach((session) => {
    const button = document.createElement("button");
    button.className = `session-item${session.id === activeSessionId ? " active" : ""}`;
    const firstUserMessage = session.messages.find((item) => item.role === "user");
    const preview = firstUserMessage ? firstUserMessage.content : "点击开始一段新问答";
    button.innerHTML = `
      <span class="session-title">${escapeHtml(session.title)}</span>
      <span class="session-subtitle">${escapeHtml(preview.slice(0, 28))}</span>
    `;
    button.addEventListener("click", () => {
      activeSessionId = session.id;
      renderApp();
    });
    sessionListEl.appendChild(button);
  });
}

function renderWelcome() {
  chatContainerEl.innerHTML = `
    <section class="welcome">
      <div class="welcome-card">
        <div class="welcome-mark">证</div>
        <h2>证监会处罚案例智能分析台</h2>
        <p>
          当前版本已经接入本地意图判断模型和本地回复模型。输入案情后，系统会先做意图识别，再执行案例检索，
          最后依据召回证据生成答案，并展示支撑案例、法条和处罚方式。
        </p>
      </div>
    </section>
  `;
}

function renderMessages() {
  const session = getActiveSession();
  if (!session.messages.length) {
    renderWelcome();
    return;
  }

  chatContainerEl.innerHTML = "";
  session.messages.forEach((message) => {
    const group = document.createElement("section");
    group.className = "message-group";
    const row = document.createElement("div");
    row.className = `message-row ${message.role}`;
    const card = document.createElement("div");
    card.className = "message-card";

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "用户" : "系统";
    card.appendChild(role);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const pre = document.createElement("pre");
    pre.textContent = message.content;
    if (message.loading) {
      pre.className = "loading-dot";
      pre.textContent = "";
    }
    bubble.appendChild(pre);
    card.appendChild(bubble);

    if (message.role === "assistant" && message.meta) {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      [
        `意图：${message.meta.intent}`,
        `路由：${message.meta.intentMethod} (${message.meta.intentConfidence})`,
        `回复：${message.meta.responseBackend}${message.meta.responseModel ? ` / ${message.meta.responseModel}` : ""}`,
        `检索：${message.meta.retrievalUnit || "-"} / top_k=${message.meta.topK || "-"}`,
      ].forEach((text) => {
        const chip = document.createElement("span");
        chip.className = "meta-chip";
        chip.textContent = text;
        meta.appendChild(chip);
      });
      card.appendChild(meta);

      const foldStack = document.createElement("div");
      foldStack.className = "fold-stack";

      const paramsSection = document.createElement("details");
      paramsSection.className = "fold-card";
      paramsSection.innerHTML = `
        <summary>本次调用参数</summary>
        <div class="fold-content">
          <div class="param-grid">
            <div class="param-item"><span class="param-label">意图</span><span class="param-value">${escapeHtml(message.meta.intent || "-")}</span></div>
            <div class="param-item"><span class="param-label">意图置信度</span><span class="param-value">${escapeHtml(String(message.meta.intentConfidence || "-"))}</span></div>
            <div class="param-item"><span class="param-label">路由方式</span><span class="param-value">${escapeHtml(message.meta.intentMethod || "-")}</span></div>
            <div class="param-item"><span class="param-label">检索单元</span><span class="param-value">${escapeHtml(message.meta.retrievalUnit || "-")}</span></div>
            <div class="param-item"><span class="param-label">top_k</span><span class="param-value">${escapeHtml(String(message.meta.topK || "-"))}</span></div>
            <div class="param-item"><span class="param-label">过滤条件</span><span class="param-value">${escapeHtml(message.meta.filters || "无")}</span></div>
            <div class="param-item"><span class="param-label">回复后端</span><span class="param-value">${escapeHtml(message.meta.responseBackend || "-")}</span></div>
            <div class="param-item"><span class="param-label">回复模型</span><span class="param-value">${escapeHtml(message.meta.responseModel || "-")}</span></div>
          </div>
          <div class="score-block">
            <div class="score-title">意图打分</div>
            <pre>${escapeHtml(JSON.stringify(message.meta.intentScores || {}, null, 2))}</pre>
          </div>
        </div>
      `;
      foldStack.appendChild(paramsSection);

      const evidenceSection = document.createElement("details");
      evidenceSection.className = "fold-card";
      evidenceSection.innerHTML = `<summary>检索证据（${(message.meta.events || []).length} 条）</summary>`;

      const evidencePanel = document.createElement("div");
      evidencePanel.className = "evidence-panel fold-content";
      (message.meta.events || []).forEach((event, index) => {
        const article = document.createElement("details");
        article.className = "case-card";
        article.innerHTML = `
          <summary class="case-summary">
            <span class="case-rank">#${index + 1}</span>
            <span class="case-title">${escapeHtml(event.title || event.event_id)}</span>
            <span class="case-inline-meta">Score ${escapeHtml(String(event.score))} / ${escapeHtml(event.declare_date || "-")} / ${escapeHtml(event.promulgator || "-")}</span>
          </summary>
          <div class="case-detail-body">
            <div class="case-meta">
              <span>EventID: ${escapeHtml(event.event_id)}</span>
              <span>Score: ${escapeHtml(String(event.score))}</span>
              <span>${escapeHtml(event.declare_date || "-")}</span>
              <span>${escapeHtml(event.promulgator || "-")}</span>
            </div>
            <div class="case-grid">
              <div class="case-block">
                <h4>处罚方式</h4>
                <ul>${renderList(event.punishment_types || [])}</ul>
              </div>
              <div class="case-block">
                <h4>法规依据</h4>
                <ul>${renderList(event.laws || [])}</ul>
              </div>
              <div class="case-block">
                <h4>证据片段</h4>
                <ul>${renderList(event.snippets || [])}</ul>
              </div>
            </div>
          </div>
        `;
        evidencePanel.appendChild(article);
      });
      evidenceSection.appendChild(evidencePanel);
      foldStack.appendChild(evidenceSection);
      card.appendChild(foldStack);
    }

    row.appendChild(card);
    group.appendChild(row);
    chatContainerEl.appendChild(group);
  });

  chatWrapEl.scrollTop = chatWrapEl.scrollHeight;
}

function renderList(items) {
  const safeItems = items.filter(Boolean);
  if (!safeItems.length) return "<li>-</li>";
  return safeItems.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("");
}

function renderApp() {
  renderSessions();
  renderMessages();
  saveSessions();
}

function escapeHtml(input) {
  const element = document.createElement("div");
  element.textContent = input || "";
  return element.innerHTML;
}

function autosizeTextarea() {
  queryInputEl.style.height = "auto";
  queryInputEl.style.height = `${Math.min(queryInputEl.scrollHeight, 220)}px`;
}

function updateBadges(payload) {
  retrievalBadgeEl.textContent = payload.query_plan?.retrieval_unit
    ? `检索:${payload.query_plan.retrieval_unit}`
    : "Hybrid";
  intentBadgeEl.textContent = `意图:${payload.intent}`;
  replyBadgeEl.textContent = `回复:${payload.response_backend}`;
  sidebarStatusEl.textContent = payload.response_model || payload.response_backend || "已连接";
}

async function sendQuery() {
  const query = queryInputEl.value.trim();
  if (!query) return;

  const session = getActiveSession();
  if (!session.messages.length) {
    session.title = summarizeTitle(query);
  }

  session.messages.push({ role: "user", content: query });
  session.messages.push({ role: "assistant", content: "", loading: true });
  queryInputEl.value = "";
  autosizeTextarea();
  renderApp();

  const history = session.messages
    .filter((item) => !item.loading)
    .map((item) => ({ role: item.role, content: item.content }));

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "查询失败");
    }

    const assistantMessage = session.messages[session.messages.length - 1];
    assistantMessage.loading = false;
    assistantMessage.content = data.answer || "模型未返回文本。";
    assistantMessage.meta = {
      intent: data.intent,
      intentConfidence: data.intent_confidence,
      intentMethod: data.intent_method,
      intentScores: data.intent_scores || {},
      responseBackend: data.response_backend,
      responseModel: data.response_model,
      retrievalUnit: data.query_plan?.retrieval_unit || "-",
      topK: data.query_plan?.top_k || "-",
      filters:
        Object.keys(data.query_plan?.metadata_filters || {}).length > 0
          ? JSON.stringify(data.query_plan.metadata_filters, null, 0)
          : "无",
      events: data.events || [],
    };
    updateBadges(data);
    renderApp();
  } catch (error) {
    const assistantMessage = session.messages[session.messages.length - 1];
    assistantMessage.loading = false;
    assistantMessage.content = `查询失败：${error.message}`;
    assistantMessage.meta = {
      intent: "-",
      intentConfidence: "-",
      intentMethod: "error",
      responseBackend: "none",
      responseModel: null,
      filters: "无",
      events: [],
    };
    renderApp();
  }
}

function createNewChat() {
  const session = createSession();
  sessions.unshift(session);
  activeSessionId = session.id;
  renderApp();
}

function clearCurrentSession() {
  const session = getActiveSession();
  session.messages = [];
  session.title = "新对话";
  renderApp();
}

sendBtnEl.addEventListener("click", sendQuery);
newChatBtnEl.addEventListener("click", createNewChat);
clearSessionBtnEl.addEventListener("click", clearCurrentSession);
toggleSidebarBtnEl.addEventListener("click", () => sidebarEl.classList.toggle("hidden"));

queryInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendQuery();
  }
});

queryInputEl.addEventListener("input", autosizeTextarea);

renderSamples();
renderApp();
autosizeTextarea();
