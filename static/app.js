// 前端逻辑：收集表单 -> POST /api/plan -> 渲染多份方案 -> 支持反馈迭代
//
// 核心交互流程：
//   1. 用户填表单（城市/日期/天数/中转城市/个人需求），点「生成旅行方案」
//   2. 前端把输入 POST 到 /api/plan，后端返回 {plans, weather, tips, _meta}
//   3. render() 把 weather 和 plans 渲染成页面上的天气卡片 + 方案卡片
//   4. 若用户不满意，在「反馈修改」框输入要求，前端把「上一轮方案摘要 + 反馈」
//      追加到 requirements 里重新请求，实现多轮迭代

const form = document.getElementById("plan-form");
const submitBtn = document.getElementById("submit-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const feedbackBtn = document.getElementById("feedback-btn");
const chatInput = document.getElementById("chat-input");
const chatBtn = document.getElementById("chat-btn");
const chatAnswer = document.getElementById("chat-answer");

let lastPayload = null;
let lastPlans = null;
// 全局「有请求在飞」标志：防止用户在等待期间重复提交，
// 导致两个并发的 ReAct 循环同时烧 token、并且后返回的结果覆盖先返回的。
let pending = false;

// 所有会发起后端请求的入口，统一在这里禁用/恢复
const requestButtons = [submitBtn, feedbackBtn, chatBtn];

function setPending(on) {
  pending = on;
  requestButtons.forEach((b) => { if (b) b.disabled = on; });
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = "";
}

// 携带结构化 detail 的错误（后端返回 {code, field, ...} 时使用）。
// 定义在使用它的 doPlan 之前 —— class 声明不像 function 那样提升，
// 虽然运行时（脚本整体求值后才触发 submit）不会出错，但按顺序写更清晰。
class PlanError extends Error {
  constructor(detail, status) {
    const msg = typeof detail === "string"
      ? detail
      : (detail && detail.message) || `请求失败 (HTTP ${status})`;
    super(msg);
    this.detail = detail;
    this.status = status;
  }
}

async function doPlan(payload) {
  // 统一的「发起请求并渲染结果」入口，表单首次提交和反馈迭代都复用这个函数。
  if (pending) return;   // 已有请求在跑，忽略重复提交
  clearError();
  clearCityPicker();
  resultEl.hidden = true;
  loadingEl.hidden = false;
  setPending(true);

  try {
    const resp = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new PlanError(data.detail, resp.status);
    }
    // 记录本次请求和结果，供后续「反馈修改」时基于上一轮方案迭代
    lastPayload = payload;
    lastPlans = data.plans || [];
    render(data);
    resultEl.hidden = false;
    document.getElementById("feedback-section").hidden = false;
    document.getElementById("feedback").value = "";
  } catch (err) {
    // 省份输入：不是错误，而是「需要用户补一个选择」
    if (err instanceof PlanError && err.status === 409 && err.detail &&
        err.detail.code === "province_needs_city") {
      showCityPicker(err.detail, payload);
    } else {
      showError(err.message || "出错了，请稍后重试");
    }
  } finally {
    loadingEl.hidden = true;
    setPending(false);
  }
}

// ---- 省份 → 城市选择 ----
// 用户填了「江苏」这类省份时，后端返回 409 + 候选城市列表。
// 我们在这里展示可点的按钮，用户点一下就把该字段替换成具体城市并自动重新提交，
// 不需要重填整个表单。
function showCityPicker(detail, payload) {
  const el = document.getElementById("city-picker");
  const label = { departure_city: "所在城市", arrival_city: "目的地城市",
                  transfer_city: "中转城市" }[detail.field] || "城市";
  const buttons = detail.cities.map(
    (c) => `<button type="button" class="city-chip" data-city="${esc(c)}">${esc(c)}</button>`
  ).join("");
  el.innerHTML = `
    <div class="picker-title">「${esc(detail.province)}」是省份，请选择具体城市作为${esc(label)}</div>
    <div class="city-chips">${buttons}</div>`;
  el.hidden = false;
  el.querySelectorAll(".city-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const chosen = btn.getAttribute("data-city");
      const next = { ...payload, [detail.field]: chosen };
      // 把选择同步回表单，让用户看到自己选了什么
      const input = document.getElementById(detail.field);
      if (input) input.value = chosen;
      clearCityPicker();
      doPlan(next);
    });
  });
}

function clearCityPicker() {
  const el = document.getElementById("city-picker");
  if (!el) return;
  el.hidden = true;
  el.innerHTML = "";
}

form.addEventListener("submit", (e) => {
  // 首次提交：从表单收集输入，组装成后端期望的 JSON 结构
  e.preventDefault();
  const payload = {
    departure_city: document.getElementById("departure_city").value.trim(),
    arrival_city: document.getElementById("arrival_city").value.trim(),
    departure_date: document.getElementById("departure_date").value,
    days: parseInt(document.getElementById("days").value, 10),
    requirements: document.getElementById("requirements").value.trim(),
  };
  const transferCity = document.getElementById("transfer_city").value.trim();
  if (transferCity) payload.transfer_city = transferCity;
  doPlan(payload);
});

feedbackBtn.addEventListener("click", () => {
  // 反馈迭代：把用户反馈 + 上一轮方案摘要追加到需求里，重新生成。
  // 之所以带上「上一轮方案摘要」，是让 DeepSeek 知道要基于哪套方案来改，
  // 而不是从零重新生成一套完全不同的方案。
  const feedback = document.getElementById("feedback").value.trim();
  if (!feedback || !lastPayload) return;
  const planSummary = (lastPlans || [])
    .map((p) => `${p.name}（总价 ${p.total_cost} 元）：${p.reason}`)
    .join("；");
  const baseReq = lastPayload.requirements || "无特殊要求";
  const newRequirements =
    `${baseReq}\n\n【上一轮方案】${planSummary}\n\n【用户反馈】${feedback}\n` +
    `请基于上一轮方案，严格按照上述反馈修改后重新给出方案。`;
  doPlan({ ...lastPayload, requirements: newRequirements });
});

function render(data) {
  renderWarnings(data._meta && data._meta.warnings);
  renderWeather(data.weather || []);
  renderPlans(data.plans || []);
  renderTips(data.tips || "");
  renderTrace(data._meta && data._meta.trace);
}

// Agent 执行轨迹面板。
// 这是「观察 agent 内部过程」的窗口 —— 默认折叠，因为它是给开发者看的，
// 普通用户不关心；但对学习 agent 来说，这里是信息量最大的地方。
function renderTrace(trace) {
  const el = document.getElementById("trace-section");
  if (!el) return;
  if (!trace || !trace.summary) { el.innerHTML = ""; el.hidden = true; return; }

  const s = trace.summary;
  const growth = (s.prompt_tokens_growth || []).map((v) => v.toLocaleString()).join(" → ");

  // 每个工具一行：调用次数 / 失败次数 / 总耗时 / 平均返回大小
  const toolRows = Object.entries(s.by_tool || {}).map(([name, t]) => `
    <tr>
      <td>${esc(name)}</td>
      <td>${t.calls}</td>
      <td class="${t.errors ? "trace-err" : ""}">${t.errors}</td>
      <td>${t.total_ms} ms</td>
    </tr>`).join("");

  el.innerHTML = `
    <details class="trace-box">
      <summary>🔍 Agent 执行轨迹（开发者视图）</summary>
      <div class="trace-summary">
        <div><b>${s.llm_rounds}</b> 轮 LLM 调用 · <b>${s.tool_calls}</b> 次工具调用
             · 失败 <b>${s.tool_errors}</b> 次 · 空响应重试 <b>${s.empty_retries}</b> 次</div>
        <div>累计 <b>${s.prompt_tokens_total.toLocaleString()}</b> 输入 tokens
             + <b>${s.completion_tokens_total.toLocaleString()}</b> 输出 tokens
             · 总耗时 <b>${(s.total_ms / 1000).toFixed(1)}s</b></div>
        <div>最终上下文 <b>${s.final_message_count}</b> 条消息</div>
        ${growth ? `<div class="trace-growth">每轮输入 tokens：${growth}</div>` : ""}
      </div>
      ${toolRows ? `
      <table class="trace-table">
        <thead><tr><th>工具</th><th>调用</th><th>失败</th><th>耗时</th></tr></thead>
        <tbody>${toolRows}</tbody>
      </table>` : ""}
      <div class="trace-raw">
        ${(trace.tool_calls || []).map((t) => `
          <div class="trace-line">
            <span class="trace-step">#${t.step}</span>
            <span class="trace-name">${esc(t.name)}</span>
            <span class="trace-args">${esc(JSON.stringify(t.args))}</span>
            <span class="trace-meta">${t.elapsed_ms}ms · ${t.observation_chars}字符</span>
            ${t.is_error ? `<span class="trace-err">ERROR</span>` : ""}
          </div>`).join("")}
      </div>
    </details>`;
  el.hidden = false;
}

// 展示后端返回的「已知问题」列表（_meta.warnings）。
// 这些不是错误——方案仍然可用——但用户有权知道：天气为什么缺了几天、
// 门票价是不是参考价、酒店价是不是起价。静默隐藏比报错更糟。
function renderWarnings(list) {
  const el = document.getElementById("warnings-section");
  if (!el) return;
  if (!list || !list.length) { el.innerHTML = ""; el.hidden = true; return; }
  const items = list.map((w) => `<li>${esc(w)}</li>`).join("");
  el.innerHTML = `<div class="warn-title">⚠️ 请注意</div><ul class="warn-list">${items}</ul>`;
  el.hidden = false;
}

function renderWeather(list) {
  const el = document.getElementById("weather-section");
  // 以前这里直接 innerHTML="" 把整块抹掉，用户分不清是「没数据」还是「页面坏了」。
  // 现在明确说明原因（最常见的成因是出发日超出预报范围）。
  if (!list.length) {
    el.innerHTML = `<h2>🌤 天气</h2><div class="tips">暂无可用天气数据（出发日期可能超出免费天气预报范围）。</div>`;
    return;
  }
  const items = list.map((w) => `
    <div class="weather-item">
      <div class="cond">${condIcon(w.day_condition)} ${esc(w.day_condition)}</div>
      <div class="temp">${esc(w.min_temp)} ~ ${esc(w.max_temp)}°C</div>
      <div class="rain">降水 ${esc(w.rain_chance)}%</div>
      <div style="color:#94a3b8;font-size:12px;margin-top:4px">${esc(w.date)}</div>
    </div>`).join("");
  el.innerHTML = `<h2>🌤 天气</h2><div class="weather-grid">${items}</div>`;
}

function condIcon(cond) {
  const c = String(cond || "");
  if (c.includes("雨") || c.includes("雪")) return "🌧";
  if (c.includes("阴")) return "☁️";
  if (c.includes("云")) return "⛅";
  if (c.includes("晴")) return "☀️";
  return "🌤";
}

function renderPlans(plans) {
  // 渲染后端返回的 2-3 份方案：每份方案一个卡片。
  // 每张卡片把该方案的 车次/酒店/门票/路线/总价/理由 一次性展示，
  // 让用户能直接「挑一份」，而不是自己去组合零散信息。
  const el = document.getElementById("plans-section");
  if (!plans.length) { el.innerHTML = ""; return; }
  const cards = plans.map((p, i) => {
    // 用模板字符串拼接 HTML，数据通过 esc() 转义防止 XSS
    const out = trainLine(p.outbound, "去程");
    const ret = trainLine(p.return, "回程");
    const hotel = p.hotel || {};
    const hotelLine = hotel.name
      ? `${esc(hotel.name)} · ¥${esc(hotel.price_per_night)}/晚 × ${esc(hotel.nights)}晚 = ¥${esc(hotel.total)}`
      : "—";
    const tickets = (p.tickets || []).map((t) => {
      const price = t.price != null ? `¥${esc(t.price)}` : "免费";
      return `<span class="spot">${esc(t.spot)} ${price}</span>`;
    }).join("");
    const itinerary = (p.itinerary || []).map((d) => `
      <div class="plan-day">
        <strong>第${esc(d.day)}天 ${esc(d.theme || "")}</strong>
        <div class="plan-day-plan">${esc(d.plan)}</div>
      </div>`).join("");
    return `
      <div class="card plan-card">
        <div class="plan-head">
          <h2 class="plan-name">方案${["一","二","三","四","五"][i] || i+1}：${esc(p.name)}</h2>
          <div class="plan-total">¥${esc(p.total_cost)}</div>
        </div>
        <div class="plan-section">${out}</div>
        <div class="plan-section">${ret}</div>
        <div class="plan-section"><span class="plan-label">🏨 酒店</span> ${hotelLine}</div>
        ${tickets ? `<div class="plan-section"><span class="plan-label">🎫 门票</span> ${tickets}</div>` : ""}
        <div class="plan-section">
          <span class="plan-label">🗺 路线</span>
          ${itinerary}
        </div>
        <div class="plan-reason">💡 ${esc(p.reason)}</div>
      </div>`;
  }).join("");
  el.innerHTML = `<h2 style="font-size:20px;margin:0 0 16px">🧭 为你生成的出行方案</h2>${cards}`;
}

function trainLine(t, label) {
  if (!t) return `<span class="plan-label">🚄 ${label}</span> 未提供`;
  const mode = t.mode ? `[${esc(t.mode)}]` : "";
  return `<span class="plan-label">🚄 ${label} ${mode}</span> ${esc(t.detail)} · ¥${esc(t.price)}`;
}

function renderTips(tips) {
  const el = document.getElementById("tips-section");
  if (!tips) { el.innerHTML = ""; return; }
  el.innerHTML = `<h2>📌 出行建议</h2><div class="tips">${esc(tips)}</div>`;
}

// ---- 自由问答：让模型自主决定查什么工具来回答 ----
async function ask() {
  // 自由问答：把用户一句话发给 /api/chat，
  // 后端走 ReAct 循环（模型自主决定查什么工具）后返回自然语言回答。
  const q = chatInput.value.trim();
  if (!q || pending) return;
  setPending(true);
  chatAnswer.hidden = false;
  chatAnswer.textContent = "正在查询…";
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "请求失败");
    chatAnswer.textContent = data.answer || "（无回答）";
  } catch (err) {
    chatAnswer.textContent = "出错：" + (err.message || "请稍后重试");
  } finally {
    setPending(false);
  }
}

chatBtn.addEventListener("click", ask);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});
