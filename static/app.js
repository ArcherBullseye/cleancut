/* Shared helpers. */

const CATEGORIES = ["profanity", "drugs", "sex", "violence", "nudity"];
const ACTIONS = ["mute", "cut", "keep"];

async function api(url, options) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data = {};
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function post(url, body) {
  return api(url, { method: "POST", body: JSON.stringify(body || {}) });
}

let toastTimer = null;
function toast(message, bad) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("bad", !!bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 6000 : 2800);
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
  }
  for (const child of children || []) if (child) node.appendChild(child);
  return node;
}

function formatBytes(bytes) {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function formatDuration(seconds) {
  if (!seconds || seconds < 0) return "";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function formatClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = String(m).padStart(h ? 2 : 1, "0");
  return h ? `${h}:${mm}:${String(sec).padStart(2, "0")}` : `${mm}:${String(sec).padStart(2, "0")}`;
}

/* Category chips and per-category action pickers, shared by the scan form and
   the settings page. */

function renderCategoryChips(container, selected) {
  container.innerHTML = "";
  for (const cat of CATEGORIES) {
    const input = el("input", { type: "checkbox", value: cat });
    input.checked = selected.includes(cat);
    container.appendChild(el("label", { class: "chip" }, [input, el("span", { text: cat })]));
  }
}

function readCategoryChips(container) {
  return [...container.querySelectorAll("input:checked")].map((i) => i.value);
}

function renderActionGrid(container, actions) {
  container.innerHTML = "";
  for (const cat of CATEGORIES) {
    const select = el("select", { "data-category": cat });
    for (const action of ACTIONS) {
      const option = el("option", { value: action, text: action });
      if (actions[cat] === action) option.selected = true;
      select.appendChild(option);
    }
    container.appendChild(el("div", { class: "action-cell" }, [el("span", { text: cat }), select]));
  }
}

function readActionGrid(container) {
  const out = {};
  for (const select of container.querySelectorAll("select[data-category]")) {
    out[select.dataset.category] = select.value;
  }
  return out;
}
