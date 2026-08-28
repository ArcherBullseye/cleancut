/* Library browser + scan form. */

let currentSettings = null;
let picked = null;

async function loadSettings() {
  const data = await api("/api/settings");
  currentSettings = data.settings;
  document.querySelector(`input[name=preset][value="${currentSettings.preset}"]`).checked = true;
  renderCategoryChips(document.getElementById("categories"), currentSettings.categories);
  renderActionGrid(document.getElementById("actions"), currentSettings.actions);
  document.getElementById("prefer-language").value = currentSettings.prefer_language;
  document.getElementById("auto-render").checked = !!currentSettings.auto_render;
}

function renderListing(data) {
  const listing = document.getElementById("listing");
  const crumbs = document.getElementById("crumbs");
  crumbs.textContent = data.path || "";
  listing.innerHTML = "";

  if (data.error) {
    listing.appendChild(el("p", { class: "muted", text: data.error }));
    return;
  }
  if (data.parent) {
    listing.appendChild(el("button", {
      class: "item", onclick: () => browse(data.parent),
    }, [el("span", { class: "glyph", text: "↑" }), el("span", { class: "name", text: ".." })]));
  }
  for (const dir of data.dirs) {
    listing.appendChild(el("button", {
      class: "item", onclick: () => browse(dir.path),
    }, [
      el("span", { class: "glyph", text: "▸" }),
      el("span", { class: "name", text: dir.name }),
    ]));
  }
  for (const file of data.files) {
    listing.appendChild(el("button", {
      class: "item", onclick: () => pick(file),
    }, [
      el("span", { class: "glyph", text: "■" }),
      el("span", { class: "name", text: file.name }),
      el("span", { class: "size", text: formatBytes(file.size) }),
    ]));
  }
  if (!data.dirs.length && !data.files.length) {
    listing.appendChild(el("p", { class: "muted", text: "Nothing here." }));
  }
}

async function browse(path) {
  const listing = document.getElementById("listing");
  listing.innerHTML = '<p class="muted">Loading...</p>';
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(path || "")}`);
    renderListing(data);
    if (path) sessionStorage.setItem("cleancut.path", path);
  } catch (e) {
    listing.innerHTML = "";
    listing.appendChild(el("p", { class: "muted", text: e.message }));
  }
}

function pick(file) {
  picked = file;
  document.getElementById("scan-title").textContent = file.name;
  document.getElementById("scan-panel").hidden = false;
  document.getElementById("scan-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function startScan() {
  if (!picked) return;
  const button = document.getElementById("start-scan");
  button.disabled = true;
  try {
    const body = {
      path: picked.path,
      preset: document.querySelector("input[name=preset]:checked").value,
      categories: readCategoryChips(document.getElementById("categories")),
      actions: readActionGrid(document.getElementById("actions")),
      prefer_language: document.getElementById("prefer-language").value.trim() || "eng",
      use_visual: document.getElementById("use-visual").checked,
      allow_solo_visual: document.getElementById("allow-solo-visual").checked,
      auto_render: document.getElementById("auto-render").checked,
    };
    const res = await post("/api/scan", body);
    window.location.href = `/job/${res.job_id}`;
  } catch (e) {
    toast(e.message, true);
    button.disabled = false;
  }
}

let searchTimer = null;
function onSearch(event) {
  clearTimeout(searchTimer);
  const term = event.target.value;
  searchTimer = setTimeout(async () => {
    if (term.trim().length < 2) {
      browse(sessionStorage.getItem("cleancut.path") || "");
      return;
    }
    const data = await api(`/api/search?q=${encodeURIComponent(term)}`);
    renderListing({ path: `search: ${term}`, parent: null, dirs: [], files: data.results });
  }, 320);
}

document.getElementById("search").addEventListener("input", onSearch);
document.getElementById("start-scan").addEventListener("click", startScan);
document.getElementById("cancel-pick").addEventListener("click", () => {
  picked = null;
  document.getElementById("scan-panel").hidden = true;
});

loadSettings().then(() => browse(sessionStorage.getItem("cleancut.path") || ""));
