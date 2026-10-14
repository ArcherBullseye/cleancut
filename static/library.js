/* Library browser + scan form. */

let currentSettings = null;
const picked = new Map();
let currentListing = null;
let browseRequest = 0;
let folderBusy = false;
let folderRequest = 0;
const MAX_BATCH_FILES = 500;

async function loadSettings() {
  const data = await api("/api/settings");
  currentSettings = data.settings;
  document.querySelector(`input[name=preset][value="${currentSettings.preset}"]`).checked = true;
  renderCategoryChips(document.getElementById("categories"), currentSettings.categories);
  renderActionGrid(document.getElementById("actions"), currentSettings.actions);
  document.getElementById("prefer-language").value = currentSettings.prefer_language;
  document.getElementById("auto-render").checked = !!currentSettings.auto_render;
  document.getElementById("use-local-ai").checked = !!currentSettings.local_ai_enabled;
  document.getElementById("use-visual").checked = !!currentSettings.use_visual;
  document.getElementById("allow-solo-visual").checked = !!currentSettings.allow_solo_visual;
  document.getElementById("word-end-padding-ms").value = currentSettings.word_end_padding_ms;
}

function renderListing(data) {
  currentListing = data;
  document.getElementById("add-folder").disabled = folderBusy || !data.path || !!data.error || !!data.search;
  document.getElementById("select-visible").disabled = !data.files.length;
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
    const checkbox = el("input", {
      type: "checkbox", "aria-label": `Select ${file.name}`,
    });
    checkbox.checked = picked.has(file.path);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) addFiles([file]);
      else { picked.delete(file.path); updateSelection(); }
      checkbox.checked = picked.has(file.path);
    });
    listing.appendChild(el("label", {
      class: "item",
    }, [
      checkbox,
      el("span", { class: "name", text: file.name }),
      el("span", { class: "size", text: formatBytes(file.size) }),
    ]));
  }
  if (!data.dirs.length && !data.files.length) {
    listing.appendChild(el("p", { class: "muted", text: "Nothing here." }));
  }
}

async function browse(path) {
  const request = ++browseRequest;
  currentListing = null;
  document.getElementById("add-folder").disabled = true;
  document.getElementById("select-visible").disabled = true;
  const listing = document.getElementById("listing");
  listing.innerHTML = '<p class="muted">Loading...</p>';
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(path || "")}`);
    if (request !== browseRequest) return;
    renderListing(data);
    if (path) {
      sessionStorage.setItem("cleancut.path", path);
    } else {
      sessionStorage.removeItem("cleancut.path");
    }
  } catch (e) {
    if (request !== browseRequest) return;
    listing.innerHTML = "";
    listing.appendChild(el("p", { class: "muted", text: e.message }));
  }
}

function addFiles(files) {
  const combined = new Map(picked);
  for (const file of files) combined.set(file.path, file);
  if (combined.size > MAX_BATCH_FILES) {
    toast(`Select at most ${MAX_BATCH_FILES} videos at once. Nothing was added.`, true);
    return;
  }
  for (const file of files) picked.set(file.path, file);
  updateSelection();
}

function updateSelection() {
  document.getElementById("scan-title").textContent = `${picked.size} video${picked.size === 1 ? "" : "s"}`;
  document.getElementById("scan-panel").hidden = !picked.size;
  document.getElementById("selection-count").textContent = `${picked.size} selected`;
  document.getElementById("start-scan").textContent = `Queue ${picked.size} scan${picked.size === 1 ? "" : "s"}`;
  const list = document.getElementById("selected-files");
  list.replaceChildren();
  for (const file of picked.values()) {
    list.appendChild(el("div", { class: "item" }, [
      el("span", { class: "name", text: file.path, title: file.path }),
      el("button", { class: "ghost", text: "Remove", "aria-label": `Remove ${file.name}`,
        onclick: () => { picked.delete(file.path); updateSelection(); } }),
    ]));
  }
  if (currentListing) renderListing(currentListing);
}

async function addFolder() {
  if (folderBusy || !currentListing?.path || currentListing.search) return;
  const request = ++folderRequest;
  folderBusy = true;
  document.getElementById("start-scan").disabled = true;
  const button = document.getElementById("add-folder");
  button.disabled = true;
  button.textContent = "Reading folder…";
  try {
    const data = await post("/api/folder-videos", {
      path: currentListing.path,
      recursive: document.getElementById("include-subfolders").checked,
    });
    if (request !== folderRequest) return;
    if (!data.files.length) toast("No original videos found in this folder.");
    else addFiles(data.files);
  } catch (e) {
    toast(e.message, true);
  } finally {
    folderBusy = false;
    document.getElementById("start-scan").disabled = false;
    button.textContent = "Add this folder";
    button.disabled = !currentListing?.path || !!currentListing.error || !!currentListing.search;
  }
}

async function startScan() {
  if (!picked.size || folderBusy) return;
  const button = document.getElementById("start-scan");
  button.disabled = true;
  try {
    const body = {
      paths: [...picked.keys()],
      preset: document.querySelector("input[name=preset]:checked").value,
      categories: readCategoryChips(document.getElementById("categories")),
      actions: readActionGrid(document.getElementById("actions")),
      prefer_language: document.getElementById("prefer-language").value.trim() || "eng",
      use_visual: document.getElementById("use-visual").checked,
      use_local_ai: document.getElementById("use-local-ai").checked,
      allow_solo_visual: document.getElementById("allow-solo-visual").checked,
      auto_render: document.getElementById("auto-render").checked,
      word_end_padding_ms: Number(document.getElementById("word-end-padding-ms").value),
    };
    const res = await post("/api/scan", body);
    window.location.href = res.batch_id ? `/job/${res.batch_id}` : res.jobs.length === 1 ? `/job/${res.job_id}` : "/jobs";
  } catch (e) {
    toast(e.message, true);
    button.disabled = false;
  }
}

let searchTimer = null;
function onSearch(event) {
  clearTimeout(searchTimer);
  const request = ++browseRequest;
  currentListing = null;
  document.getElementById("add-folder").disabled = true;
  document.getElementById("select-visible").disabled = true;
  const term = event.target.value;
  searchTimer = setTimeout(async () => {
    if (term.trim().length < 2) {
      browse(sessionStorage.getItem("cleancut.path") || "");
      return;
    }
    try {
      const data = await api(`/api/search?q=${encodeURIComponent(term)}`);
      if (request !== browseRequest) return;
      renderListing({ path: `search: ${term}`, search: true, parent: null, dirs: [], files: data.results });
    } catch (e) { if (request === browseRequest) toast(e.message, true); }
  }, 320);
}

document.getElementById("search").addEventListener("input", onSearch);
document.getElementById("locations").addEventListener("click", () => {
  clearTimeout(searchTimer);
  document.getElementById("search").value = "";
  browse("");
});
document.getElementById("start-scan").addEventListener("click", startScan);
document.getElementById("cancel-pick").addEventListener("click", () => {
  folderRequest++;
  picked.clear();
  updateSelection();
});
document.getElementById("select-visible").addEventListener("click", () => addFiles(currentListing?.files || []));
document.getElementById("add-folder").addEventListener("click", addFolder);

loadSettings().then(() => browse(sessionStorage.getItem("cleancut.path") || ""));
