/* Settings form. */

let settings = null;

function updateOutputFolderState() {
  const sourceMode = document.getElementById("output-location").value === "source";
  document.getElementById("output-dir").disabled = sourceMode;
}

async function load() {
  settings = (await api("/api/settings")).settings;
  document.getElementById("preset").value = settings.preset;
  document.getElementById("ollama-host").value = settings.ollama_host;
  document.getElementById("llm-model").value = settings.llm_model;
  document.getElementById("vlm-model").value = settings.vlm_model;
  document.getElementById("local-ai-enabled").checked = !!settings.local_ai_enabled;
  document.getElementById("quality").value = settings.quality;
  document.getElementById("encoder").value = settings.encoder;
  document.getElementById("analysis-height").value = settings.analysis_height;
  document.getElementById("analysis-proxy").checked = !!settings.analysis_proxy;
  document.getElementById("nudity-model").value = settings.nudity_model;
  document.getElementById("render-validation").value = settings.render_validation;
  document.getElementById("output-dir").value = settings.output_dir;
  document.getElementById("output-location").value = settings.output_location;
  updateOutputFolderState();
  document.getElementById("subtitle-mode").value = settings.subtitle_mode;
  document.getElementById("auto-render").checked = !!settings.auto_render;
  renderCategoryChips(document.getElementById("categories"), settings.categories);
  renderActionGrid(document.getElementById("actions"), settings.actions);

  const health = await api("/api/health");
  for (const option of document.querySelectorAll("#encoder option[data-mac-only]")) {
    option.disabled = !health.native_macos;
  }
  document.getElementById("health").textContent =
    `version ${health.version} · ${health.platform}/${health.machine} · media roots: ${health.media_roots.join(", ") || "none mounted"} · output: ${health.output_dir}`;
}

document.getElementById("save").addEventListener("click", async () => {
  const body = {
    preset: document.getElementById("preset").value,
    ollama_host: document.getElementById("ollama-host").value.trim(),
    llm_model: document.getElementById("llm-model").value.trim(),
    vlm_model: document.getElementById("vlm-model").value.trim(),
    local_ai_enabled: document.getElementById("local-ai-enabled").checked,
    quality: Number(document.getElementById("quality").value),
    encoder: document.getElementById("encoder").value,
    analysis_height: Number(document.getElementById("analysis-height").value),
    analysis_proxy: document.getElementById("analysis-proxy").checked,
    nudity_model: document.getElementById("nudity-model").value,
    render_validation: document.getElementById("render-validation").value,
    output_dir: document.getElementById("output-dir").value.trim(),
    output_location: document.getElementById("output-location").value,
    subtitle_mode: document.getElementById("subtitle-mode").value,
    auto_render: document.getElementById("auto-render").checked,
    categories: readCategoryChips(document.getElementById("categories")),
    actions: readActionGrid(document.getElementById("actions")),
  };
  try {
    await post("/api/settings", body);
    document.getElementById("saved").textContent = "Saved";
    setTimeout(() => { document.getElementById("saved").textContent = ""; }, 2500);
  } catch (e) {
    toast(e.message, true);
  }
});

document.getElementById("output-location").addEventListener("change", updateOutputFolderState);

document.getElementById("test-ollama").addEventListener("click", async () => {
  const host = document.getElementById("ollama-host").value.trim();
  const status = document.getElementById("ollama-status");
  status.textContent = "Checking...";
  const res = await api(`/api/ollama?host=${encodeURIComponent(host)}`);
  status.textContent = res.ok
    ? `Connected. Models: ${res.models.join(", ") || "none pulled yet"}`
    : `Not reachable: ${res.reason}`;
});

load();
