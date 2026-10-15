/* Credentials are never placed in a URL, localStorage, or job options. */
const serviceNames = ["radarr", "sonarr"];
const fieldNames = ["url", "api_key", "server_root", "local_root", "tag"];
const fieldLabels = {url: "Local URL (literal LAN IP)", api_key: "API key", server_root: "Server media root",
  local_root: "Mounted Mac root", tag: "Required opt-in tag"};

function showImports(events) {
  const target = document.getElementById("imports");
  target.replaceChildren();
  if (!events.length) target.appendChild(el("p", {text: "No imports yet. Integration is disabled until you enable it."}));
  for (const event of events) {
    const box = el("div", {class: "panel"});
    box.appendChild(el("p", {text: `${event.service}: ${event.local_path}`}));
    box.appendChild(el("p", {text: `Publication: ${event.state} · Job: ${event.job_status}`}));
    if (event.job_id && event.job_status !== "deleted") {
      box.appendChild(el("a", {href: `/job/${event.job_id}`, text: "Review scan"}));
    }
    if (event.render_id) box.appendChild(el("a", {href: `/job/${event.render_id}`, text: " · Render details"}));
    if (event.job_error || event.error) box.appendChild(el("p", {text: event.error || event.job_error}));
    if (["queued", "scanning"].includes(event.state) && ["failed", "canceled"].includes(event.job_status)) {
      box.appendChild(el("button", {class: "ghost", text: "Retry scan", onclick: async () => {
        try { await post(`/api/integrations/${event.id}/retry`); await refresh(false); }
        catch (error) { toast(error.message, true); }
      }}));
    }
    if (["preparing", "publishing", "published"].includes(event.state)) {
      box.appendChild(el("p", {text: `Recovery backup: ${event.backup}`}));
      box.appendChild(el("p", {text: `Upload: ${event.partial}`}));
      box.appendChild(el("button", {class: "ghost", text: "Recover / confirm rescan", onclick: async (click) => {
        if (!confirm("Check publication and retry the library rescan? If the original is verified intact, this attempt's temporary copies will be removed so you can render again.")) return;
        click.target.disabled = true;
        try { await post(`/api/integrations/${event.id}/recover`); await refresh(false); }
        catch (error) { toast(error.message, true); }
        finally { click.target.disabled = false; }
      }}));
    }
    target.appendChild(box);
  }
}

async function refresh(loadSettings) {
  const data = await api("/api/integrations");
  if (loadSettings) {
    document.getElementById("integration-enabled").checked = data.settings.enabled;
    document.getElementById("integration-auto").checked = data.settings.auto_render;
    const target = document.getElementById("services");
    target.replaceChildren();
    for (const name of serviceNames) {
      const service = data.settings.services[name];
      const box = el("section", {class: "panel"});
      box.appendChild(el("h2", {text: name === "radarr" ? "Radarr" : "Sonarr"}));
      for (const key of fieldNames) {
        const id = `${name}-${key}`;
        box.appendChild(el("label", {for: id, text: fieldLabels[key]}));
        box.appendChild(el("input", {id, type: key === "api_key" ? "password" : "text",
          value: service[key] || "", autocomplete: "off",
          placeholder: key === "api_key" && service.key_configured ? "Saved (leave blank to keep)" : ""}));
      }
      box.appendChild(el("button", {class: "ghost", text: "Test saved connection", onclick: async () => {
        try {
          const result = await post(`/api/integrations/${name}/test`);
          toast(`${name} ${result.version}; NAS ${result.mount_available ? "mounted" : "unavailable"}; roots: ${result.roots.join(", ")}`);
        } catch (error) { toast(error.message, true); }
      }}));
      target.appendChild(box);
    }
  }
  showImports(data.events);
}

document.getElementById("save-integrations").onclick = async () => {
  const services = {};
  for (const name of serviceNames) {
    services[name] = Object.fromEntries(fieldNames.map(key => [key, document.getElementById(`${name}-${key}`).value]));
  }
  const enabled = document.getElementById("integration-enabled").checked;
  if (enabled && !confirm("Enable tagged imports? Rendering these jobs replaces the library original after verification. Temporary backups are deleted after a successful rescan.")) return;
  try {
    await post("/api/integrations", {enabled, auto_render: document.getElementById("integration-auto").checked, services});
    await refresh(true);
    document.getElementById("integration-status").textContent = "Saved locally.";
  } catch (error) { toast(error.message, true); }
};
document.getElementById("show-webhook-token").onclick = async () => {
  try {
    const input = document.getElementById("webhook-token");
    if (input.type === "text") { input.type = "password"; input.value = ""; return; }
    input.value = (await post("/api/integrations/token")).token;
    input.type = "text";
  } catch (error) { toast(error.message, true); }
};
document.getElementById("rotate-webhook-token").onclick = async () => {
  if (!confirm("Invalidate the current webhook password? You must update both Arr connections.")) return;
  try {
    await post("/api/integrations", {rotate_token: true});
    document.getElementById("webhook-token").value = "";
    document.getElementById("webhook-token").type = "password";
    toast("Password rotated. Show it and update your Arr connections.");
  } catch (error) { toast(error.message, true); }
};
document.getElementById("test-path").onclick = async () => {
  try {
    const result = await post("/api/integrations/path-test", {
      service: document.getElementById("path-service").value, path: document.getElementById("server-file").value});
    document.getElementById("path-result").textContent = `Readable source and writable parent: ${result.path}`;
  } catch (error) { document.getElementById("path-result").textContent = error.message; }
};
document.getElementById("refresh-imports").onclick = () => refresh(false).catch(error => toast(error.message, true));
refresh(true).catch(error => {
  document.getElementById("integration-status").textContent = error.message;
});
