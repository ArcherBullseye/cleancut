/* Job progress + EDL review. */

const JOB_ID = document.getElementById("job-root").dataset.jobId;
const ACTIVE = new Set(["queued", "running"]);

let job = null;
let edl = null;
let loadedEdlFor = null;

/* ---------------------------------------------------------------- status */

function renderStatus() {
  const pill = document.getElementById("job-status");
  pill.textContent = job.status;
  pill.className = `pill ${job.status}`;

  document.getElementById("job-stage").textContent = job.stage || "";
  document.getElementById("cancel-job").hidden = !ACTIVE.has(job.status);
  document.getElementById("progress-box").hidden = !ACTIVE.has(job.status);
  document.getElementById("job-bar").style.width = `${job.progress || 0}%`;

  const elapsed = document.getElementById("job-elapsed");
  if (job.started_at) {
    const end = job.finished_at || Date.now() / 1000;
    elapsed.textContent = formatDuration(end - job.started_at);
  } else {
    elapsed.textContent = "waiting for a free worker";
  }

  const error = document.getElementById("job-error");
  error.hidden = !job.error;
  error.textContent = job.error || "";

  renderActions();
}

function renderActions() {
  const row = document.getElementById("job-actions");
  row.innerHTML = "";

  if (job.kind === "scan" && job.status === "done" && job.edl_path) {
    row.appendChild(el("button", {
      class: "primary", text: "Render cleaned video", onclick: startRender,
    }));
  }
  if (job.kind === "render" && job.status === "done" && job.output_exists) {
    row.appendChild(el("a", {
      class: "button primary", href: `/api/job/${job.id}/download`, text: "Open cleaned video",
    }));
    row.appendChild(el("a", {
      class: "button ghost", href: `/api/job/${job.id}/download`, download: "", text: "Download",
    }));
    row.appendChild(el("span", {
      class: "muted small", text: formatBytes(job.output_size),
    }));
  }
  if (job.parent_id) {
    row.appendChild(el("a", {
      class: "button ghost", href: `/job/${job.parent_id}`, text: "Back to the scan",
    }));
  }
  if (job.status === "failed" || job.status === "canceled") {
    row.appendChild(el("a", { class: "button ghost", href: "/", text: "Start over" }));
  }
}

async function startRender() {
  try {
    const res = await post(`/api/job/${JOB_ID}/render`);
    window.location.href = `/job/${res.job_id}`;
  } catch (e) {
    toast(e.message, true);
  }
}

/* ---------------------------------------------------------------- review */

function statTile(value, label) {
  return el("div", { class: "stat" }, [
    el("div", { class: "n", text: String(value) }),
    el("div", { class: "k", text: label }),
  ]);
}

function renderSummary(summary) {
  const container = document.getElementById("summary");
  container.innerHTML = "";
  container.appendChild(statTile(summary.accepted, "accepted"));
  container.appendChild(statTile(summary.rejected, "rejected"));
  container.appendChild(statTile(summary.cuts, "cuts"));
  container.appendChild(statTile(formatClock(summary.seconds_cut), "removed"));
  container.appendChild(statTile(summary.mutes, "mutes"));
  container.appendChild(statTile(formatClock(summary.seconds_muted), "muted"));

  // Rendering only re-encodes when something is actually cut. A mute-only edit
  // stream-copies the video and finishes in minutes; a single accepted cut
  // forces a full re-encode of the whole film, which is hours on this hardware.
  const note = document.getElementById("render-note");
  if (note) {
    note.textContent = summary.cuts > 0
      ? `${summary.cuts} cut${summary.cuts === 1 ? "" : "s"} accepted, so the render re-encodes the whole film -- expect hours. Rejecting every cut and keeping only mutes copies the video instead and finishes in minutes.`
      : "No cuts accepted, only mutes, so the render copies the video rather than re-encoding it. This will be quick.";
  }
}

function decisionCard(decision) {
  const midpoint = decision.start + Math.min(2, (decision.end - decision.start) / 2);
  const accepted = decision.accepted !== false;
  const headCategory = String(decision.category || "").split("+")[0];

  const thumb = el("img", {
    class: "thumb", loading: "lazy", alt: "",
    src: `/api/job/${JOB_ID}/thumb?t=${midpoint.toFixed(2)}`,
    title: "Click to preview this moment",
  });

  const card = el("div", {
    class: `decision${accepted ? "" : " rejected"}`,
    "data-index": decision.index,
    "data-category": headCategory,
    "data-accepted": accepted ? "1" : "0",
  });

  const actionSelect = el("select", {
    onchange: (e) => edit(decision.index, { action: e.target.value }),
  });
  for (const action of ACTIONS) {
    const option = el("option", { value: action, text: action });
    if (decision.action === action) option.selected = true;
    actionSelect.appendChild(option);
  }

  const toggle = el("button", {
    class: accepted ? "ghost" : "primary",
    text: accepted ? "Reject" : "Accept",
    onclick: () => edit(decision.index, { accepted: !accepted }),
  });

  const quote = [];
  if (decision.text_before) {
    quote.push(el("p", { class: "quote" }, [
      el("s", { text: decision.text_before }),
      decision.text_after ? document.createTextNode("  →  ") : null,
      decision.text_after ? el("span", { text: decision.text_after }) : null,
    ].filter(Boolean)));
  }

  const body = el("div", { class: "body" }, [
    el("div", { class: "head" }, [
      el("span", { class: "time", text: `${decision.start_label} – ${decision.end_label}` }),
      el("span", { class: `tag ${decision.action}`, text: decision.action }),
      el("span", { class: "tag", text: decision.category || "?" }),
      el("span", { class: "tag", text: decision.source || "?" }),
      el("span", { class: "muted small", text: `${decision.duration.toFixed(1)}s` }),
    ]),
    el("div", { class: "reason", text: decision.reason || "" }),
    ...quote,
    el("div", { class: "controls" }, [
      toggle,
      actionSelect,
      el("button", {
        class: "ghost", text: "Preview clip",
        onclick: (e) => showClip(card, decision, e.target),
      }),
      el("button", {
        class: "ghost", text: "Trim",
        onclick: () => trim(decision),
      }),
    ]),
  ]);

  thumb.addEventListener("click", () => showClip(card, decision, null));
  card.appendChild(thumb);
  card.appendChild(body);
  return card;
}

function showClip(card, decision, button) {
  if (card.querySelector("video")) return;
  if (button) { button.disabled = true; button.textContent = "Building..."; }
  const video = el("video", {
    controls: "", preload: "metadata", autoplay: "",
    src: `/api/job/${JOB_ID}/clip?start=${decision.start.toFixed(2)}&end=${decision.end.toFixed(2)}`,
  });
  video.addEventListener("loadeddata", () => {
    if (button) { button.disabled = false; button.textContent = "Hide clip"; }
  });
  video.addEventListener("error", () => {
    toast("Could not build a preview for that range.", true);
    video.remove();
    if (button) { button.disabled = false; button.textContent = "Preview clip"; }
  });
  card.querySelector(".body").appendChild(video);
  if (button) {
    button.onclick = () => {
      video.remove();
      button.textContent = "Preview clip";
      button.onclick = (e) => showClip(card, decision, e.target);
    };
  }
}

async function trim(decision) {
  const start = prompt("New start (MM:SS or H:MM:SS)", decision.start_label);
  if (start === null) return;
  const end = prompt("New end (MM:SS or H:MM:SS)", decision.end_label);
  if (end === null) return;
  await edit(decision.index, { start, end });
}

async function edit(index, changes) {
  try {
    const res = await post(`/api/job/${JOB_ID}/edl/decision`, { index, ...changes });
    renderSummary(res.summary);
    await loadEdl(true);
  } catch (e) {
    toast(e.message, true);
  }
}

function applyFilters() {
  const hideRejected = document.getElementById("hide-rejected").checked;
  const category = document.getElementById("filter-category").value;
  for (const card of document.querySelectorAll(".decision")) {
    const rejected = card.dataset.accepted === "0";
    const matches = !category || card.dataset.category === category;
    card.hidden = (hideRejected && rejected) || !matches;
  }
}

function renderDecisions() {
  const container = document.getElementById("decisions");
  container.innerHTML = "";
  if (!edl.decisions.length) {
    container.appendChild(el("p", { class: "muted", text: "The scan found nothing to edit." }));
    return;
  }
  for (const decision of edl.decisions) container.appendChild(decisionCard(decision));

  const filter = document.getElementById("filter-category");
  const chosen = filter.value;
  const seen = [...new Set(edl.decisions.map((d) => String(d.category || "").split("+")[0]))].sort();
  filter.innerHTML = '<option value="">all</option>';
  for (const category of seen) {
    filter.appendChild(el("option", { value: category, text: category }));
  }
  filter.value = chosen;
  applyFilters();
}

async function loadEdl(force) {
  if (!job.edl_path) return;
  if (loadedEdlFor === JOB_ID && !force) return;
  try {
    const data = await api(`/api/job/${JOB_ID}/edl`);
    edl = data.edl;
    loadedEdlFor = JOB_ID;
    document.getElementById("review-panel").hidden = false;
    renderSummary(data.summary);
    renderDecisions();
  } catch (e) {
    toast(e.message, true);
  }
}

/* ------------------------------------------------------------------ poll */

async function refreshLog() {
  if (!document.getElementById("log-box").open) return;
  const data = await api(`/api/job/${JOB_ID}/log`);
  const pre = document.getElementById("job-log");
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
  pre.textContent = data.log;
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

async function poll() {
  try {
    job = await api(`/api/job/${JOB_ID}`);
    renderStatus();
    await refreshLog();
    if (job.kind === "scan" && job.status === "done") await loadEdl(false);
  } catch (e) {
    toast(e.message, true);
  }
  if (job && ACTIVE.has(job.status)) setTimeout(poll, 3000);
}

document.getElementById("cancel-job").addEventListener("click", async () => {
  await post(`/api/job/${JOB_ID}/cancel`);
  poll();
});
document.getElementById("log-box").addEventListener("toggle", refreshLog);
document.getElementById("hide-rejected").addEventListener("change", applyFilters);
document.getElementById("filter-category").addEventListener("change", applyFilters);

for (const button of document.querySelectorAll("[data-bulk]")) {
  button.addEventListener("click", async () => {
    const accepted = button.dataset.bulk === "accept";
    const category = document.getElementById("filter-category").value || null;
    const label = category ? `all ${category} decisions` : "every decision";
    if (!confirm(`${accepted ? "Accept" : "Reject"} ${label}?`)) return;
    const res = await post(`/api/job/${JOB_ID}/edl/bulk`, { accepted, category });
    renderSummary(res.summary);
    await loadEdl(true);
    toast(`${res.changed} changed`);
  });
}

document.getElementById("add-cut-toggle").addEventListener("click", () => {
  const form = document.getElementById("add-cut");
  form.hidden = !form.hidden;
});

document.getElementById("add-cut").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  try {
    const res = await post(`/api/job/${JOB_ID}/edl/add`, Object.fromEntries(form));
    renderSummary(res.summary);
    await loadEdl(true);
    event.target.reset();
    event.target.hidden = true;
    toast("Cut added");
  } catch (e) {
    toast(e.message, true);
  }
});

poll();
