/* Job list, polled while anything is active. */
let visibleJobs = [];
let refreshTimer = null;
let requestVersion = 0;

function jobCard(job) {
  const parts = [job.kind === "batch" ? "Batch" : job.kind === "scan" ? "Scan" : "Render", job.preset].filter(Boolean);
  if (job.status === "running" && job.stage) parts.push(job.stage);
  if (job.status === "done" && job.finished_at && job.started_at) {
    parts.push(`took ${formatDuration(job.finished_at - job.started_at)}`);
  }
  return el("div", { class: "job-card" }, [
    el("div", { class: "job-main" }, [
      el("div", { class: "job-name" }, [el("a", { href: `/job/${job.id}`, text: job.title })]),
      el("div", { class: "job-sub", text: parts.join(" · ") }),
    ]),
    el("span", { class: `pill ${job.status}`, text: job.status }),
    el("button", {
      class: "ghost", text: "Delete",
      onclick: async () => {
        if (!confirm(`Delete this ${job.kind} job and its child jobs? Active work is canceled; logs, review edits and previews are removed. Source and cleaned videos are kept.`)) return;
        await api(`/api/job/${job.id}`, { method: "DELETE" });
        refresh();
      },
    }),
  ]);
}

async function refresh() {
  clearTimeout(refreshTimer);
  const version = ++requestVersion;
  document.getElementById("delete-filtered").disabled = true;
  const container = document.getElementById("jobs");
  try {
    const data = await api(`/api/jobs?state=${encodeURIComponent(document.getElementById("job-state").value)}`);
    if (version !== requestVersion) return;
    visibleJobs = data.jobs;
    document.getElementById("delete-filtered").disabled = !visibleJobs.some(j => !["queued", "running"].includes(j.status));
    container.innerHTML = "";
    if (!data.jobs.length) {
      container.appendChild(el("p", { class: "muted" }, [
        document.createTextNode("No jobs yet. "),
        el("a", { href: "/", text: "Pick a video" }),
        document.createTextNode(" to start one."),
      ]));
      refreshTimer = setTimeout(refresh, 4000);
      return;
    }
    for (const job of data.jobs) container.appendChild(jobCard(job));
    refreshTimer = setTimeout(refresh, 4000);
  } catch (e) {
    toast(e.message, true);
  }
}

document.getElementById("job-state").addEventListener("change", refresh);
document.getElementById("delete-filtered").addEventListener("click", async () => {
  const ids = visibleJobs.filter(j => !["queued", "running"].includes(j.status)).map(j => j.id);
  if (!ids.length || !confirm(`Delete ${ids.length} jobs and their logs, review edits and previews? Videos are kept. This cannot be undone.`)) return;
  document.getElementById("delete-filtered").disabled = true;
  try {
    const result = await post("/api/jobs/delete", {ids});
    toast(`Deleted ${result.deleted.length} jobs; ${result.skipped.length} active jobs skipped.`);
    await refresh();
  } catch (e) { toast(e.message, true); await refresh(); }
});
refresh();
