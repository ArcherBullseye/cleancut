/* Job list, polled while anything is active. */

function jobCard(job) {
  const parts = [job.kind === "scan" ? "Scan" : "Render", job.preset].filter(Boolean);
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
        if (!confirm(`Delete this ${job.kind} job? Its EDL and previews go with it.`)) return;
        await api(`/api/job/${job.id}`, { method: "DELETE" });
        refresh();
      },
    }),
  ]);
}

async function refresh() {
  const container = document.getElementById("jobs");
  try {
    const data = await api("/api/jobs");
    container.innerHTML = "";
    if (!data.jobs.length) {
      container.appendChild(el("p", { class: "muted" }, [
        document.createTextNode("No jobs yet. "),
        el("a", { href: "/", text: "Pick a video" }),
        document.createTextNode(" to start one."),
      ]));
      return;
    }
    for (const job of data.jobs) container.appendChild(jobCard(job));
    if (data.jobs.some((j) => j.status === "running" || j.status === "queued")) {
      setTimeout(refresh, 4000);
    }
  } catch (e) {
    toast(e.message, true);
  }
}

refresh();
