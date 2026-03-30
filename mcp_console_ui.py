"""Browser UI for the MCP job console."""

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Imago Job Console</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #ddd;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; flex-shrink: 0; flex-wrap: wrap; }
  header h1 { font-size: 15px; font-weight: 600; color: #eee; letter-spacing: .02em; }
  header .subtitle { font-size: 12px; color: #666; }
  .main { display: flex; flex: 1; overflow: hidden; }
  .mobile-nav { display: none; padding: 8px 12px; gap: 8px; border-bottom: 1px solid #222;
                background: #141414; }
  .mobile-nav button { flex: 1; min-height: 40px; padding: 8px 10px; border-radius: 8px;
                       border: 1px solid #333; background: #1b1b1b; color: #aaa;
                       font-size: 12px; font-weight: 600; cursor: pointer; }
  .mobile-nav button.active { background: #243046; color: #eef4ff; border-color: #4a7fc1; }

  .job-list { width: 340px; flex-shrink: 0; border-right: 1px solid #2a2a2a;
              display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
  .job-list-header { padding: 8px 12px; font-size: 11px; font-weight: 600;
                     color: #666; text-transform: uppercase; letter-spacing: .08em;
                     border-bottom: 1px solid #222; background: #161616; flex-shrink: 0; }
  .job-list-body { overflow-y: auto; flex: 1; }
  .job-item { padding: 10px 12px; border-bottom: 1px solid #1e1e1e; cursor: pointer;
              display: flex; flex-direction: column; gap: 4px; transition: background .1s; }
  .job-item:hover { background: #1a1a1a; }
  .job-item.selected { background: #1c2333; border-left: 3px solid #4a7fc1; }
  .job-item .job-name { font-size: 12px; font-weight: 500; color: #ccc;
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .job-item .job-meta { font-size: 11px; color: #555; display: flex; gap: 8px; }
  .job-item .job-actions { margin-top: 4px; }
  .badge { display: inline-block; font-size: 10px; font-weight: 600; padding: 1px 6px;
           border-radius: 3px; text-transform: uppercase; letter-spacing: .04em; }
  .badge-running { background: #0d3d1a; color: #3fb950; }
  .badge-completed { background: #1e1e1e; color: #666; }
  .badge-failed { background: #3d0d0d; color: #f85149; }
  .badge-interrupted { background: #3d2a0d; color: #d29922; }
  .badge-cancelled { background: #3d2a0d; color: #d29922; }
  .badge-pending { background: #0d2a3d; color: #58a6ff; }
  .btn-cancel { font-size: 10px; padding: 2px 8px; border-radius: 3px; border: 1px solid #555;
                background: transparent; color: #bbb; cursor: pointer; }
  .btn-cancel:hover { border-color: #f85149; color: #f85149; }
  .btn-cancel:active { transform: translateY(1px); }
  .empty { padding: 24px 12px; text-align: center; font-size: 12px; color: #888; }

  .log-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
  .log-header { padding: 8px 14px; border-bottom: 1px solid #222; background: #161616;
                display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .log-header .log-title { font-size: 12px; font-weight: 500; color: #ccc; flex: 1;
                           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .log-header .log-controls { display: flex; align-items: center; gap: 10px; flex-shrink: 0;
                              flex-wrap: wrap; justify-content: flex-end; }
  .log-header label { font-size: 11px; color: #666; display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .log-header select { font-size: 11px; background: #222; color: #aaa; border: 1px solid #333;
                       border-radius: 3px; padding: 2px 4px; }
  .artifact-panel { border-bottom: 1px solid #222; background: #141414; padding: 10px 14px;
                    display: flex; flex-direction: column; gap: 8px; }
  .artifact-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
  .artifact-title { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #7d8ca3; }
  .artifact-count { font-size: 11px; color: #666; }
  .artifact-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px;
                   overflow-y: auto; max-height: 180px; }
  .artifact-group { border: 1px solid #2d3950; background: #192131; border-radius: 10px; padding: 10px; }
  .artifact-group strong { display: block; font-size: 12px; font-weight: 600; white-space: nowrap;
                           overflow: hidden; text-overflow: ellipsis; color: #dbe7ff; }
  .artifact-group span { display: block; margin-top: 3px; font-size: 11px; color: #9cb3d8; white-space: nowrap;
                         overflow: hidden; text-overflow: ellipsis; }
  .artifact-actions { display: flex; gap: 8px; margin-top: 9px; flex-wrap: wrap; }
  .artifact-item { text-align: left; padding: 8px 10px; border-radius: 8px;
                   border: 1px solid #2d3950; background: #1d2940; color: #dbe7ff; cursor: pointer; }
  .artifact-item:hover { border-color: #4a7fc1; background: #223554; }
  .artifact-item small { display: block; margin-top: 3px; font-size: 10px; color: #9cb3d8; }
  .artifact-empty, .artifact-error { font-size: 12px; color: #777; padding: 2px 0; }
  .artifact-error { color: #e38b8b; }
  .log-body { flex: 1; overflow-y: auto; padding: 10px 14px; }
  .xmp-panel { border-bottom: 1px solid #222; background: #101722; display: flex;
               flex-direction: column; min-height: 0; }
  .xmp-panel[hidden] { display: none; }
  .xmp-panel-header { padding: 12px 14px; border-bottom: 1px solid #22304a; display: flex;
                      align-items: flex-start; justify-content: space-between; gap: 12px; }
  .xmp-panel-title { min-width: 0; }
  .xmp-panel-title h2 { font-size: 14px; color: #eef4ff; margin-bottom: 3px; }
  .xmp-panel-title p { font-size: 11px; color: #8ea4c9; word-break: break-all; }
  .xmp-panel-close { border: 1px solid #3a4a68; background: #182338; color: #dbe7ff; border-radius: 8px;
                     min-width: 72px; min-height: 34px; cursor: pointer; }
  .xmp-panel-close:hover { background: #1e2d47; border-color: #587ebd; }
  .xmp-panel-body { overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 14px;
                    max-height: 50vh; }
  .xmp-panel-body pre, .log-body pre { font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
                                       font-size: 11.5px; line-height: 1.55; white-space: pre-wrap;
                                       word-break: break-all; color: #b0b0b0; }
  .log-placeholder { display: flex; align-items: center; justify-content: center;
                     height: 100%; font-size: 13px; color: #444; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
         background: #3fb950; margin-right: 6px; animation: pulse 1.5s infinite; }
  .xmp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
  .xmp-card { border: 1px solid #22304a; border-radius: 10px; background: #121c2d; padding: 10px; }
  .xmp-card h3 { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #87a3d7; margin-bottom: 8px; }
  .xmp-kv { display: flex; flex-direction: column; gap: 7px; }
  .xmp-kv div { display: flex; flex-direction: column; gap: 2px; }
  .xmp-kv span { font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #61779d; }
  .xmp-kv strong, .xmp-list li, .xmp-card p { font-size: 12px; color: #dbe7ff; font-weight: 500; }
  .xmp-list { list-style: none; display: flex; flex-wrap: wrap; gap: 6px; padding-left: 0; }
  .xmp-list li { border-radius: 999px; padding: 4px 8px; background: #1a2840; border: 1px solid #2a3d60; }
  .xmp-text { white-space: pre-wrap; word-break: break-word; }
  .xmp-subphotos { display: flex; flex-direction: column; gap: 8px; }
  .xmp-subphoto { border: 1px solid #22304a; border-radius: 8px; padding: 8px; background: #0f1726; }
  .xmp-subphoto strong { display: block; margin-bottom: 4px; font-size: 12px; color: #dbe7ff; }
  .xmp-subphoto p { font-size: 12px; color: #b9c8e3; }
  details { border: 1px solid #22304a; border-radius: 8px; background: #0f1726; padding: 8px 10px; }
  summary { cursor: pointer; color: #c8d9f9; font-size: 12px; font-weight: 600; }
  details pre { margin-top: 8px; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .3 } }
  @media (max-width: 900px) {
    body { overflow: auto; }
    .mobile-nav { display: flex; }
    .main { position: relative; }
    .job-list, .log-panel { width: 100%; flex: 1 1 100%; border-right: 0; }
    .job-list { border-bottom: 1px solid #222; }
    .main.mobile-jobs .log-panel { display: none; }
    .main.mobile-logs .job-list { display: none; }
    .log-header { align-items: flex-start; flex-direction: column; }
    .log-header .log-title { width: 100%; white-space: normal; }
    .log-header .log-controls { width: 100%; justify-content: space-between; }
    .artifact-list { grid-template-columns: 1fr; max-height: 220px; }
    .job-item .job-meta { flex-wrap: wrap; }
    .btn-cancel { min-height: 32px; padding: 4px 10px; }
    .xmp-grid { grid-template-columns: 1fr; }
    .xmp-panel-body { max-height: none; }
  }
</style>
</head>
<body>
<header>
  <h1>Imago Job Console</h1>
  <span class="subtitle">mcp/jobs &bull; auto-refreshes every 2s</span>
</header>
<div class="mobile-nav" id="mobile-nav">
  <button type="button" id="mobile-show-jobs" onclick="setMobilePanel('jobs')">Jobs</button>
  <button type="button" id="mobile-show-logs" onclick="setMobilePanel('logs')">Logs</button>
</div>
<div class="main" id="main">
  <div class="job-list">
    <div class="job-list-header">Jobs</div>
    <div class="job-list-body" id="job-list"></div>
  </div>
  <div class="log-panel">
    <div class="log-header">
      <span class="log-title" id="log-title">Select a job to view logs</span>
      <div class="log-controls">
        <label>
          Lines:
          <select id="log-lines">
            <option value="100">100</option>
            <option value="300">300</option>
            <option value="1000">1000</option>
            <option value="5000">5000</option>
          </select>
        </label>
        <label>
          <input type="checkbox" id="auto-scroll" checked> Auto-scroll
        </label>
      </div>
    </div>
    <div class="artifact-panel" id="artifact-panel" hidden>
      <div class="artifact-header">
        <span class="artifact-title">Photoalbums Outputs</span>
        <span class="artifact-count" id="artifact-count"></span>
      </div>
      <div class="artifact-list" id="artifact-list"></div>
    </div>
    <div class="xmp-panel" id="xmp-panel" hidden>
      <div class="xmp-panel-header">
        <div class="xmp-panel-title">
          <h2 id="xmp-panel-heading">XMP Review</h2>
          <p id="xmp-panel-subtitle"></p>
        </div>
        <button type="button" class="xmp-panel-close" id="xmp-panel-close">Close</button>
      </div>
      <div class="xmp-panel-body" id="xmp-panel-body"></div>
    </div>
    <div class="log-body" id="log-body">
      <div class="log-placeholder">No job selected</div>
    </div>
  </div>
</div>
<script>
  let selectedId = null;
  let jobs = [];
  let logLines = [];
  let jobArtifacts = [];
  let activeStream = null;
  let mobilePanel = 'jobs';

  function isMobileLayout() {
    return window.matchMedia('(max-width: 900px)').matches;
  }

  function updateMobileNav() {
    const main = document.getElementById('main');
    const jobsBtn = document.getElementById('mobile-show-jobs');
    const logsBtn = document.getElementById('mobile-show-logs');
    const current = isMobileLayout() ? mobilePanel : 'desktop';
    main.classList.toggle('mobile-jobs', current === 'jobs');
    main.classList.toggle('mobile-logs', current === 'logs');
    jobsBtn.classList.toggle('active', current === 'jobs');
    logsBtn.classList.toggle('active', current === 'logs');
    logsBtn.disabled = !selectedId;
  }

  function setMobilePanel(panel) {
    mobilePanel = panel === 'logs' && !selectedId ? 'jobs' : panel;
    updateMobileNav();
  }

  function badge(status) {
    return `<span class="badge badge-${status}">${status}</span>`;
  }

  function duration(job) {
    if (!job.started_at) return '';
    const start = new Date(job.started_at);
    const end = job.ended_at ? new Date(job.ended_at) : new Date();
    const seconds = Math.round((end - start) / 1000);
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  }

  function timeAgo(iso) {
    if (!iso) return '';
    const value = new Date(iso);
    return value.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  }

  function escHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fileName(path) {
    const text = String(path || '');
    const parts = text.split(/[\\/]/);
    return parts[parts.length - 1] || text;
  }

  function selectedJob() {
    return jobs.find((job) => job.id === selectedId) || null;
  }

  function isPhotoalbumsAiJob(job) {
    return !!job && String(job.name || '').startsWith('photoalbums_ai_index:');
  }

  async function fetchJobs() {
    try {
      const response = await fetch('/api/jobs');
      jobs = await response.json();
      renderJobList();
      if (selectedId) {
        const job = selectedJob();
        if (!job) {
          selectedId = null;
          jobArtifacts = [];
          logLines = [];
          renderArtifacts();
          renderLogLines();
          updateMobileNav();
          return;
        }
        await fetchArtifacts(selectedId);
      }
    } catch (error) {
      const element = document.getElementById('job-list');
      if (element && !jobs.length) {
        element.innerHTML = `<div class="empty">Error: ${escHtml(error.message)}</div>`;
      }
    }
  }

  function renderJobList() {
    const element = document.getElementById('job-list');
    if (!jobs.length) {
      element.innerHTML = '<div class="empty">No jobs yet</div>';
      return;
    }
    element.innerHTML = jobs.map((job) => {
      const selected = job.id === selectedId ? ' selected' : '';
      const cancelButton = job.status === 'running'
        ? `<div class="job-actions"><button class="btn-cancel" onclick="cancelJob('${job.id}', event)">Cancel</button></div>`
        : '';
      const live = job.status === 'running' ? '<span class="dot"></span>' : '';
      return `
        <div class="job-item${selected}" onclick="selectJob('${job.id}')">
          <div class="job-name">${live}${escHtml(job.name)}</div>
          <div class="job-meta">
            ${badge(job.status)}
            <span>${timeAgo(job.started_at)}</span>
            <span>${duration(job)}</span>
          </div>
          ${cancelButton}
        </div>`;
    }).join('');
  }

  function renderLogLines() {
    const body = document.getElementById('log-body');
    if (!selectedId) {
      body.innerHTML = '<div class="log-placeholder">No job selected</div>';
      return;
    }
    const autoScroll = document.getElementById('auto-scroll').checked;
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
    const count = parseInt(document.getElementById('log-lines').value, 10);
    const visible = logLines.slice(-count);
    body.innerHTML = `<pre>${escHtml(visible.join('\n') || '(no output yet)')}</pre>`;
    if (autoScroll && atBottom) {
      body.scrollTop = body.scrollHeight;
    }
  }

  function clearXmpPanel() {
    document.getElementById('xmp-panel').hidden = true;
    document.getElementById('xmp-panel-heading').textContent = 'XMP Review';
    document.getElementById('xmp-panel-subtitle').textContent = '';
    document.getElementById('xmp-panel-body').innerHTML = '';
  }

  async function fetchArtifacts(id) {
    const panel = document.getElementById('artifact-panel');
    if (!id) {
      jobArtifacts = [];
      panel.hidden = true;
      clearXmpPanel();
      renderArtifacts();
      return;
    }
    const job = jobs.find((item) => item.id === id);
    if (!isPhotoalbumsAiJob(job)) {
      jobArtifacts = [];
      panel.hidden = true;
      clearXmpPanel();
      renderArtifacts();
      return;
    }
    try {
      const response = await fetch(`/api/jobs/${id}/artifacts`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      jobArtifacts = await response.json();
      renderArtifacts();
    } catch (error) {
      jobArtifacts = [];
      renderArtifacts(`Error loading outputs: ${error.message}`);
    }
  }

  function renderArtifacts(errorMessage = '') {
    const panel = document.getElementById('artifact-panel');
    const count = document.getElementById('artifact-count');
    const list = document.getElementById('artifact-list');
    const job = selectedJob();
    if (!isPhotoalbumsAiJob(job)) {
      panel.hidden = true;
      list.innerHTML = '';
      count.textContent = '';
      return;
    }
    panel.hidden = false;
    count.textContent = jobArtifacts.length ? `${jobArtifacts.length} output(s)` : 'Waiting for outputs';
    if (errorMessage) {
      list.innerHTML = `<div class="artifact-error">${escHtml(errorMessage)}</div>`;
      return;
    }
    if (!jobArtifacts.length) {
      list.innerHTML = '<div class="artifact-empty">No outputs recorded yet for this job.</div>';
      return;
    }
    const groups = new Map();
    jobArtifacts.forEach((artifact, index) => {
      const key = String(artifact.image_path || artifact.sidecar_path || artifact.label || `artifact-${index}`);
      if (!groups.has(key)) {
        groups.set(key, {
          label: artifact.label || fileName(artifact.image_path || artifact.sidecar_path || ''),
          artifacts: [],
        });
      }
      groups.get(key).artifacts.push({ index, artifact });
    });
    list.innerHTML = Array.from(groups.values()).map((group) => {
      const xmpEntry = group.artifacts.find((entry) => String(entry.artifact.kind || '') === 'photoalbums_xmp');
      const promptEntry = group.artifacts.find((entry) => String(entry.artifact.kind || '') === 'photoalbums_prompts');
      const subtitle = fileName(
        (xmpEntry && xmpEntry.artifact.sidecar_path)
        || (promptEntry && promptEntry.artifact.image_path)
        || '',
      );
      const imagePath = (xmpEntry && xmpEntry.artifact.image_path)
        || (promptEntry && promptEntry.artifact.image_path)
        || '';
      const actions = [];
      if (imagePath) {
        actions.push(`
          <a href="/api/image?path=${encodeURIComponent(imagePath)}" target="_blank" rel="noopener" class="artifact-item">
            Image
            <small>${escHtml(fileName(imagePath))}</small>
          </a>`);
      }
      if (xmpEntry && xmpEntry.artifact.sidecar_path) {
        actions.push(`
          <button type="button" class="artifact-item"
                  data-sidecar="${escHtml(String(xmpEntry.artifact.sidecar_path || ''))}"
                  onclick="openXmpReview(this.dataset.sidecar)">
            XMP
            <small>${escHtml(fileName(xmpEntry.artifact.sidecar_path || ''))}</small>
          </button>`);
      }
      if (promptEntry) {
        const promptCount = Number(promptEntry.artifact.step_count || (promptEntry.artifact.steps || []).length || 0);
        const responseCount = (Array.isArray(promptEntry.artifact.steps) ? promptEntry.artifact.steps : [])
          .filter((step) => String((step && step.response) || '').trim())
          .length;
        actions.push(`
          <button type="button" class="artifact-item"
                  onclick="openPromptArtifact(${Number(promptEntry.index)})">
            Prompts
            <small>${escHtml(String(promptCount))} step(s)</small>
          </button>`);
        actions.push(`
          <button type="button" class="artifact-item"
                  onclick="openResponseArtifact(${Number(promptEntry.index)})">
            AI JSON
            <small>${escHtml(String(responseCount))} response(s)</small>
          </button>`);
      }
      return `
        <div class="artifact-group">
          <strong>${escHtml(group.label || subtitle)}</strong>
          <span>${escHtml(subtitle)}</span>
          <div class="artifact-actions">${actions.join('')}</div>
        </div>`;
    }).join('');
  }

  function renderListCard(title, items) {
    if (!items || !items.length) return '';
    return `
      <section class="xmp-card">
        <h3>${escHtml(title)}</h3>
        <ul class="xmp-list">
          ${items.map((item) => `<li>${escHtml(item)}</li>`).join('')}
        </ul>
      </section>`;
  }

  function renderValue(label, value) {
    if (!value && value !== 0 && value !== false) return '';
    return `<div><span>${escHtml(label)}</span><strong>${escHtml(value)}</strong></div>`;
  }

  function renderTextCard(title, value) {
    if (!value) return '';
    return `
      <section class="xmp-card">
        <h3>${escHtml(title)}</h3>
        <p class="xmp-text">${escHtml(value)}</p>
      </section>`;
  }

  function renderRawXmlCard(rawXml) {
    if (!rawXml) return '';
    return `
      <section class="xmp-card">
        <h3>Raw XMP</h3>
        <pre>${escHtml(rawXml)}</pre>
      </section>`;
  }

  function renderPromptStepCard(step) {
    const metadata = step && step.metadata && Object.keys(step.metadata).length
      ? `<section class="xmp-card"><h3>Metadata</h3><pre>${escHtml(JSON.stringify(step.metadata, null, 2))}</pre></section>`
      : '';
    return `
      <section class="xmp-card">
        <h3>${escHtml((step && step.step) || 'Prompt')}</h3>
        <div class="xmp-kv">
          ${renderValue('Engine', step && step.engine)}
          ${renderValue('Model', step && step.model)}
          ${renderValue('Prompt Source', step && step.prompt_source)}
          ${renderValue('Source Path', step && step.source_path)}
        </div>
      </section>
      ${step && step.system_prompt ? `<section class="xmp-card"><h3>System Prompt</h3><pre>${escHtml(step.system_prompt)}</pre></section>` : ''}
      ${step && step.prompt ? `<section class="xmp-card"><h3>User Prompt</h3><pre>${escHtml(step.prompt)}</pre></section>` : ''}
      ${metadata}`;
  }

  function renderResponseStepCard(step) {
    return `
      <section class="xmp-card">
        <h3>${escHtml((step && step.step) || 'AI JSON')}</h3>
        <div class="xmp-kv">
          ${renderValue('Engine', step && step.engine)}
          ${renderValue('Model', step && step.model)}
          ${renderValue('Finish Reason', step && step.finish_reason)}
        </div>
      </section>
      ${step && step.response ? `<section class="xmp-card"><h3>AI JSON</h3><pre>${escHtml(step.response)}</pre></section>` : ''}`;
  }

  function renderSummaryCard(summary, data) {
    return `
      <section class="xmp-card">
        <h3>Summary</h3>
        <div class="xmp-kv">
          ${renderValue('Creator Tool', data.creator_tool)}
          ${renderValue('Title', data.title)}
          ${renderValue('Album Title', data.album_title)}
          ${renderValue('People In Image', summary.people_in_image_count)}
          ${renderValue('Subjects', summary.subject_count)}
          ${renderValue('Detected People', summary.detected_people_count)}
          ${renderValue('Detected Objects', summary.detected_object_count)}
          ${renderValue('OCR Characters', summary.ocr_char_count)}
          ${renderValue('Subphotos', summary.subphoto_count)}
        </div>
      </section>`;
  }

  function renderMetadataCard(data) {
    const values = [
      renderValue('Source Text', data.source_text),
      renderValue('GPS Latitude', data.gps_latitude),
      renderValue('GPS Longitude', data.gps_longitude),
      renderValue('OCR Authority', data.ocr_authority_source),
      renderValue('Stitch Key', data.stitch_key),
      renderValue('OCR Ran', data.ocr_ran),
      renderValue('People Detected', data.people_detected),
      renderValue('People Identified', data.people_identified),
    ].join('');
    if (!values) return '';
    return `
      <section class="xmp-card">
        <h3>Metadata</h3>
        <div class="xmp-kv">${values}</div>
      </section>`;
  }

  function renderSubphotos(subphotos) {
    if (!subphotos || !subphotos.length) return '';
    return `
      <section class="xmp-card">
        <h3>Subphotos</h3>
        <div class="xmp-subphotos">
          ${subphotos.map((item) => `
            <div class="xmp-subphoto">
              <strong>Subphoto ${escHtml(item.index)}</strong>
              <p>${escHtml(item.description || '(no description)')}</p>
              <p>Bounds: ${escHtml(JSON.stringify(item.bounds || {}))}</p>
            </div>`).join('')}
        </div>
      </section>`;
  }

  function showXmpPanel(heading, subtitle, bodyHtml) {
    document.getElementById('xmp-panel-heading').textContent = heading;
    document.getElementById('xmp-panel-subtitle').textContent = subtitle;
    document.getElementById('xmp-panel-body').innerHTML = bodyHtml;
    document.getElementById('xmp-panel').hidden = false;
  }

  function openPromptArtifact(index) {
    const artifact = jobArtifacts[Number(index)];
    if (!artifact) {
      showXmpPanel('Prompt Debug Error', '', '<div class="artifact-error">Prompt artifact not found.</div>');
      return;
    }
    const steps = Array.isArray(artifact.steps) ? artifact.steps : [];
    const body = steps.length
      ? steps.map((step) => renderPromptStepCard(step)).join('')
      : '<div class="artifact-empty">No prompt steps recorded.</div>';
    showXmpPanel(
      `${fileName(artifact.image_path || artifact.label || 'Prompt Debug')} Prompts`,
      String(artifact.image_path || ''),
      body,
    );
  }

  function openResponseArtifact(index) {
    const artifact = jobArtifacts[Number(index)];
    if (!artifact) {
      showXmpPanel('AI JSON Error', '', '<div class="artifact-error">Prompt artifact not found.</div>');
      return;
    }
    const steps = (Array.isArray(artifact.steps) ? artifact.steps : [])
      .filter((step) => String((step && step.response) || '').trim());
    const body = steps.length
      ? steps.map((step) => renderResponseStepCard(step)).join('')
      : '<div class="artifact-empty">No AI JSON responses recorded.</div>';
    showXmpPanel(
      `${fileName(artifact.image_path || artifact.label || 'Prompt Debug')} AI JSON`,
      String(artifact.image_path || ''),
      body,
    );
  }

  async function openXmpReview(sidecarPath) {
    const subtitle = String(sidecarPath || '');
    showXmpPanel('XMP Review', subtitle, '<div class="artifact-empty">Loading XMP...</div>');
    try {
      const response = await fetch(`/api/xmp-review?sidecar_path=${encodeURIComponent(subtitle)}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      const summary = payload.summary || {};
      const parsedSections = [
        `<div class="xmp-grid">${renderSummaryCard(summary, payload)}${renderMetadataCard(payload)}</div>`,
        renderListCard('People', payload.person_names || []),
        renderListCard('Subjects', payload.subjects || []),
        renderTextCard('Description', payload.description),
        renderTextCard('OCR Text', payload.ocr_text),
        renderSubphotos(payload.subphotos || []),
      ].filter(Boolean);
      const sections = [
        renderRawXmlCard(payload.raw_xml),
      ].filter(Boolean);
      if (parsedSections.length) {
        sections.push(`
          <details>
            <summary>Parsed Summary</summary>
            ${parsedSections.join('')}
          </details>`);
      }
      if (payload.detections) {
        sections.push(`
          <details>
            <summary>Detections JSON</summary>
            <pre>${escHtml(JSON.stringify(payload.detections, null, 2))}</pre>
          </details>`);
      }
      showXmpPanel(fileName(payload.sidecar_path || subtitle), payload.sidecar_path || subtitle, sections.join(''));
    } catch (error) {
      showXmpPanel('XMP Review Error', subtitle, `<div class="artifact-error">${escHtml(error.message)}</div>`);
    }
  }

  function openStream(id) {
    if (activeStream) {
      activeStream.close();
      activeStream = null;
    }
    logLines = [];
    const job = jobs.find((item) => item.id === id);
    if (!job) return;
    document.getElementById('log-title').textContent = `${job.name} [${id}]`;
    renderLogLines();

    const source = new EventSource(`/api/jobs/${id}/stream`);
    activeStream = source;

    source.onmessage = (event) => {
      logLines.push(event.data);
      renderLogLines();
    };

    source.addEventListener('done', () => {
      source.close();
      activeStream = null;
      fetchJobs();
    });

    source.onerror = () => {
      source.close();
      activeStream = null;
      fetchJobs();
    };
  }

  async function selectJob(id) {
    selectedId = id;
    jobArtifacts = [];
    clearXmpPanel();
    renderJobList();
    renderArtifacts();
    openStream(id);
    await fetchArtifacts(id);
    if (isMobileLayout()) setMobilePanel('logs');
  }

  async function cancelJob(id, event) {
    event.stopPropagation();
    if (!confirm('Cancel this job?')) return;
    await fetch(`/api/jobs/${id}/cancel`, {method: 'POST'});
    await fetchJobs();
  }

  document.getElementById('log-lines').addEventListener('change', renderLogLines);
  document.getElementById('xmp-panel-close').addEventListener('click', clearXmpPanel);
  window.addEventListener('resize', updateMobileNav);

  setInterval(fetchJobs, 2000);
  updateMobileNav();
  fetchJobs();
</script>
</body>
</html>
"""
