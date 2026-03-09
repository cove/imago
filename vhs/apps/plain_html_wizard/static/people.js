function canonicalizePeopleEntries(rawEntries) {
  const duration = chapterDurationSeconds();
  const rows = [];
  (Array.isArray(rawEntries) ? rawEntries : []).forEach((item, idx) => {
    const startRaw = item && Object.prototype.hasOwnProperty.call(item, 'start_seconds')
      ? item.start_seconds
      : (item ? item.start : null);
    const endRaw = item && Object.prototype.hasOwnProperty.call(item, 'end_seconds')
      ? item.end_seconds
      : (item ? item.end : null);
    const start = parseTimestampSeconds(startRaw);
    const end = parseTimestampSeconds(endRaw);
    const people = String((item && item.people) || '').replace(/\s+/g, ' ').trim();
    const laneRaw = item && Object.prototype.hasOwnProperty.call(item, 'lane') ? item.lane : 0;
    const laneNum = Number(laneRaw);
    const lane = Number.isFinite(laneNum) ? Math.max(0, Math.trunc(laneNum)) : 0;
    if (!people) return;
    if (start === null || end === null || end <= start) return;
    const a = _clamp(start, 0, duration);
    const b = _clamp(end, 0, duration);
    if (b <= a) return;
    rows.push({ start: Number(a.toFixed(3)), end: Number(b.toFixed(3)), people, lane, idx });
  });
  if (!rows.length) return [];
  rows.sort((a, b) => {
    if (a.lane !== b.lane) return a.lane - b.lane;
    if (a.start !== b.start) return a.start - b.start;
    if (a.end !== b.end) return a.end - b.end;
    return a.idx - b.idx;
  });
  const out = [];
  rows.forEach((row) => {
    if (!out.length) {
      out.push({
        start_seconds: row.start,
        end_seconds: row.end,
        start: formatTimestampSeconds(row.start),
        end: formatTimestampSeconds(row.end),
        people: row.people,
        lane: row.lane,
      });
      return;
    }
    const prev = out[out.length - 1];
    if (
      String(prev.people) === row.people
      && Number(prev.lane || 0) === row.lane
      && Number(prev.end_seconds) + 0.001 >= row.start
    ) {
      prev.end_seconds = Number(Math.max(Number(prev.end_seconds), row.end).toFixed(3));
      prev.end = formatTimestampSeconds(prev.end_seconds);
      return;
    }
    out.push({
      start_seconds: row.start,
      end_seconds: row.end,
      start: formatTimestampSeconds(row.start),
      end: formatTimestampSeconds(row.end),
      people: row.people,
      lane: row.lane,
    });
  });
  return out;
}

function normalizePeopleProfile(rawProfile) {
  const raw = (rawProfile && typeof rawProfile === 'object') ? rawProfile : {};
  return {
    entries: canonicalizePeopleEntries(raw.entries || []),
    source: String(raw.source || 'default'),
  };
}

function renderPeopleEditorGrid(entries) {
  if (!peopleEditorEl) return;
  const savedScroll = peopleEditorEl.scrollTop;
  const rows = canonicalizePeopleEntries(entries);
  const withBlank = rows.concat([{ start: '', end: '', people: '' }]);
  const body = withBlank.map((row, idx) => {
    const deleteBtn = idx < rows.length
      ? `<button class="subtitles-editor-row-delete" type="button" data-people-row-delete="${idx}" title="Delete people row">x</button>`
      : '';
    return `
      <tr data-people-row="1">
        <td class="rownum">${idx + 1}</td>
        <td><input class="subtitles-editor-cell start" type="text" spellcheck="false" data-people-field="start" value="${escapeHtml(String(row.start || ''))}" placeholder="00:00:00.000" aria-label="People start"></td>
        <td><input class="subtitles-editor-cell end" type="text" spellcheck="false" data-people-field="end" value="${escapeHtml(String(row.end || ''))}" placeholder="00:00:00.000" aria-label="People end"></td>
        <td><input class="subtitles-editor-cell text" type="text" spellcheck="false" data-people-field="people" value="${escapeHtml(String(row.people || ''))}" placeholder="Person or people" aria-label="People names"></td>
        <td>${deleteBtn}</td>
      </tr>
    `;
  }).join('');
  peopleEditorEl.innerHTML = `
    <table class="subtitles-editor-grid">
      <thead>
        <tr>
          <th style="width:34px">#</th>
          <th style="width:110px">Start</th>
          <th style="width:110px">End</th>
          <th>People</th>
          <th style="width:26px"></th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
    <div class="subtitles-editor-hint">Edits save to <code>people.tsv</code> as <code>start</code>, <code>end</code>, and <code>people</code>.</div>
  `;
  peopleEditorEl.scrollTop = savedScroll;
}

function parsePeopleEditorGrid() {
  const errors = [];
  if (!peopleEditorEl) {
    return { entries: canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []), errors };
  }
  const rowEls = Array.from(peopleEditorEl.querySelectorAll('tr[data-people-row="1"]'));
  const entries = [];
  rowEls.forEach((rowEl, idx) => {
    const startEl = rowEl.querySelector('[data-people-field="start"]');
    const endEl = rowEl.querySelector('[data-people-field="end"]');
    const peopleEl = rowEl.querySelector('[data-people-field="people"]');
    const start = String(startEl && startEl.value || '').trim();
    const end = String(endEl && endEl.value || '').trim();
    const people = String(peopleEl && peopleEl.value || '').trim();
    if (!start && !end && !people) return;
    if (!start || !end || !people) {
      errors.push(`People row ${idx + 1}: start, end, and people are required.`);
      return;
    }
    entries.push({ start, end, people });
  });
  return { entries: canonicalizePeopleEntries(entries), errors };
}

function syncPeopleEditorToCursor(cursorSeconds, options = {}) {
  if (!peopleEditorEl || !isPeopleStepActive()) return;
  const force = Boolean(options.force);
  const rows = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  const rowEls = Array.from(peopleEditorEl.querySelectorAll('tr[data-people-row="1"]'));
  if (!rows.length || !rowEls.length) return;
  let activeIdx = rows.findIndex((row) => (
    Number(row.start_seconds) <= cursorSeconds + 0.001
    && Number(row.end_seconds) >= cursorSeconds - 0.001
  ));
  if (activeIdx < 0) activeIdx = rows.findIndex((row) => Number(row.start_seconds) > cursorSeconds);
  if (activeIdx < 0) activeIdx = Math.max(0, rows.length - 1);
  rowEls.forEach((rowEl, idx) => {
    rowEl.classList.toggle('active-row', idx === activeIdx);
  });
  if (!force) return;
  const target = rowEls[activeIdx];
  if (!target) return;
  const headerEl = peopleEditorEl.querySelector('thead');
  const headerHeight = Number(headerEl && headerEl.getBoundingClientRect().height || 0);
  const rowTop = Number(target.offsetTop || 0);
  const rowHeight = Number(target.getBoundingClientRect().height || 0);
  const desired = Math.max(0, rowTop - Math.max(0, Math.floor((Number(peopleEditorEl.clientHeight || 0) - rowHeight) / 2)) - headerHeight);
  peopleEditorEl.scrollTo({ top: desired, behavior: 'auto' });
}

function refreshPeopleEditorFromState() {
  renderPeopleEditorGrid((state.peopleProfile && state.peopleProfile.entries) || []);
  syncPeopleEditorToCursor(currentPeopleTimelineCursorSeconds(), { force: true });
  updatePeopleMeta();
}

function syncPeopleProfileFromEditor(showErrors = true) {
  if (!peopleEditorEl) {
    state.peopleProfile = normalizePeopleProfile(state.peopleProfile || null);
    return true;
  }
  const parsed = parsePeopleEditorGrid();
  if (parsed.errors.length) {
    if (showErrors) {
      setStatus(parsed.errors[0], true);
    }
    return false;
  }
  state.peopleProfile = {
    ...(state.peopleProfile || {}),
    entries: parsed.entries,
  };
  peopleTimelineDrag = null;
  if (!peopleEditorHasFocus()) {
    renderPeopleEditorGrid(parsed.entries);
  }
  updatePeopleMeta();
  return true;
}

