let subtitleEditorProgrammaticScroll = false;
let subtitleEditorUserScrolling = false;
let _subtitleEditorUserScrollTimer = null;

function normalizeSubtitleText(raw) {
  return String(raw || '').replace(/\s+/g, ' ').trim();
}

function parseSubtitleConfidenceValue(raw) {
  const text = String(raw || '').trim();
  if (!text) return null;
  const value = Number(text);
  if (!Number.isFinite(value)) return null;
  return _clamp(value, 0, 1);
}

function formatSubtitleConfidenceValue(raw) {
  const value = parseSubtitleConfidenceValue(raw);
  if (value === null) return '';
  return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
}

function canonicalizeSubtitlesEntries(rawEntries) {
  const durationRaw = Number(chapterDurationSeconds() || 0);
  const duration = Number.isFinite(durationRaw) && durationRaw > 0 ? durationRaw : null;
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
    const text = normalizeSubtitleText(item && item.text);
    if (start === null || end === null || end <= start || !text) return;
    const a = duration === null ? Math.max(0, start) : _clamp(start, 0, duration);
    const b = duration === null ? Math.max(0, end) : _clamp(end, 0, duration);
    if (b <= a) return;
    const speaker = normalizeSubtitleText(item && item.speaker);
    const confidence = parseSubtitleConfidenceValue(item && item.confidence);
    const source = normalizeSubtitleText(item && item.source);
    rows.push({
      start: Number(a.toFixed(3)),
      end: Number(b.toFixed(3)),
      text,
      speaker,
      confidence,
      source,
      idx: Number(idx || 0),
    });
  });
  rows.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    if (a.end !== b.end) return a.end - b.end;
    return a.idx - b.idx;
  });
  return rows.map((row) => ({
    start_seconds: row.start,
    end_seconds: row.end,
    start: formatTimestampSeconds(row.start),
    end: formatTimestampSeconds(row.end),
    text: row.text,
    speaker: row.speaker || '',
    confidence: row.confidence,
    source: row.source || '',
  }));
}

function normalizeSubtitlesProfile(rawProfile) {
  const raw = (rawProfile && typeof rawProfile === 'object') ? rawProfile : {};
  return {
    entries: canonicalizeSubtitlesEntries(raw.entries || []),
    source: String(raw.source || 'default'),
  };
}

function subtitlesProfileFromApiResult(result) {
  const payload = (result && typeof result === 'object') ? result : {};
  let rawProfile = null;
  if (payload.subtitles_profile && typeof payload.subtitles_profile === 'object') {
    rawProfile = payload.subtitles_profile;
  } else if (payload.subtitles && typeof payload.subtitles === 'object') {
    rawProfile = payload.subtitles;
  } else if (payload.profile && typeof payload.profile === 'object') {
    const nested = payload.profile;
    if (nested.subtitles_profile && typeof nested.subtitles_profile === 'object') {
      rawProfile = nested.subtitles_profile;
    } else if (nested.subtitles && typeof nested.subtitles === 'object') {
      rawProfile = nested.subtitles;
    }
  }

  let profile = normalizeSubtitlesProfile(rawProfile || null);
  const generatedCount = Math.max(0, Math.trunc(Number(payload.generated_count || payload.count || 0) || 0));
  if (profile.entries.length > 0 || generatedCount <= 0) {
    return profile;
  }

  let fallbackEntries = [];
  if (rawProfile && Array.isArray(rawProfile.entries)) {
    fallbackEntries = rawProfile.entries;
  } else if (Array.isArray(payload.subtitle_entries)) {
    fallbackEntries = payload.subtitle_entries;
  } else if (Array.isArray(payload.entries)) {
    fallbackEntries = payload.entries;
  }
  if (!fallbackEntries.length) {
    return profile;
  }
  profile = normalizeSubtitlesProfile({
    entries: fallbackEntries,
    source: String((rawProfile && rawProfile.source) || payload.source || 'api_fallback'),
  });
  return profile;
}

function canonicalizeSplitEntries(rawEntries) {
  const frameCount = splitChapterFrameBounds().frameCount;
  const rows = [];
  (Array.isArray(rawEntries) ? rawEntries : []).forEach((item, idx) => {
    const startRaw = item && Object.prototype.hasOwnProperty.call(item, 'start_frame')
      ? item.start_frame
      : (item ? item.start : null);
    const endRaw = item && Object.prototype.hasOwnProperty.call(item, 'end_frame')
      ? item.end_frame
      : (item ? item.end : null);
    const start = parseFrameIndex(startRaw);
    const end = parseFrameIndex(endRaw);
    const title = normalizeSubtitleText(item && (item.title ?? item.text));
    if (start === null || end === null || end <= start) return;
    const a = _clamp(start, 0, frameCount);
    const b = _clamp(end, 0, frameCount);
    if (b <= a) return;
    const normalizedTitle = title || `Chapter ${Math.max(1, Math.trunc(Number(idx || 0)) + 1)}`;
    rows.push({
      start_frame: Math.trunc(a),
      end_frame: Math.trunc(b),
      title: normalizedTitle,
      idx: Number(idx || 0),
    });
  });
  rows.sort((a, b) => {
    if (a.start_frame !== b.start_frame) return a.start_frame - b.start_frame;
    if (a.end_frame !== b.end_frame) return a.end_frame - b.end_frame;
    return a.idx - b.idx;
  });
  return rows.slice(0, 1).map((row) => ({
    start_frame: row.start_frame,
    end_frame: row.end_frame,
    start: String(row.start_frame),
    end: String(row.end_frame),
    title: row.title,
  }));
}

function normalizeSplitProfile(rawProfile) {
  const raw = (rawProfile && typeof rawProfile === 'object') ? rawProfile : {};
  return {
    entries: canonicalizeSplitEntries(raw.entries || []),
    source: String(raw.source || 'default'),
  };
}

function splitEntriesForTimeline() {
  return canonicalizeSplitEntries((state.splitProfile && state.splitProfile.entries) || []).map((row, idx) => ({
    start_frame: Math.max(0, Math.trunc(Number(row.start_frame || 0))),
    end_frame: Math.max(0, Math.trunc(Number(row.end_frame || 0))),
    start_seconds: secondsFromFrameIndex(row.start_frame),
    end_seconds: secondsFromFrameIndex(row.end_frame),
    text: String(row.title || ''),
    idx: Number(idx || 0),
  }));
}

function splitEditorHasFocus() {
  return Boolean(splitEditorEl && splitEditorEl.contains(document.activeElement));
}

function splitEntriesToEditorRows(entries) {
  const rows = canonicalizeSplitEntries(entries);
  const bounds = splitChapterFrameBounds();
  const row = rows[0] || {
    start_frame: 0,
    end_frame: bounds.frameCount,
    title: String(state.chapter || ''),
  };
  return [{
    start: String(splitDisplayStartFrame(row.start_frame)),
    end: String(splitDisplayEndFrameInclusive(row.end_frame, row.start_frame)),
    title: String(row.title || ''),
  }];
}

function renderSplitEditorGrid(entries) {
  if (!splitEditorEl) return;
  const rows = splitEntriesToEditorRows(entries);
  const bodyHtml = rows.map((row, idx) => {
    return `
      <tr data-split-row="1" data-split-row-idx="${idx}">
        <td><input class="subtitles-editor-cell start" type="text" spellcheck="false" data-split-field="start" value="${escapeHtml(row.start)}" placeholder="0" aria-label="Chapter start frame"></td>
        <td><input class="subtitles-editor-cell end" type="text" spellcheck="false" data-split-field="end" value="${escapeHtml(row.end)}" placeholder="0" aria-label="Chapter end frame"></td>
        <td><input class="subtitles-editor-cell text" type="text" spellcheck="false" data-split-field="title" value="${escapeHtml(row.title)}" placeholder="Chapter title" aria-label="Chapter title"></td>
      </tr>
    `;
  }).join('');
  splitEditorEl.innerHTML = `
    <table class="subtitles-editor-grid">
      <thead>
        <tr>
          <th style="width:110px">Start Frame</th>
          <th style="width:110px">End Frame</th>
          <th>Title</th>
        </tr>
      </thead>
      <tbody>${bodyHtml}</tbody>
    </table>
    <div class="subtitles-editor-hint">Edit the loaded chapter range with archive frame numbers. End frame is inclusive. Title is optional.</div>
  `;
}

function parseSplitEditorGrid() {
  const entries = [];
  const errors = [];
  if (!splitEditorEl) {
    return { entries: canonicalizeSplitEntries((state.splitProfile && state.splitProfile.entries) || []), errors };
  }
  const rowEls = Array.from(splitEditorEl.querySelectorAll('tr[data-split-row="1"]'));
  rowEls.forEach((rowEl, idx) => {
    const rowNum = idx + 1;
    const start = String((rowEl.querySelector('[data-split-field="start"]') && rowEl.querySelector('[data-split-field="start"]').value) || '').trim();
    const end = String((rowEl.querySelector('[data-split-field="end"]') && rowEl.querySelector('[data-split-field="end"]').value) || '').trim();
    const title = String((rowEl.querySelector('[data-split-field="title"]') && rowEl.querySelector('[data-split-field="title"]').value) || '').trim();
    const hasAny = Boolean(start || end || title);
    if (!hasAny) return;
    if (start && !end && !title) return;
    if (!start || !end) {
      errors.push(`Row ${rowNum}: start frame and end frame are required.`);
      return;
    }
    const startFrameLocal = parseSplitInputFrameToLocal(start, { isEnd: false });
    const endFrameLocalInclusive = parseSplitInputFrameToLocal(end, { isEnd: true });
    const endFrameLocal = endFrameLocalInclusive === null ? null : (endFrameLocalInclusive + 1);
    const bounds = splitChapterFrameBounds();
    if (startFrameLocal === null || endFrameLocal === null) {
      errors.push(`Row ${rowNum}: start/end must be valid chapter frame numbers.`);
      return;
    }
    if (
      startFrameLocal < 0
      || startFrameLocal >= bounds.frameCount
      || endFrameLocal <= 0
      || endFrameLocal > bounds.frameCount
    ) {
      errors.push(`Row ${rowNum}: frame range is outside this chapter.`);
      return;
    }
    if (endFrameLocal <= startFrameLocal) {
      errors.push(`Row ${rowNum}: end frame must be greater than start frame.`);
      return;
    }
    entries.push({ start_frame: startFrameLocal, end_frame: endFrameLocal, title: title || '' });
  });
  return { entries: canonicalizeSplitEntries(entries), errors };
}

function syncSplitEditorToCursor(cursorSeconds, options = {}) {
  if (!splitEditorEl || !isSplitStepActive()) return;
  const force = Boolean(options && options.force);
  if (!force && splitEditorHasFocus()) return;

  const rows = canonicalizeSplitEntries((state.splitProfile && state.splitProfile.entries) || []);
  const rowEls = Array.from(splitEditorEl.querySelectorAll('tr[data-split-row="1"]'));
  if (!rows.length || !rowEls.length) {
    rowEls.forEach((rowEl) => rowEl.classList.remove('active-row'));
    return;
  }
  const cursorFrame = frameIndexFromSeconds(cursorSeconds);
  let targetIdx = 0;
  if (Number.isFinite(cursorFrame)) {
    targetIdx = -1;
    for (let i = 0; i < rows.length; i += 1) {
      const startFrame = Number(rows[i] && rows[i].start_frame);
      const endFrame = Number(rows[i] && rows[i].end_frame);
      if (!Number.isFinite(startFrame) || !Number.isFinite(endFrame)) continue;
      if (cursorFrame >= startFrame && cursorFrame < endFrame) {
        targetIdx = i;
        break;
      }
      if (targetIdx < 0 && startFrame >= cursorFrame) {
        targetIdx = i;
        break;
      }
    }
    if (targetIdx < 0) {
      targetIdx = Math.max(0, rows.length - 1);
    }
  }
  rowEls.forEach((rowEl, idx) => {
    rowEl.classList.toggle('active-row', idx === targetIdx);
  });

  const targetRow = rowEls[targetIdx];
  if (!targetRow) return;
  const headerEl = splitEditorEl.querySelector('thead');
  const headerHeight = headerEl ? Number(headerEl.getBoundingClientRect().height || 0) : 0;
  const rowTop = Number(targetRow.offsetTop || 0);
  const rowHeight = Number(targetRow.offsetHeight || 0);
  const viewTop = Number(splitEditorEl.scrollTop || 0) + headerHeight;
  const viewBottom = Number(splitEditorEl.scrollTop || 0) + Number(splitEditorEl.clientHeight || 0);
  const pad = 18;
  const outside = rowTop < (viewTop + pad) || (rowTop + rowHeight) > (viewBottom - pad);
  if (outside) {
    const desired = Math.max(
      0,
      rowTop - Math.max(0, Math.floor((Number(splitEditorEl.clientHeight || 0) - rowHeight) / 2))
    );
    splitEditorEl.scrollTo({ top: desired, behavior: 'auto' });
  }
}

function refreshSplitEditorFromState() {
  renderSplitEditorGrid((state.splitProfile && state.splitProfile.entries) || []);
  syncSplitEditorToCursor(currentPeopleTimelineCursorSeconds(), { force: true });
  updateSplitMeta();
}

function syncSplitProfileFromEditor(showErrors = true) {
  if (!splitEditorEl) {
    state.splitProfile = normalizeSplitProfile(state.splitProfile || null);
    return true;
  }
  const parsed = parseSplitEditorGrid();
  if (parsed.errors.length) {
    if (showErrors) {
      setStatus(parsed.errors[0], true);
    }
    return false;
  }
  state.splitProfile = {
    ...(state.splitProfile || {}),
    entries: parsed.entries,
  };
  if (!splitEditorHasFocus()) {
    renderSplitEditorGrid(parsed.entries);
    syncSplitEditorToCursor(currentPeopleTimelineCursorSeconds(), { force: true });
  }
  updateSplitMeta();
  return true;
}

function updateSplitMeta() {
  if (splitEditorEl && !splitEditorHasFocus()) {
    renderSplitEditorGrid((state.splitProfile && state.splitProfile.entries) || []);
  }
  if (!isSplitStepActive()) {
    renderPeopleTimeline();
    return;
  }
  renderPeopleTimeline();
}

function subtitleEditorHasFocus() {
  return Boolean(subtitlesEditorEl && subtitlesEditorEl.contains(document.activeElement));
}

function peopleEditorHasFocus() {
  return Boolean(peopleEditorEl && peopleEditorEl.contains(document.activeElement));
}

function subtitleEntriesToEditorRows(entries) {
  const rows = canonicalizeSubtitlesEntries(entries);
  const mapped = rows.map((row) => ({
    start: String(row.start || '').trim(),
    end: String(row.end || '').trim(),
    text: String(row.text || '').trim(),
    speaker: String(row.speaker || '').trim(),
    confidence: formatSubtitleConfidenceValue(row.confidence),
    source: String(row.source || '').trim(),
    isBlank: false,
  }));
  mapped.push({
    start: '',
    end: '',
    text: '',
    speaker: '',
    confidence: '',
    source: '',
    isBlank: true,
  });
  return mapped;
}

function renderSubtitlesEditorGrid(entries) {
  if (!subtitlesEditorEl) return;
  const savedScroll = subtitlesEditorEl.scrollTop;
  const rows = subtitleEntriesToEditorRows(entries);
  const bodyHtml = rows.map((row, idx) => {
    const rowNum = idx + 1;
    const deleteHtml = row.isBlank
      ? ''
      : `<button class="subtitles-editor-row-delete" type="button" data-subtitle-row-delete="${idx}" title="Delete subtitle row">x</button>`;
    return `
      <tr data-sub-row="1" data-sub-row-idx="${idx}">
        <td class="rownum">${rowNum}</td>
        <td><input class="subtitles-editor-cell start" type="text" spellcheck="false" data-sub-field="start" value="${escapeHtml(row.start)}" placeholder="00:00:00.000" aria-label="Subtitle start"></td>
        <td><input class="subtitles-editor-cell end" type="text" spellcheck="false" data-sub-field="end" value="${escapeHtml(row.end)}" placeholder="00:00:00.000" aria-label="Subtitle end"></td>
        <td><input class="subtitles-editor-cell text" type="text" spellcheck="false" data-sub-field="text" value="${escapeHtml(row.text)}" placeholder="Subtitle text" aria-label="Subtitle text"></td>
        <td><input class="subtitles-editor-cell speaker" type="text" spellcheck="false" data-sub-field="speaker" value="${escapeHtml(row.speaker)}" placeholder="Speaker" aria-label="Subtitle speaker"></td>
        <td><input class="subtitles-editor-cell confidence" type="text" spellcheck="false" data-sub-field="confidence" value="${escapeHtml(row.confidence)}" placeholder="0-1" aria-label="Subtitle confidence"></td>
        <td><input class="subtitles-editor-cell source" type="text" spellcheck="false" data-sub-field="source" value="${escapeHtml(row.source)}" placeholder="Source" aria-label="Subtitle source"></td>
        <td>${deleteHtml}</td>
      </tr>
    `;
  }).join('');
  subtitlesEditorEl.innerHTML = `
    <table class="subtitles-editor-grid">
      <thead>
        <tr>
          <th style="width:34px">#</th>
          <th style="width:110px">Start</th>
          <th style="width:110px">End</th>
          <th>Text</th>
          <th style="width:100px">Speaker</th>
          <th style="display:none">Confidence</th>
          <th style="display:none">Source</th>
          <th style="width:26px"></th>
        </tr>
      </thead>
      <tbody>${bodyHtml}</tbody>
    </table>
    <div class="subtitles-editor-hint">Click each cell to edit fields. Leave the last blank row empty or fill it to add a new subtitle.</div>
  `;
  subtitlesEditorEl.scrollTop = savedScroll;
}

function parseSubtitlesEditorGrid() {
  const entries = [];
  const errors = [];
  if (!subtitlesEditorEl) {
    return { entries: canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []), errors };
  }
  const rowEls = Array.from(subtitlesEditorEl.querySelectorAll('tr[data-sub-row="1"]'));
  rowEls.forEach((rowEl, idx) => {
    const rowNum = idx + 1;
    const start = String((rowEl.querySelector('[data-sub-field="start"]') && rowEl.querySelector('[data-sub-field="start"]').value) || '').trim();
    const end = String((rowEl.querySelector('[data-sub-field="end"]') && rowEl.querySelector('[data-sub-field="end"]').value) || '').trim();
    const text = String((rowEl.querySelector('[data-sub-field="text"]') && rowEl.querySelector('[data-sub-field="text"]').value) || '').trim();
    const speaker = String((rowEl.querySelector('[data-sub-field="speaker"]') && rowEl.querySelector('[data-sub-field="speaker"]').value) || '').trim();
    const confidence = String((rowEl.querySelector('[data-sub-field="confidence"]') && rowEl.querySelector('[data-sub-field="confidence"]').value) || '').trim();
    const source = String((rowEl.querySelector('[data-sub-field="source"]') && rowEl.querySelector('[data-sub-field="source"]').value) || '').trim();
    const hasAny = Boolean(start || end || text || speaker || confidence || source);
    if (!hasAny) return;
    if (!start || !end || !text) {
      errors.push(`Row ${rowNum}: start, end, and text are required.`);
      return;
    }
    if (parseTimestampSeconds(start) === null || parseTimestampSeconds(end) === null) {
      errors.push(`Row ${rowNum}: start/end must be valid timestamps.`);
      return;
    }
    const startSec = parseTimestampSeconds(start);
    const endSec = parseTimestampSeconds(end);
    if (startSec === null || endSec === null || endSec <= startSec) {
      errors.push(`Row ${rowNum}: end must be greater than start.`);
      return;
    }
    if (confidence && parseSubtitleConfidenceValue(confidence) === null) {
      errors.push(`Row ${rowNum}: confidence must be a number between 0 and 1.`);
      return;
    }
    entries.push({ start, end, text, speaker, confidence, source });
  });
  return { entries: canonicalizeSubtitlesEntries(entries), errors };
}

function syncSubtitlesEditorToCursor(cursorSeconds, options = {}) {
  if (!subtitlesEditorEl || !isSubtitlesStepActive()) return;
  const force = Boolean(options && options.force);
  if (subtitleEditorHasFocus()) return;

  const rows = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  const rowEls = Array.from(subtitlesEditorEl.querySelectorAll('tr[data-sub-row="1"]'));
  if (!rows.length || !rowEls.length) {
    rowEls.forEach((rowEl) => rowEl.classList.remove('active-row'));
    return;
  }

  const sec = Number(cursorSeconds);
  let targetIdx = 0;
  if (Number.isFinite(sec)) {
    targetIdx = -1;
    for (let i = 0; i < rows.length; i += 1) {
      const startSec = Number(rows[i] && rows[i].start_seconds);
      const endSec = Number(rows[i] && rows[i].end_seconds);
      if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) continue;
      if (sec >= startSec && sec <= endSec) {
        targetIdx = i;
        break;
      }
      if (targetIdx < 0 && startSec >= sec) {
        targetIdx = i;
        break;
      }
    }
    if (targetIdx < 0) {
      targetIdx = Math.max(0, rows.length - 1);
    }
  }

  rowEls.forEach((rowEl, idx) => {
    rowEl.classList.toggle('active-row', idx === targetIdx);
  });

  if (!force || subtitleEditorUserScrolling) return;
  const targetRow = rowEls[targetIdx];
  if (!targetRow) return;
  const headerEl = subtitlesEditorEl.querySelector('thead');
  const headerHeight = headerEl ? Number(headerEl.getBoundingClientRect().height || 0) : 0;
  const rowTop = Number(targetRow.offsetTop || 0);
  const rowHeight = Number(targetRow.offsetHeight || 0);
  const viewTop = Number(subtitlesEditorEl.scrollTop || 0) + headerHeight;
  const viewBottom = Number(subtitlesEditorEl.scrollTop || 0) + Number(subtitlesEditorEl.clientHeight || 0);
  const pad = 18;
  const outside = rowTop < (viewTop + pad) || (rowTop + rowHeight) > (viewBottom - pad);
  if (outside) {
    const desired = Math.max(
      0,
      rowTop - Math.max(0, Math.floor((Number(subtitlesEditorEl.clientHeight || 0) - rowHeight) / 2))
    );
    subtitleEditorProgrammaticScroll = true;
    subtitlesEditorEl.scrollTo({ top: desired, behavior: 'auto' });
  }
}

function refreshSubtitlesEditorFromState() {
  subtitleTimelineDraft = null;
  renderSubtitlesEditorGrid((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  syncSubtitlesEditorToCursor(currentPeopleTimelineCursorSeconds(), { force: true });
  updateSubtitlesMeta();
}

function syncSubtitlesProfileFromEditor(showErrors = true) {
  if (!subtitlesEditorEl) {
    state.subtitlesProfile = normalizeSubtitlesProfile(state.subtitlesProfile || null);
    return true;
  }
  const parsed = parseSubtitlesEditorGrid();
  if (parsed.errors.length) {
    if (showErrors) {
      setStatus(parsed.errors[0], true);
    }
    return false;
  }
  state.subtitlesProfile = {
    ...(state.subtitlesProfile || {}),
    entries: parsed.entries,
  };
  if (!subtitleEditorHasFocus()) {
    renderSubtitlesEditorGrid(parsed.entries);
  }
  updateSubtitlesMeta();
  return true;
}

function deleteSubtitleTimelineEntry(entryIndex) {
  const idx = Number(entryIndex);
  const rows = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  if (!Number.isFinite(idx) || idx < 0 || idx >= rows.length) return;
  const next = rows.slice(0, idx).concat(rows.slice(idx + 1));
  state.subtitlesProfile = {
    ...(state.subtitlesProfile || {}),
    entries: canonicalizeSubtitlesEntries(next),
  };
  if (subtitleTimelineDraft) {
    const draftIdx = Math.trunc(Number(subtitleTimelineDraft.entryIndex));
    if (draftIdx === idx) {
      subtitleTimelineDraft = null;
    } else if (Number.isFinite(draftIdx) && draftIdx > idx) {
      subtitleTimelineDraft = {
        ...subtitleTimelineDraft,
        entryIndex: draftIdx - 1,
      };
    }
  }
  refreshSubtitlesEditorFromState();
  updateReviewStatsDisplay();
  setStatus('Deleted subtitle entry.');
}

function editSubtitleTimelineEntry(entryIndex) {
  const idx = Math.trunc(Number(entryIndex));
  const rows = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  if (!Number.isFinite(idx) || idx < 0 || idx >= rows.length) return;
  const row = rows[idx];
  peopleTimelineDraft = null;
  peopleTimelineDrag = null;
  subtitleTimelineDraft = {
    entryIndex: idx,
    text: String(row && row.text || ''),
  };
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
}

function clearSubtitleTimelineDraft() {
  if (!subtitleTimelineDraft) return;
  subtitleTimelineDraft = null;
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
}

function commitSubtitleTimelineDraft() {
  if (!subtitleTimelineDraft) return false;
  const idx = Math.trunc(Number(subtitleTimelineDraft.entryIndex));
  const rows = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  if (!Number.isFinite(idx) || idx < 0 || idx >= rows.length) {
    subtitleTimelineDraft = null;
    updateSubtitlesMeta();
    return false;
  }
  const priorText = String(rows[idx] && rows[idx].text || '');
  const nextText = normalizeSubtitleText(subtitleTimelineDraft.text);
  if (!nextText) {
    setStatus('Subtitle text cannot be empty.', true);
    return false;
  }
  if (nextText !== priorText) {
    rows[idx] = {
      ...rows[idx],
      text: nextText,
    };
    state.subtitlesProfile = {
      ...(state.subtitlesProfile || {}),
      entries: canonicalizeSubtitlesEntries(rows),
    };
    updateReviewStatsDisplay();
    setStatus('Updated subtitle text.');
  }
  subtitleTimelineDraft = null;
  updateSubtitlesMeta();
  return true;
}

function updateSubtitlesMeta() {
  if (subtitlesMetaEl) {
    if (!isSubtitlesStepActive()) {
      subtitlesMetaEl.textContent = '';
    } else {
      subtitlesMetaEl.textContent = '';
    }
  }
  if (subtitlesEditorEl && !subtitleEditorHasFocus()) {
    renderSubtitlesEditorGrid((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  }
  if (flipbookPreviewEl && flipbookPreviewEl.classList.contains('active') && sparkPlayFrames.length) {
    renderFlipbookSubtitles(flipbookCurrentFrameSeconds());
  }
  renderPeopleTimeline();
}

function updatePeopleMeta() {
  if (peopleMetaEl) {
    if (!isPeopleStepActive()) {
      peopleMetaEl.textContent = '';
    } else {
      const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
      if (!entries.length) {
        peopleMetaEl.textContent = 'People rows: no entries yet.';
      } else {
        const preview = entries
          .slice(0, 3)
          .map((row) => `${row.start}-${row.end} ${row.people}`)
          .join(' | ');
        const suffix = entries.length > 3 ? ` | +${entries.length - 3} more` : '';
        peopleMetaEl.textContent = `People rows: ${entries.length} entr${entries.length === 1 ? 'y' : 'ies'} | ${preview}${suffix}`;
      }
    }
  }
  renderPeopleTimeline();
}

function currentPeopleTimelineCursorSeconds() {
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (frames.length && timelineScrubEl) {
    const idx = _clamp(Number(timelineScrubEl.value || 0), 0, frames.length - 1);
    return chapterLocalSecondsFromFid(frames[idx] && frames[idx].fid);
  }
  if (state.visibleRange && frames.length) {
    const startIdx = _clamp(Number(state.visibleRange.start || 0), 0, frames.length - 1);
    const endIdx = _clamp(Number(state.visibleRange.end || startIdx), startIdx, frames.length - 1);
    const centerIdx = _clamp(Math.round((startIdx + endIdx) / 2), 0, frames.length - 1);
    return chapterLocalSecondsFromFid(frames[centerIdx] && frames[centerIdx].fid);
  }
  return 0;
}
