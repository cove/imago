function isReviewFullscreenActive() {
  return document.fullscreenElement === page2El || document.body.classList.contains('review-fullscreen-fallback');
}

function isFlipbookFocusModeActive() {
  return Boolean(page2El && page2El.classList.contains('flipbook-focus'));
}

function clearFlipbookSubtitleRail() {
  if (!flipbookSubtitleRailEl) return;
  flipbookSubtitleRailEl.classList.remove('empty');
  flipbookSubtitleRailEl.innerHTML = '';
  flipbookSubtitleRailRenderKey = '';
  flipbookSubtitleRailActiveIndex = -1;
  flipbookSubtitleRailManualUntilMs = 0;
}

function updateFlipbookSubtitleRailMode() {
  if (!page2El) return;
  const on = Boolean(
    page2El.classList.contains('flipbook-focus')
    && (isPeopleStepActive() || isSubtitlesStepActive())
    && flipbookPreviewEl
    && flipbookPreviewEl.classList.contains('active')
  );
  page2El.classList.toggle('flipbook-subtitle-mode', on);
  if (!on) clearFlipbookSubtitleRail();
}

function markFlipbookSubtitleRailManual(ms = 1400) {
  const nowMs = (typeof performance !== 'undefined' && performance.now)
    ? performance.now()
    : Date.now();
  flipbookSubtitleRailManualUntilMs = nowMs + Math.max(0, Number(ms || 0));
}

function setFlipbookFocusMode(active) {
  if (!page2El) return;
  const on = Boolean(active);
  page2El.classList.toggle('flipbook-focus', on);
  if (!on) {
    syncFrameGridPlaybackCursor(-1);
  }
  updateFlipbookSubtitleRailMode();
  updatePeopleStepLayoutSizing();
}

async function ensureReviewFullscreenActive() {
  if (!page2El || isReviewFullscreenActive()) return;
  if (document.fullscreenEnabled && page2El.requestFullscreen) {
    try {
      await page2El.requestFullscreen();
      return;
    } catch (_err) {}
  }
  document.body.classList.add('review-fullscreen-fallback');
}

async function exitReviewFullscreenIfActive() {
  if (document.fullscreenElement === page2El && document.exitFullscreen) {
    try {
      await document.exitFullscreen();
    } catch (_err) {}
  }
  if (document.body.classList.contains('review-fullscreen-fallback')) {
    document.body.classList.remove('review-fullscreen-fallback');
  }
}

function _clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, Number(value)));
}

function ensureFrameGridVirtualElements() {
  if (!frameGridEl) return null;
  if (frameGridSizerEl && frameGridItemsEl) return { sizer: frameGridSizerEl, items: frameGridItemsEl };
  frameGridEl.innerHTML = '';
  frameGridSizerEl = document.createElement('div');
  frameGridSizerEl.className = 'frame-grid-sizer';
  frameGridItemsEl = document.createElement('div');
  frameGridItemsEl.className = 'frame-grid-items';
  frameGridSizerEl.appendChild(frameGridItemsEl);
  frameGridEl.appendChild(frameGridSizerEl);
  return { sizer: frameGridSizerEl, items: frameGridItemsEl };
}

function currentReviewFrames() {
  return (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
}

function frameGridMetrics(frameCountRaw) {
  const frameCount = Math.max(0, Math.trunc(Number(frameCountRaw || 0)));
  const viewportWidth = Math.max(1, Number(frameGridEl && frameGridEl.clientWidth) || 1);
  const innerWidth = Math.max(1, viewportWidth - (FRAME_GRID_PADDING_PX * 2));
  const columnCount = Math.max(
    1,
    Math.floor((innerWidth + FRAME_GRID_GAP_PX) / (FRAME_GRID_CARD_MIN_WIDTH_PX + FRAME_GRID_GAP_PX))
  );
  const itemWidth = Math.max(
    120,
    Math.floor((innerWidth - (Math.max(0, columnCount - 1) * FRAME_GRID_GAP_PX)) / columnCount)
  );
  const rowCount = Math.max(1, Math.ceil(frameCount / columnCount));
  const rowStride = FRAME_GRID_CARD_HEIGHT_PX + FRAME_GRID_GAP_PX;
  const totalHeight = (
    (FRAME_GRID_PADDING_PX * 2)
    + (rowCount * FRAME_GRID_CARD_HEIGHT_PX)
    + (Math.max(0, rowCount - 1) * FRAME_GRID_GAP_PX)
  );
  return {
    frameCount,
    columnCount,
    itemWidth,
    itemHeight: FRAME_GRID_CARD_HEIGHT_PX,
    rowCount,
    rowStride,
    totalHeight,
  };
}

function frameGridVisibleIndexRange(metrics, overscanRows = 0) {
  if (!metrics || metrics.frameCount <= 0) return null;
  const scrollTop = Math.max(0, Number(frameGridEl && frameGridEl.scrollTop) || 0);
  const viewportHeight = Math.max(1, Number(frameGridEl && frameGridEl.clientHeight) || 1);
  const topPx = Math.max(0, scrollTop - FRAME_GRID_PADDING_PX);
  const bottomPx = Math.max(0, (scrollTop + viewportHeight) - FRAME_GRID_PADDING_PX);
  const firstVisibleRow = Math.max(0, Math.floor(topPx / metrics.rowStride));
  const lastVisibleRow = Math.min(
    metrics.rowCount - 1,
    Math.floor(Math.max(topPx, bottomPx - 1) / metrics.rowStride)
  );
  const startRow = Math.max(0, firstVisibleRow - Math.max(0, overscanRows));
  const endRow = Math.min(metrics.rowCount - 1, lastVisibleRow + Math.max(0, overscanRows));
  const start = Math.max(0, startRow * metrics.columnCount);
  const end = Math.min(
    metrics.frameCount - 1,
    ((endRow + 1) * metrics.columnCount) - 1
  );
  const visibleStart = Math.max(0, firstVisibleRow * metrics.columnCount);
  const visibleEnd = Math.min(
    metrics.frameCount - 1,
    ((lastVisibleRow + 1) * metrics.columnCount) - 1
  );
  return { start, end, visibleStart, visibleEnd };
}

function _sparkClientXToIndex(clientX) {
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) return null;
  const rect = iqrSparkEl.getBoundingClientRect();
  if (!rect.width || rect.width <= 0) return null;
  const x = _clamp(Number(clientX) - rect.left, 0, rect.width);
  const ratio = rect.width <= 1 ? 0 : x / rect.width;
  const maxIndex = Math.max(0, state.review.frames.length - 1);
  return _clamp(Math.round(ratio * maxIndex), 0, maxIndex);
}

function _scrollFrameGridToIndex(index) {
  const frames = currentReviewFrames();
  if (!frameGridEl || !frames.length) return;
  const metrics = frameGridMetrics(frames.length);
  const idx = _clamp(Number(index), 0, frames.length - 1);
  const row = Math.floor(idx / metrics.columnCount);
  const top = Math.max(
    0,
    FRAME_GRID_PADDING_PX
      + (row * metrics.rowStride)
      - Math.floor(frameGridEl.clientHeight * 0.12)
  );
  frameGridEl.scrollTo({ top, behavior: 'auto' });
  scheduleVisibleRangeRefresh();
}

function scrubTimelineToIndex(rawIndex, options = {}) {
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) return;
  const scrollGrid = !(options && options.scrollGrid === false);
  const forceMeta = Boolean(options && options.forceMeta);
  const idx = _clamp(Number(rawIndex), 0, state.review.frames.length - 1);
  if (timelineScrubEl) {
    timelineScrubEl.value = String(idx);
  }
  updateTimelineAudioPlayheadFromIndex(idx);
  if (scrollGrid) {
    _scrollFrameGridToIndex(idx);
  }
  const frame = state.review.frames[idx] || null;
  const cursorSeconds = frame ? chapterLocalSecondsFromFid(frame.fid) : 0;
  renderPeopleTimeline(cursorSeconds);
  if (timelineScrubEl && (document.activeElement === timelineScrubEl || forceMeta)) {
    const tc = frame ? timelineLabelFromFid(frame.fid) : '';
    if (timelineScrubMetaEl) {
      timelineScrubMetaEl.textContent = `Timeline: ${tc || '?'} (${idx + 1}/${state.review.frames.length})`;
    }
  }
}

function beginPeopleTimelineDraftFromPoint(clientX, clientY) {
  if (!isPeopleStepActive()) return;
  const secRaw = peopleTimelineSecondsFromClientX(clientX);
  if (!Number.isFinite(secRaw)) return;
  const duration = Math.max(chapterDurationSeconds(), timelineFrameStepSeconds());
  const step = timelineFrameStepSeconds();
  const start = _clamp(snapTimelineSeconds(secRaw), 0, Math.max(0, duration - step));
  const end = _clamp(
    snapTimelineSeconds(start + PEOPLE_TIMELINE_DEFAULT_DURATION_SECONDS),
    start + step,
    duration
  );
  const lane = peopleTimelineLaneFromClientY(clientY);
  openPeopleTimelineDraft({
    mode: 'new',
    lane,
    start,
    end,
    text: '',
  });
}

function deletePeopleTimelineEntry(entryIndex) {
  const idx = Math.trunc(Number(entryIndex));
  const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  if (!Number.isFinite(idx) || idx < 0 || idx >= entries.length) return;
  entries.splice(idx, 1);
  state.peopleProfile = {
    ...(state.peopleProfile || {}),
    entries: canonicalizePeopleEntries(entries),
  };
  refreshPeopleEditorFromState();
  updateReviewStatsDisplay();
}

function openPeopleTimelineEdit(entryIndex) {
  const idx = Math.trunc(Number(entryIndex));
  const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  if (!Number.isFinite(idx) || idx < 0 || idx >= entries.length) return;
  const row = entries[idx];
  openPeopleTimelineDraft({
    mode: 'edit',
    entryIndex: idx,
    lane: Number(row.lane || 0),
    start: Number(row.start_seconds || 0),
    end: Number(row.end_seconds || 0),
    text: String(row.people || ''),
  });
}

function startPeopleTimelineDrag(event, kind, entryIndex) {
  if (!isPeopleStepActive()) return;
  const idx = Math.trunc(Number(entryIndex));
  const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  if (!Number.isFinite(idx) || idx < 0 || idx >= entries.length) return;
  const row = entries[idx];
  const start = Number(row.start_seconds || 0);
  const end = Number(row.end_seconds || 0);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
  const grabSecRaw = peopleTimelineSecondsFromClientX(event.clientX);
  const grabSec = Number.isFinite(grabSecRaw) ? grabSecRaw : start;
  peopleTimelineDrag = {
    pointerId: event.pointerId,
    kind: String(kind || 'move'),
    entryIndex: idx,
    baseEntries: entries.map((item) => ({ ...item })),
    start,
    end,
    lane: Math.max(0, Math.trunc(Number(row.lane || 0))),
    grabOffset: grabSec - start,
  };
  peopleTimelineDraft = null;
  if (peopleTimelineEl && peopleTimelineEl.setPointerCapture) {
    try {
      peopleTimelineEl.setPointerCapture(event.pointerId);
    } catch (_err) {}
  }
  event.preventDefault();
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
}

function updatePeopleTimelineDrag(event) {
  if (!peopleTimelineDrag) return;
  if (Number(event.pointerId) !== Number(peopleTimelineDrag.pointerId)) return;
  const secRaw = peopleTimelineSecondsFromClientX(event.clientX);
  if (!Number.isFinite(secRaw)) return;
  const step = timelineFrameStepSeconds();
  const duration = Math.max(
    chapterDurationSeconds(),
    Number(peopleTimelineRenderState && peopleTimelineRenderState.duration) || 0,
    Number(peopleTimelineDrag.end || 0)
  );
  const baseStart = Number(peopleTimelineDrag.start || 0);
  const baseEnd = Number(peopleTimelineDrag.end || 0);
  const baseDuration = Math.max(step, baseEnd - baseStart);
  let nextStart = baseStart;
  let nextEnd = baseEnd;
  const snappedSec = _clamp(snapTimelineSeconds(secRaw), 0, duration);

  if (peopleTimelineDrag.kind === 'resize-start') {
    nextStart = _clamp(snappedSec, 0, Math.max(0, baseEnd - step));
  } else if (peopleTimelineDrag.kind === 'resize-end') {
    nextEnd = _clamp(snappedSec, Math.min(duration, baseStart + step), duration);
  } else {
    nextStart = snapTimelineSeconds(snappedSec - Number(peopleTimelineDrag.grabOffset || 0));
    nextStart = _clamp(nextStart, 0, Math.max(0, duration - baseDuration));
    nextEnd = _clamp(snapTimelineSeconds(nextStart + baseDuration), nextStart + step, duration);
  }

  if (nextEnd <= nextStart) {
    nextEnd = _clamp(nextStart + step, nextStart + step, duration);
  }

  const working = (peopleTimelineDrag.baseEntries || []).map((item) => ({ ...item }));
  if (!working.length) return;
  if (peopleTimelineDrag.entryIndex < 0 || peopleTimelineDrag.entryIndex >= working.length) return;
  const target = working[peopleTimelineDrag.entryIndex];
  target.start_seconds = Number(nextStart.toFixed(3));
  target.end_seconds = Number(nextEnd.toFixed(3));
  target.start = formatTimestampSeconds(target.start_seconds);
  target.end = formatTimestampSeconds(target.end_seconds);
  target.lane = Math.max(0, Math.trunc(Number(peopleTimelineDrag.lane || target.lane || 0)));

  state.peopleProfile = {
    ...(state.peopleProfile || {}),
    entries: working,
  };
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
  event.preventDefault();
}

function finishPeopleTimelineDrag(event = null) {
  if (!peopleTimelineDrag) return;
  if (event && Number(event.pointerId) !== Number(peopleTimelineDrag.pointerId)) return;
  const pointerId = Number(peopleTimelineDrag.pointerId);
  peopleTimelineDrag = null;
  if (peopleTimelineEl && peopleTimelineEl.releasePointerCapture && Number.isFinite(pointerId)) {
    try {
      peopleTimelineEl.releasePointerCapture(pointerId);
    } catch (_err) {}
  }
  state.peopleProfile = {
    ...(state.peopleProfile || {}),
    entries: canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []),
  };
  refreshPeopleEditorFromState();
  updateReviewStatsDisplay();
}

function _centerFrameCardInGrid(card) {
  if (!frameGridEl || !card) return;

  const viewportH = Math.max(1, window.innerHeight || document.documentElement.clientHeight || 1);
  const gridRect = frameGridEl.getBoundingClientRect();
  if (gridRect.bottom < 0 || gridRect.top > viewportH) {
    frameGridEl.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'auto' });
  }

  const cardRect = card.getBoundingClientRect();
  const padY = Math.max(18, Math.floor(gridRect.height * 0.2));
  const padX = Math.max(18, Math.floor(gridRect.width * 0.12));
  const inViewY = cardRect.top >= (gridRect.top + padY) && cardRect.bottom <= (gridRect.bottom - padY);
  const inViewX = cardRect.left >= (gridRect.left + padX) && cardRect.right <= (gridRect.right - padX);
  if (inViewY && inViewX) return;

  if (typeof card.scrollIntoView === 'function') {
    card.scrollIntoView({ block: 'center', inline: 'center', behavior: 'auto' });
  } else {
    const top = Math.max(0, card.offsetTop - Math.floor((frameGridEl.clientHeight - card.offsetHeight) / 2));
    const left = Math.max(0, card.offsetLeft - Math.floor((frameGridEl.clientWidth - card.offsetWidth) / 2));
    frameGridEl.scrollTo({ top, left, behavior: 'auto' });
  }
  scheduleVisibleRangeRefresh();
}

function syncFrameGridPlaybackCursor(index, keepVisible = false) {
  const frames = currentReviewFrames();
  if (!frames.length) {
    flipbookGridCursorIndex = -1;
    renderFrameGridWindow(true);
    return;
  }
  const nextRaw = Number(index);
  if (!Number.isFinite(nextRaw) || nextRaw < 0) {
    flipbookGridCursorIndex = -1;
    renderFrameGridWindow(true);
    return;
  }
  const idx = _clamp(nextRaw, 0, frames.length - 1);
  if (keepVisible) {
    _scrollFrameGridToIndex(idx);
  }
  const prevIdx = flipbookGridCursorIndex;
  flipbookGridCursorIndex = idx;
  // Fast path: toggle the class on just the two affected cards without rebuilding the fragment.
  if (frameGridItemsEl && prevIdx !== idx) {
    if (prevIdx >= 0) {
      const prevCard = frameGridItemsEl.querySelector(`[data-index="${prevIdx}"]`);
      if (prevCard) prevCard.classList.remove('playback-current');
    }
    const nextCard = frameGridItemsEl.querySelector(`[data-index="${idx}"]`);
    if (nextCard) {
      nextCard.classList.add('playback-current');
      return;
    }
  } else if (frameGridItemsEl && prevIdx === idx) {
    return;
  }
  renderFrameGridWindow(true);
}

function updateSparkPlayButton() {
  if (!flipbookPlayBtnEl) return;
  const playing = Boolean(sparkPlayTimer);
  flipbookPlayBtnEl.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
  flipbookPlayBtnEl.title = playing ? 'Pause flipbook playback' : 'Play flipbook';
}

function updateFlipbookControls() {
  const hasFrames = sparkPlayFrames.length > 0;
  if (flipbookRevBtnEl) {
    flipbookRevBtnEl.disabled = !hasFrames;
  }
  if (flipbookPlayBtnEl) {
    flipbookPlayBtnEl.disabled = !hasFrames;
  }
  if (flipbookFwdBtnEl) {
    flipbookFwdBtnEl.disabled = !hasFrames;
  }
  if (flipbookVolumeEl) {
    flipbookVolumeEl.disabled = !hasFrames;
  }
  if (flipbookFrameEl) {
    const frame = sparkPlayFrames[sparkPlayIndex] || null;
    flipbookFrameEl.classList.toggle('good', Boolean(frame && frame.status === 'good'));
    flipbookFrameEl.classList.toggle('bad', Boolean(frame && frame.status === 'bad'));
    const canToggle = Boolean(frame) && !sparkPlayTimer && !flipbookToggleInFlight;
    flipbookFrameEl.classList.toggle('paused', canToggle);
    if (canToggle) {
      const nextState = frame.status === 'bad' ? 'good' : 'bad';
      flipbookFrameEl.title = `Click to mark this frame ${nextState.toUpperCase()}`;
    } else {
      flipbookFrameEl.removeAttribute('title');
    }
  }
}

function resetFlipbookAudioState() {
  if (flipbookAudioEl) {
    try {
      flipbookAudioEl.pause();
    } catch (_err) {}
    flipbookAudioEl.removeAttribute('src');
    flipbookAudioEl.load();
  }
  sparkPlayUseAudioClock = false;
  flipbookAudioSrcKey = '';
}

function applyFlipbookVolume() {
  if (!flipbookAudioEl) return;
  const value = Number(flipbookVolumeEl && flipbookVolumeEl.value);
  const volume = _clamp(Number.isFinite(value) ? value : 1.0, 0, 1);
  flipbookAudioEl.volume = volume;
}

function flipbookFrameSecondsForIndex(rawIndex) {
  if (!sparkPlayFrames.length) return 0;
  const idx = _clamp(Number(rawIndex || 0), 0, sparkPlayFrames.length - 1);
  const frame = sparkPlayFrames[idx] || null;
  if (!frame) return 0;
  return chapterLocalSecondsFromFid(frame.fid);
}

function flipbookCurrentFrameSeconds() {
  return flipbookFrameSecondsForIndex(sparkPlayIndex);
}

function flipbookIndexFromChapterSeconds(rawSeconds) {
  if (!sparkPlayFrames.length) return 0;
  const sec = Math.max(0, Number(rawSeconds || 0));
  // O(1): frames are ordered by fid which maps linearly to time, so interpolate directly.
  const first = sparkPlayFrames[0];
  const last = sparkPlayFrames[sparkPlayFrames.length - 1];
  const startFid = Math.trunc(Number((first && first.fid) || 0));
  const endFid = Math.trunc(Number((last && last.fid) || startFid));
  const targetFid = startFid + Math.round(sec * TIMELINE_FPS_NUM / TIMELINE_FPS_DEN);
  const fidRange = Math.max(1, endFid - startFid);
  const fraction = (targetFid - startFid) / fidRange;
  return _clamp(Math.round(fraction * (sparkPlayFrames.length - 1)), 0, sparkPlayFrames.length - 1);
}

function loadedFrameSecondsRange() {
  const chapterDuration = Math.max(0, Number(chapterDurationSeconds() || 0));
  const fallback = chapterDuration > 0
    ? { start: 0, end: chapterDuration }
    : null;
  const frames = reviewLoadedFrames();
  if (!frames.length) return fallback;
  let minSec = Number.POSITIVE_INFINITY;
  let maxSec = Number.NEGATIVE_INFINITY;
  frames.forEach((frame) => {
    const sec = chapterLocalSecondsFromFid(frame && frame.fid);
    if (!Number.isFinite(sec)) return;
    if (sec < minSec) minSec = sec;
    if (sec > maxSec) maxSec = sec;
  });
  if (!Number.isFinite(minSec) || !Number.isFinite(maxSec)) return fallback;
  const start = Math.max(0, minSec);
  const end = Math.max(0, maxSec + timelineFrameStepSeconds());
  if (fallback && (end <= start || start >= fallback.end || end <= 0)) {
    return fallback;
  }
  return {
    start,
    end,
  };
}

function subtitleEntriesForFlipbook() {
  if (isSplitStepActive()) {
    return splitEntriesForTimeline().map((row) => ({
      start_seconds: Number(row.start_seconds || 0),
      end_seconds: Number(row.end_seconds || 0),
      start: formatFrameIndex(splitDisplayStartFrame(row.start_frame)),
      end: formatFrameIndex(splitDisplayEndFrameInclusive(row.end_frame, row.start_frame)),
      text: String(row.text || ''),
      speaker: '',
      confidence: null,
      source: '',
    }));
  }
  if (isPeopleStepActive()) {
    return canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []).map((row) => ({
      start_seconds: Number(row.start_seconds || 0),
      end_seconds: Number(row.end_seconds || 0),
      start: String(row.start || formatTimestampSeconds(Number(row.start_seconds || 0))),
      end: String(row.end || formatTimestampSeconds(Number(row.end_seconds || 0))),
      text: String(row.people || ''),
      speaker: '',
      confidence: null,
      source: '',
    }));
  }
  const canonical = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  if (canonical.length) return canonical;
  const rawEntries = (state.subtitlesProfile && Array.isArray(state.subtitlesProfile.entries))
    ? state.subtitlesProfile.entries
    : [];
  const fallback = [];
  rawEntries.forEach((item, idx) => {
    if (!item || typeof item !== 'object') return;
    const start = parseTimestampSeconds(item.start_seconds ?? item.start ?? item.begin ?? item.t0);
    const end = parseTimestampSeconds(item.end_seconds ?? item.end ?? item.stop ?? item.t1);
    const text = normalizeSubtitleText(
      item.text ?? item.subtitle ?? item.caption ?? item.line ?? item.value ?? ''
    );
    if (start === null || end === null || end <= start || !text) return;
    const speaker = normalizeSubtitleText(item.speaker ?? item.name ?? '');
    const confidence = parseSubtitleConfidenceValue(item.confidence ?? item.score);
    const source = normalizeSubtitleText(item.source ?? item.provider ?? '');
    const a = Number(start.toFixed(3));
    const b = Number(end.toFixed(3));
    fallback.push({
      start_seconds: a,
      end_seconds: b,
      start: formatTimestampSeconds(a),
      end: formatTimestampSeconds(b),
      text,
      speaker,
      confidence,
      source,
      idx: Number(idx || 0),
    });
  });
  fallback.sort((a, b) => {
    if (a.start_seconds !== b.start_seconds) return a.start_seconds - b.start_seconds;
    if (a.end_seconds !== b.end_seconds) return a.end_seconds - b.end_seconds;
    return Number(a.idx || 0) - Number(b.idx || 0);
  });
  return fallback.map(({ idx, ...row }) => row);
}

function findSubtitlesAroundSeconds(rawSeconds) {
  const sec = Math.max(0, Number(rawSeconds || 0));
  const range = loadedFrameSecondsRange();
  const allEntries = subtitleEntriesForFlipbook();
  let entries = allEntries
    .filter((row) => {
      if (!range) return true;
      const start = Number(row.start_seconds || 0);
      const end = Number(row.end_seconds || 0);
      return end > range.start && start < range.end;
    });
  // If range derivation is off, fall back to available subtitle rows instead of showing empty.
  if (range && !entries.length && allEntries.length) {
    entries = allEntries;
  }
  const active = [];
  let prev = null;
  let next = null;
  entries.forEach((row) => {
    const start = Number(row.start_seconds || 0);
    const end = Number(row.end_seconds || 0);
    if (start <= (sec + 0.0005) && sec <= (end + 0.0005)) {
      active.push(row);
      return;
    }
    if (end <= sec && (!prev || Number(prev.end_seconds || 0) < end)) {
      prev = row;
      return;
    }
    if (start > sec && (!next || Number(next.start_seconds || 0) > start)) {
      next = row;
    }
  });
  return { entries, active, prev, next };
}

function flipbookSubtitleRailEntryKey(entries) {
  const rows = Array.isArray(entries) ? entries : [];
  let checksum = 0;
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i] || {};
    const startMs = Math.round(Number(row.start_seconds || 0) * 1000);
    const endMs = Math.round(Number(row.end_seconds || 0) * 1000);
    const text = String(row.text || '');
    const speaker = String(row.speaker || '');
    const textHead = text.length ? text.charCodeAt(0) : 0;
    const textTail = text.length ? text.charCodeAt(text.length - 1) : 0;
    checksum = (
      checksum
      + ((i + 1) * 17)
      + (startMs * 31)
      + (endMs * 13)
      + (text.length * 7)
      + (speaker.length * 11)
      + textHead
      + textTail
    ) % 2147483647;
  }
  return `${rows.length}|${checksum}`;
}

function flipbookSubtitleRailCenterIndex(around, sec) {
  const entries = Array.isArray(around && around.entries) ? around.entries : [];
  if (!entries.length) return -1;
  if (Array.isArray(around && around.active) && around.active.length) {
    const idx = entries.indexOf(around.active[0]);
    if (idx >= 0) return idx;
  }
  for (let i = 0; i < entries.length; i += 1) {
    if (Number(entries[i].start_seconds || 0) > sec) return i;
  }
  return entries.length - 1;
}

function renderFlipbookSubtitleRail(rawSeconds, around) {
  if (!flipbookSubtitleRailEl) return;
  if (isSplitStepActive()) {
    clearFlipbookSubtitleRail();
    return;
  }
  const railMode = Boolean(page2El && page2El.classList.contains('flipbook-subtitle-mode'));
  if (!railMode) {
    clearFlipbookSubtitleRail();
    return;
  }
  const sec = Math.max(0, Number(rawSeconds || 0));
  const entries = Array.isArray(around && around.entries) ? around.entries : [];
  if (!entries.length) {
    flipbookSubtitleRailEl.classList.add('empty');
    flipbookSubtitleRailEl.textContent = isPeopleStepActive()
      ? 'No people entries for this chapter.'
      : 'No subtitle entries for this chapter.';
    flipbookSubtitleRailRenderKey = 'empty';
    flipbookSubtitleRailActiveIndex = -1;
    return;
  }

  const nextKey = flipbookSubtitleRailEntryKey(entries);
  const needRebuild = nextKey !== flipbookSubtitleRailRenderKey;
  if (needRebuild) {
    const rowsHtml = entries.map((row, idx) => {
      const startText = String(row.start || formatTimestampSeconds(Number(row.start_seconds || 0)));
      const endText = String(row.end || formatTimestampSeconds(Number(row.end_seconds || 0)));
      const speaker = String(row.speaker || '').trim();
      const speakerHtml = speaker
        ? `<span class="flipbook-subtitle-rail-speaker">${escapeHtml(speaker)}:</span>`
        : '';
      return `
        <div class="flipbook-subtitle-rail-row" data-subtitle-rail-index="${idx}" data-subtitle-start-seconds="${Number(row.start_seconds || 0)}">
          <div class="flipbook-subtitle-rail-time">${escapeHtml(`${startText} - ${endText}`)}</div>
          <div class="flipbook-subtitle-rail-text">${speakerHtml}${escapeHtml(String(row.text || ''))}</div>
        </div>
      `;
    }).join('');
    flipbookSubtitleRailEl.classList.remove('empty');
    flipbookSubtitleRailEl.innerHTML = rowsHtml;
    flipbookSubtitleRailRenderKey = nextKey;
    flipbookSubtitleRailActiveIndex = -1;
  }

  const activeIdx = flipbookSubtitleRailCenterIndex(around, sec);
  if (activeIdx < 0) return;
  const previous = flipbookSubtitleRailEl.querySelector('.flipbook-subtitle-rail-row.playback-current');
  if (previous) previous.classList.remove('playback-current');
  const activeRow = flipbookSubtitleRailEl.querySelector(`[data-subtitle-rail-index="${activeIdx}"]`);
  if (!activeRow) return;
  activeRow.classList.add('playback-current');
  const railHeight = Number(flipbookSubtitleRailEl.clientHeight || 0);
  if (railHeight > 0) {
    const railRect = flipbookSubtitleRailEl.getBoundingClientRect();
    const rowRect = activeRow.getBoundingClientRect();
    const rowCenter = Number(flipbookSubtitleRailEl.scrollTop || 0)
      + (Number(rowRect.top || 0) - Number(railRect.top || 0))
      + (Number(rowRect.height || 0) / 2);
    const viewCenter = Number(flipbookSubtitleRailEl.scrollTop || 0) + (railHeight / 2);
    const driftPx = rowCenter - viewCenter;
    const nowMs = (typeof performance !== 'undefined' && performance.now)
      ? performance.now()
      : Date.now();
    const manualHold = nowMs < Number(flipbookSubtitleRailManualUntilMs || 0);
    if (!manualHold && (needRebuild || activeIdx !== flipbookSubtitleRailActiveIndex || Math.abs(driftPx) > 2)) {
      const maxTop = Math.max(0, Number(flipbookSubtitleRailEl.scrollHeight || 0) - railHeight);
      const desiredTop = _clamp(Math.round(rowCenter - (railHeight / 2)), 0, maxTop);
      if (Math.abs(Number(flipbookSubtitleRailEl.scrollTop || 0) - desiredTop) > 1) {
        flipbookSubtitleRailProgrammaticScroll = true;
        flipbookSubtitleRailEl.scrollTo({ top: desiredTop, behavior: 'auto' });
        window.requestAnimationFrame(() => {
          flipbookSubtitleRailProgrammaticScroll = false;
        });
      }
    }
  } else {
    flipbookSubtitleRailActiveIndex = -1;
  }
  flipbookSubtitleRailActiveIndex = activeIdx;
}

function renderFlipbookSubtitles(rawSeconds) {
  if (isSplitStepActive()) {
    if (flipbookSubtitlesEl) {
      flipbookSubtitlesEl.classList.add('empty');
      flipbookSubtitlesEl.innerHTML = '';
    }
    clearFlipbookSubtitleRail();
    return;
  }
  const sec = Math.max(0, Number(rawSeconds || 0));
  const around = findSubtitlesAroundSeconds(sec);
  if (flipbookSubtitlesEl) {
    if (!around.entries.length || !around.active.length) {
      flipbookSubtitlesEl.classList.add('empty');
      flipbookSubtitlesEl.innerHTML = '';
    } else {
      const activeHtml = around.active.map((row) => {
        const speaker = String(row.speaker || '').trim();
        const speakerHtml = speaker
          ? `<span class="flipbook-subtitle-speaker">${escapeHtml(speaker)}:</span>`
          : '';
        return `<div class="flipbook-subtitles-line">${speakerHtml}${escapeHtml(String(row.text || ''))}</div>`;
      }).join('');
      flipbookSubtitlesEl.classList.remove('empty');
      flipbookSubtitlesEl.innerHTML = activeHtml;
    }
  }
  renderFlipbookSubtitleRail(sec, around);
}

function syncFlipbookAudioToCurrentFrame() {
  if (!flipbookAudioEl || !sparkPlayFrames.length) return;
  const target = Math.max(0, flipbookCurrentFrameSeconds());
  const duration = Number(flipbookAudioEl.duration || chapterDurationSeconds() || 0);
  const maxSec = duration > 0 ? Math.max(0, duration - 0.01) : target;
  const seekTo = _clamp(target, 0, maxSec);
  if (Math.abs(Number(flipbookAudioEl.currentTime || 0) - seekTo) > 0.01) {
    try {
      flipbookAudioEl.currentTime = seekTo;
    } catch (_err) {}
  }
}

async function ensureFlipbookAudioLoaded(force = false) {
  if (!flipbookAudioEl || !state.archive || !state.chapter) return false;
  const key = chapterAudioCacheKey();
  const audioUrl = `/api/chapter_audio?key=${encodeURIComponent(key)}`;
  const absoluteAudioUrl = new URL(audioUrl, window.location.href).toString();
  if (!force && flipbookAudioSrcKey === key && String(flipbookAudioEl.src || '') === absoluteAudioUrl) {
    applyFlipbookVolume();
    return true;
  }
  flipbookAudioSrcKey = key;
  flipbookAudioEl.src = absoluteAudioUrl;
  flipbookAudioEl.preload = 'auto';
  try {
    flipbookAudioEl.load();
  } catch (_err) {}
  applyFlipbookVolume();
  return true;
}

function setFlipbookVisible(visible) {
  if (!flipbookPreviewEl) return;
  const on = Boolean(visible);
  flipbookPreviewEl.classList.toggle('active', on);
  flipbookPreviewEl.setAttribute('aria-hidden', on ? 'false' : 'true');
  updateFlipbookSubtitleRailMode();
  if (!on && flipbookMetaEl) {
    flipbookMetaEl.textContent = '';
  }
  if (!on) {
    if (flipbookImageEl) {
      try {
        const _ctx = flipbookImageEl.getContext('2d');
        if (_ctx) _ctx.clearRect(0, 0, flipbookImageEl.width, flipbookImageEl.height);
      } catch (_err) { flipbookImageEl.removeAttribute('src'); }
    }
    if (flipbookSubtitlesEl) {
      flipbookSubtitlesEl.classList.add('empty');
      flipbookSubtitlesEl.innerHTML = '';
    }
    if (flipbookAudioEl) {
      try {
        flipbookAudioEl.pause();
      } catch (_err) {}
    }
    clearFlipbookSubtitleRail();
    syncFrameGridPlaybackCursor(-1);
  }
  updateFlipbookControls();
}

function _collectFlipbookFramesFromGrid() {
  // Return loaded frames directly — image src, label, timecode are computed on demand
  // in renderSparkPlaybackFrame to avoid O(n) precomputation for large frame sets.
  return reviewLoadedFrames();
}

function renderSparkPlaybackFrame(index, options = {}) {
  if (!sparkPlayFrames.length || !flipbookImageEl) return;
  const keepVisible = !(options && options.keepVisible === false);
  const syncAudio = Boolean(options && options.syncAudio);
  const idx = _clamp(Number(index), 0, sparkPlayFrames.length - 1);
  const frame = sparkPlayFrames[idx];
  if (!frame) return;
  const fidKey = String((frame.fid !== undefined && frame.fid !== null) ? frame.fid : '');
  const tc = String(timelineLabelFromFid(frame.fid) || '');
  const frameSeconds = chapterLocalSecondsFromFid(frame.fid);
  // Apply freeze frame simulation: bad frames display their clean source frame instead.
  let displayFid = frame.fid;
  let isFrozen = false;
  if (state.simulateFreezeFrame && frame.status === 'bad' && state.freezeReplacementMap) {
    const sourceFid = state.freezeReplacementMap.get(fidKey);
    if (sourceFid !== undefined && sourceFid !== null) {
      displayFid = sourceFid;
      isFrozen = true;
    }
  }
  // Render frame: prefer contact-sheet sprite (already browser-cached from grid) via canvas
  // drawImage — zero network requests during playback. Fall back to URL-based img load.
  const _fIdx = _flipbookFrameGlobalIndex(displayFid);
  // Use metrics-based URL so it matches the sheets already loaded by the grid display,
  // avoiding a separate count=512 download just for the flipbook canvas.
  const _sprite = _fIdx >= 0 ? frameContactSheetSpecForIndex(_fIdx, frameGridMetrics(currentReviewFrames().length)) : null;
  const _sheetImg = _sprite
    ? (frameSheetImageObjects.get(_sprite.url) || frameSheetPrefetchPending.get(_sprite.url))
    : null;
  if (_sheetImg && _sheetImg.complete && _sheetImg.naturalWidth > 0) {
    const ctx = flipbookImageEl.getContext && flipbookImageEl.getContext('2d');
    if (ctx) {
      const sx = _sprite.col * _sprite.thumbWidth;
      const sy = _sprite.row * _sprite.thumbHeight;
      ctx.drawImage(_sheetImg, sx, sy, _sprite.thumbWidth, _sprite.thumbHeight,
        0, 0, flipbookImageEl.width, flipbookImageEl.height);
    }
  } else {
    // Secondary: any already-loaded contact sheet (e.g. the metrics-based grid sheet)
    // that happens to cover the target frame. This gives an immediate draw while the
    // preferred no-metrics sheet is still loading in the background.
    const _anySheet = _findAnyLoadedSheetForFrameIndex(_fIdx);
    if (_anySheet) {
      const ctx = flipbookImageEl.getContext && flipbookImageEl.getContext('2d');
      if (ctx) {
        const sx = _anySheet.col * _anySheet.thumbWidth;
        const sy = _anySheet.row * _anySheet.thumbHeight;
        ctx.drawImage(_anySheet.img, sx, sy, _anySheet.thumbWidth, _anySheet.thumbHeight,
          0, 0, flipbookImageEl.width, flipbookImageEl.height);
      }
    } else {
      // Fallback: individual frame URL. After the first request the browser caches it.
      const displayFidKey = String((displayFid !== undefined && displayFid !== null) ? displayFid : '');
      const src = frameImageSrcForFid(displayFidKey, isFrozen ? '' : frame.image);
      if (src) {
        const _fallbackImg = new Image();
        const _capturedIdx = idx;
        _fallbackImg.onload = () => {
          if (sparkPlayIndex !== _capturedIdx) return;
          const ctx2 = flipbookImageEl.getContext && flipbookImageEl.getContext('2d');
          if (ctx2) ctx2.drawImage(_fallbackImg, 0, 0, flipbookImageEl.width, flipbookImageEl.height);
        };
        _fallbackImg.src = src;
      }
    }
  }
  flipbookImageEl.setAttribute('aria-label', `Flipbook frame ${tc || (idx + 1)}`);
  if (flipbookMetaEl) {
    const statusTag = isFrozen ? 'FROZEN' : (frame.status === 'bad' ? 'BAD' : 'GOOD');
    flipbookMetaEl.textContent = `${idx + 1}/${sparkPlayFrames.length}  T:${tc || '?'}  ${statusTag}`;
  }
  sparkPlayIndex = idx;
  if (flipbookImageEl) {
    if (isGammaStepActive()) {
      const gamma = gammaLevelForFrameId(frame.fid);
      if (Math.abs(gamma - 1.0) < 0.001) {
        flipbookImageEl.style.removeProperty('filter');
      } else {
        flipbookImageEl.style.filter = `brightness(${gamma.toFixed(3)})`;
      }
    } else {
      flipbookImageEl.style.removeProperty('filter');
    }
  }
  renderFlipbookSubtitles(frameSeconds);
  syncFrameGridPlaybackCursor(idx, keepVisible);
  if (syncAudio) {
    syncFlipbookAudioToCurrentFrame();
  }
  updateFlipbookControls();
}

// Render a chapter frame by index to an arbitrary canvas element.
// Used by the audio sync step for its inline frame display.
function renderFrameToCanvas(frameIndex, targetCanvas) {
  if (!targetCanvas) return;
  const frames = currentReviewFrames();
  if (!frames.length) return;
  const idx = _clamp(Math.round(Number(frameIndex)), 0, frames.length - 1);
  const frame = frames[idx];
  if (!frame) return;
  const displayFid = frame.fid;
  const _fIdx = _flipbookFrameGlobalIndex(displayFid);
  const _sprite = _fIdx >= 0
    ? frameContactSheetSpecForIndex(_fIdx, frameGridMetrics(frames.length))
    : null;
  const _sheetImg = _sprite
    ? (frameSheetImageObjects.get(_sprite.url) || frameSheetPrefetchPending.get(_sprite.url))
    : null;
  const drawToCtx = (ctx, img, sx, sy, sw, sh) => {
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, targetCanvas.width, targetCanvas.height);
  };
  if (_sheetImg && _sheetImg.complete && _sheetImg.naturalWidth > 0) {
    const ctx = targetCanvas.getContext('2d');
    if (ctx) drawToCtx(ctx, _sheetImg, _sprite.col * _sprite.thumbWidth, _sprite.row * _sprite.thumbHeight, _sprite.thumbWidth, _sprite.thumbHeight);
  } else {
    const _anySheet = _findAnyLoadedSheetForFrameIndex(_fIdx);
    if (_anySheet) {
      const ctx = targetCanvas.getContext('2d');
      if (ctx) drawToCtx(ctx, _anySheet.img, _anySheet.col * _anySheet.thumbWidth, _anySheet.row * _anySheet.thumbHeight, _anySheet.thumbWidth, _anySheet.thumbHeight);
    } else {
      const src = frameImageSrcForFid(String(displayFid ?? ''), frame.image);
      if (src) {
        const img = new Image();
        img.onload = () => {
          const ctx2 = targetCanvas.getContext('2d');
          if (ctx2) ctx2.drawImage(img, 0, 0, targetCanvas.width, targetCanvas.height);
        };
        img.src = src;
      }
    }
  }
}

function syncSparkPlayFramesFromGrid() {
  if (!sparkPlayFrames.length) return;
  const current = sparkPlayFrames[sparkPlayIndex] || null;
  const currentFid = current ? String(current.fid || '') : '';
  const updated = _collectFlipbookFramesFromGrid();
  if (!updated.length) return;
  let nextIndex = _clamp(Number(sparkPlayIndex || 0), 0, updated.length - 1);
  if (currentFid) {
    const match = updated.findIndex(f => String(f.fid || '') === currentFid);
    if (match >= 0) nextIndex = match;
  }
  sparkPlayFrames = updated;
  sparkPlayIndex = nextIndex;
}

function seekFlipbookToFrameId(fid, options = {}) {
  const fidKey = String(fid || '').trim();
  if (!fidKey) return false;
  const keepVisible = !(options && options.keepVisible === false);

  if (sparkPlayFrames.length) {
    syncSparkPlayFramesFromGrid();
  } else {
    sparkPlayFrames = _collectFlipbookFramesFromGrid();
  }
  if (!sparkPlayFrames.length) return false;

  const idx = sparkPlayFrames.findIndex(f => String(f.fid || '') === fidKey);
  if (idx < 0) return false;

  sparkPlayIndex = idx;
  if (flipbookPreviewEl && flipbookPreviewEl.classList.contains('active')) {
    renderSparkPlaybackFrame(idx, { keepVisible, syncAudio: !sparkPlayTimer });
  }
  return true;
}

function ensureFlipbookReady(resetIndex = false) {
  if (!isReviewStepActive()) return;
  const panelOpen = Boolean(flipbookPreviewEl && flipbookPreviewEl.classList.contains('active'));
  const shouldSync = panelOpen || Boolean(sparkPlayTimer);
  if (!shouldSync) return;
  const collected = _collectFlipbookFramesFromGrid();
  if (!collected.length) {
    sparkPlayFrames = [];
    sparkPlayIndex = 0;
    setFlipbookVisible(false);
    return;
  }

  const current = sparkPlayFrames[sparkPlayIndex] || null;
  const currentFid = current ? String(current.fid || '') : '';
  sparkPlayFrames = collected;
  if (resetIndex) {
    sparkPlayIndex = 0;
  } else if (currentFid) {
    const match = sparkPlayFrames.findIndex(f => String(f.fid || '') === currentFid);
    if (match >= 0) {
      sparkPlayIndex = match;
    } else {
      sparkPlayIndex = _clamp(Number(sparkPlayIndex || 0), 0, sparkPlayFrames.length - 1);
    }
  } else {
    sparkPlayIndex = _clamp(Number(sparkPlayIndex || 0), 0, sparkPlayFrames.length - 1);
  }

  if (panelOpen) {
    renderSparkPlaybackFrame(sparkPlayIndex, { keepVisible: false, syncAudio: !sparkPlayTimer });
  }
  updateSparkPlayButton();
}

function _pauseSparkWindowPlayback() {
  if (sparkPlayTimer) {
    window.cancelAnimationFrame(sparkPlayTimer);
    sparkPlayTimer = null;
  }
  if (sparkPlayUseAudioClock && flipbookAudioEl) {
    const idx = flipbookIndexFromChapterSeconds(Number(flipbookAudioEl.currentTime || 0));
    renderSparkPlaybackFrame(idx, { syncAudio: false });
  }
  sparkPlayUseAudioClock = false;
  sparkPlayLastTickMs = 0;
  sparkPlayAccumulatorMs = 0;
  sparkPlayAudioClockLastTime = -1;
  sparkPlayAudioClockStallCount = 0;
  if (flipbookAudioEl) {
    try {
      flipbookAudioEl.pause();
    } catch (_err) {}
  }
  syncFlipbookAudioToCurrentFrame();
  updateSparkPlayButton();
  updateFlipbookControls();
}

function stopSparkWindowPlayback() {
  _pauseSparkWindowPlayback();
  sparkPlayFrames = [];
  sparkPlayIndex = 0;
  setFlipbookVisible(false);
  syncFrameGridPlaybackCursor(-1);
  updateSparkPlayButton();
  updateFlipbookControls();
}

function stepSparkWindowRight() {
  if (!isReviewStepActive()) {
    stopSparkWindowPlayback();
    return;
  }
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) {
    stopSparkWindowPlayback();
    return;
  }
  if (!sparkPlayFrames.length) {
    sparkPlayFrames = _collectFlipbookFramesFromGrid();
    if (!sparkPlayFrames.length) {
      stopSparkWindowPlayback();
      return;
    }
  }
  if (sparkPlayIndex >= sparkPlayFrames.length - 1) {
    sparkPlayIndex = 0;
  } else {
    sparkPlayIndex += 1;
  }
  renderSparkPlaybackFrame(sparkPlayIndex, { syncAudio: !sparkPlayTimer });
}

function stepSparkWindowLeft() {
  if (!sparkPlayFrames.length) return;
  if (sparkPlayIndex <= 0) {
    sparkPlayIndex = sparkPlayFrames.length - 1;
  } else {
    sparkPlayIndex -= 1;
  }
  renderSparkPlaybackFrame(sparkPlayIndex, { syncAudio: !sparkPlayTimer });
}

function runSparkWindowPlaybackFrameClock(timestampMs) {
  if (!sparkPlayTimer) return;
  if (!isReviewStepActive()) {
    stopSparkWindowPlayback();
    return;
  }
  if (!sparkPlayFrames.length) {
    sparkPlayFrames = _collectFlipbookFramesFromGrid();
  }
  if (!sparkPlayFrames.length) {
    stopSparkWindowPlayback();
    return;
  }

  if (sparkPlayUseAudioClock && flipbookAudioEl && !flipbookAudioEl.paused) {
    const audioTime = Math.max(0, Number(flipbookAudioEl.currentTime || 0));
    const advanced = (
      sparkPlayAudioClockLastTime < 0
      || Math.abs(audioTime - sparkPlayAudioClockLastTime) > FLIPBOOK_AUDIO_CLOCK_STALL_EPSILON
    );
    if (advanced) {
      sparkPlayAudioClockLastTime = audioTime;
      sparkPlayAudioClockStallCount = 0;
      const idx = flipbookIndexFromChapterSeconds(audioTime);
      if (idx !== sparkPlayIndex) {
        renderSparkPlaybackFrame(idx, { syncAudio: false });
      } else {
        renderFlipbookSubtitles(audioTime);
      }
    } else {
      sparkPlayAudioClockStallCount += 1;
      if (sparkPlayAudioClockStallCount >= FLIPBOOK_AUDIO_CLOCK_STALL_FRAMES) {
        sparkPlayUseAudioClock = false;
        sparkPlayLastTickMs = Number(timestampMs || 0);
        sparkPlayAccumulatorMs = 0;
        sparkPlayAudioClockLastTime = -1;
        sparkPlayAudioClockStallCount = 0;
      } else {
        renderFlipbookSubtitles(audioTime);
      }
    }
  } else {
    if (!sparkPlayLastTickMs) sparkPlayLastTickMs = timestampMs;
    const delta = Math.max(0, Number(timestampMs) - Number(sparkPlayLastTickMs));
    sparkPlayLastTickMs = Number(timestampMs);
    sparkPlayAccumulatorMs += delta;
    let stepped = false;
    while (sparkPlayAccumulatorMs >= VHS_FRAME_MS) {
      sparkPlayAccumulatorMs -= VHS_FRAME_MS;
      if (sparkPlayIndex >= sparkPlayFrames.length - 1) {
        sparkPlayIndex = 0;
      } else {
        sparkPlayIndex += 1;
      }
      stepped = true;
    }
    if (stepped) {
      renderSparkPlaybackFrame(sparkPlayIndex, { syncAudio: false });
    }
  }

  if (!sparkPlayTimer) return;
  sparkPlayTimer = window.requestAnimationFrame(runSparkWindowPlaybackFrameClock);
}

function startSparkWindowPlayback() {
  if (!sparkPlayFrames.length) {
    sparkPlayFrames = _collectFlipbookFramesFromGrid();
    _prefetchFlipbookContactSheets(sparkPlayFrames);
  }
  if (!sparkPlayFrames.length) {
    setStatus('No frame images are available for flipbook playback.', true);
    return;
  }
  sparkPlayIndex = _clamp(Number(sparkPlayIndex || 0), 0, sparkPlayFrames.length - 1);
  setFlipbookVisible(true);
  renderSparkPlaybackFrame(sparkPlayIndex, { syncAudio: true });
  sparkPlayUseAudioClock = false;
  sparkPlayLastTickMs = 0;
  sparkPlayAccumulatorMs = 0;
  sparkPlayAudioClockLastTime = -1;
  sparkPlayAudioClockStallCount = 0;
  sparkPlayTimer = window.requestAnimationFrame(runSparkWindowPlaybackFrameClock);
  ensureFlipbookAudioLoaded(false).then(async (ok) => {
    if (!ok || !sparkPlayTimer || !flipbookAudioEl) return;
    syncFlipbookAudioToCurrentFrame();
    try {
      flipbookAudioEl.playbackRate = 1.0;
      await flipbookAudioEl.play();
      sparkPlayUseAudioClock = true;
      sparkPlayAudioClockLastTime = Math.max(0, Number(flipbookAudioEl.currentTime || 0));
      sparkPlayAudioClockStallCount = 0;
    } catch (_err) {
      sparkPlayUseAudioClock = false;
      sparkPlayAudioClockLastTime = -1;
      sparkPlayAudioClockStallCount = 0;
    }
  }).catch(() => {
    sparkPlayUseAudioClock = false;
    sparkPlayAudioClockLastTime = -1;
    sparkPlayAudioClockStallCount = 0;
  });
  updateSparkPlayButton();
  updateFlipbookControls();
}

function _flipbookFrameGlobalIndex(fid) {
  const span = chapterFrameSpan();
  const startFid = Math.trunc(Number(span.start || 0));
  const frameIndex = Math.trunc(Number(fid)) - startFid;
  const frames = currentReviewFrames();
  if (frameIndex < 0 || frameIndex >= frames.length) return -1;
  return frameIndex;
}

function _prefetchFlipbookContactSheets(frames) {
  const span = chapterFrameSpan();
  const startFid = Math.trunc(Number(span.start || 0));
  const plan = frameGridContactSheetPlan();
  const chunkSize = Math.max(
    Math.max(1, Math.trunc(Number(plan.columns || FRAME_SHEET_DEFAULT_COLUMNS))),
    Math.trunc(Number(plan.count || FRAME_SHEET_DEFAULT_CHUNK_SIZE))
  );
  const seen = new Set();
  for (const frame of (frames || [])) {
    const frameIndex = Math.trunc(Number(frame.fid)) - startFid;
    if (frameIndex < 0) continue;
    const sheetStart = Math.floor(frameIndex / chunkSize) * chunkSize;
    if (seen.has(sheetStart)) continue;
    seen.add(sheetStart);
    const spec = frameContactSheetSpecForSheetStart(sheetStart);
    if (spec && spec.url) prefetchFrameContactSheet(spec.url);
  }
}

// Called by the audio sync step to pre-fetch all contact sheets for smooth playback.
function ensureAudioSyncFramesReady() {
  const frames = currentReviewFrames();
  if (frames.length) _prefetchFlipbookContactSheets(frames);
}

async function openFlipbookPanel() {
  if (!isReviewStepActive()) return;
  if (!reviewLoadedFrameCount()) {
    setStatus('Load and review frames before opening fullscreen flipbook.', true);
    return;
  }
  sparkPlayFrames = _collectFlipbookFramesFromGrid();
  if (!sparkPlayFrames.length) {
    setStatus('No frame images are available for fullscreen flipbook.', true);
    return;
  }
  _prefetchFlipbookContactSheets(sparkPlayFrames);
  sparkPlayIndex = _clamp(Number(sparkPlayIndex || 0), 0, sparkPlayFrames.length - 1);
  setFlipbookFocusMode(true);
  await ensureReviewFullscreenActive();
  setFlipbookVisible(true);
  _pauseSparkWindowPlayback();
  renderSparkPlaybackFrame(sparkPlayIndex, { syncAudio: true });
  ensureFlipbookAudioLoaded(false);
  updateSparkPlayButton();
  updateFlipbookControls();
}

function toggleSparkWindowPlayback() {
  if (sparkPlayTimer) {
    _pauseSparkWindowPlayback();
    return;
  }
  if (!isReviewStepActive()) return;
  if (!reviewLoadedFrameCount()) return;
  startSparkWindowPlayback();
}

async function closeFlipbookPanel() {
  stopSparkWindowPlayback();
  setFlipbookVisible(false);
  setFlipbookFocusMode(false);
  await exitReviewFullscreenIfActive();
  updateFullscreenButton();
}

function seekFrameGridFromSparkClientX(clientX) {
  const idx = _sparkClientXToIndex(clientX);
  if (idx === null) return;
  _scrollFrameGridToIndex(idx);
}

function queueSparkDragSeek(clientX) {
  sparkDragClientX = Number(clientX);
  if (sparkDragRaf) return;
  sparkDragRaf = window.requestAnimationFrame(() => {
    sparkDragRaf = null;
    seekFrameGridFromSparkClientX(sparkDragClientX);
  });
}

function endSparkDrag() {
  if (sparkDragRaf) {
    window.cancelAnimationFrame(sparkDragRaf);
    sparkDragRaf = null;
  }
  if (sparkDragPointerId !== null && iqrSparkEl.releasePointerCapture) {
    try {
      iqrSparkEl.releasePointerCapture(sparkDragPointerId);
    } catch (_err) {}
  }
  sparkDragPointerId = null;
  iqrSparkEl.classList.remove('dragging');
}

function scheduleAutoApplyIqr() {
  if (!state.review) return;
  if (autoIqrTimer) window.clearTimeout(autoIqrTimer);
  autoIqrTimer = window.setTimeout(() => {
    applyIqr(false);
  }, 120);
}

function updateFullscreenButton() {
  if (!fullscreenBtnEl) return;
  const active = isReviewFullscreenActive();
  fullscreenBtnEl.textContent = active ? 'Exit Fullscreen' : 'Fullscreen';
}

async function toggleReviewFullscreen() {
  if (!page2El) return;
  if (isReviewFullscreenActive()) {
    setFlipbookFocusMode(false);
    await exitReviewFullscreenIfActive();
  } else {
    await ensureReviewFullscreenActive();
  }
  updateFullscreenButton();
}

function rememberFrameImages(frames) {
  for (const frame of (frames || [])) {
    if (!frame || frame.fid === undefined || frame.fid === null) continue;
    const image = frame.image;
    if (typeof image !== 'string' || !image) continue;
    state.frameImages.set(String(frame.fid), image);
  }
}

function resetFrameSheetPrefetchState() {
  frameSheetPrefetchDone = new Set();
  frameSheetPrefetchPending.forEach((img) => {
    if (!img) return;
    try {
      img.onload = null;
      img.onerror = null;
    } catch (_err) {}
  });
  frameSheetPrefetchPending = new Map();
  frameSheetImageObjects = new Map();
  frameSheetRanges = new Map();
}

// Return sprite data from any already-loaded contact sheet that covers frameIndex,
// regardless of the sheet's column count or chunk size. Used as a fallback when the
// preferred contact sheet hasn't finished loading yet.
// frameSheetRanges is used as a parse cache; URLs not yet in the cache are parsed on first use.
function _findAnyLoadedSheetForFrameIndex(frameIndex) {
  if (!Number.isFinite(frameIndex) || frameIndex < 0) return null;
  const config = normalizeFrameSheetConfig(state.frameSheetConfig || null);
  const thumbWidth = Math.max(1, Math.trunc(Number(config.thumbWidth || FRAME_SHEET_DEFAULT_THUMB_WIDTH)));
  const thumbHeight = Math.max(1, Math.trunc(Number(config.thumbHeight || FRAME_SHEET_DEFAULT_THUMB_HEIGHT)));
  for (const [url, img] of frameSheetImageObjects) {
    if (!img || !img.complete || !img.naturalWidth) continue;
    let range = frameSheetRanges.get(url);
    if (!range) {
      try {
        const p = new URL(url, window.location.href).searchParams;
        range = {
          start: Math.trunc(Number(p.get('start') || 0)),
          count: Math.trunc(Number(p.get('count') || 0)),
          columns: Math.max(1, Math.trunc(Number(p.get('columns') || FRAME_SHEET_DEFAULT_COLUMNS))),
        };
        frameSheetRanges.set(url, range);
      } catch (_unused) { continue; }
    }
    if (frameIndex < range.start || frameIndex >= range.start + range.count) continue;
    const offset = frameIndex - range.start;
    return {
      img,
      col: offset % range.columns,
      row: Math.floor(offset / range.columns),
      thumbWidth,
      thumbHeight,
    };
  }
  return null;
}

function _frameListToRanges(frameIds) {
  const ints = Array.from(
    new Set(
      (frameIds || [])
        .map(v => Number(v))
        .filter(v => Number.isFinite(v) && v >= 0)
        .map(v => Math.trunc(v))
    )
  ).sort((a, b) => a - b);
  if (!ints.length) return [];
  const out = [];
  let start = ints[0];
  let prev = ints[0];
  for (let i = 1; i < ints.length; i += 1) {
    const f = ints[i];
    if (f === prev + 1) {
      prev = f;
      continue;
    }
    out.push([start, prev]);
    start = f;
    prev = f;
  }
  out.push([start, prev]);
  return out;
}

function _bridgeBadRanges(ranges) {
  if (!Array.isArray(ranges) || !ranges.length) return [];
  const merged = [[Number(ranges[0][0]), Number(ranges[0][1])]];
  for (let i = 1; i < ranges.length; i += 1) {
    const a = Number(ranges[i][0]);
    const b = Number(ranges[i][1]);
    const last = merged[merged.length - 1];
    const la = Number(last[0]);
    const lb = Number(last[1]);
    const gap = a - lb - 1;
    const leftLen = lb - la + 1;
    const rightLen = b - a + 1;
    const shouldBridge = (
      gap <= FREEZE_BRIDGE_ALWAYS_GAP ||
      (gap <= FREEZE_BRIDGE_SINGLETON_GAP && (leftLen === 1 || rightLen === 1))
    );
    if (shouldBridge) {
      last[1] = Math.max(lb, b);
    } else {
      merged.push([a, b]);
    }
  }
  return merged;
}

function _mergeRepairRanges(repairs) {
  if (!Array.isArray(repairs) || !repairs.length) return [];
  const sorted = repairs
    .map(item => [Number(item[0]), Number(item[1]), item[2] === null ? null : Number(item[2])])
    .sort((a, b) => (
      (a[0] - b[0]) ||
      (a[1] - b[1]) ||
      ((a[2] === null ? -1 : a[2]) - (b[2] === null ? -1 : b[2]))
    ));
  const merged = [sorted[0]];
  for (let i = 1; i < sorted.length; i += 1) {
    const [a, b, src] = sorted[i];
    const last = merged[merged.length - 1];
    if (src === last[2] && a <= last[1] + 1) {
      last[1] = Math.max(last[1], b);
    } else {
      merged.push([a, b, src]);
    }
  }
  return merged;
}

function _localBadFramesToRepairs(badFrameIds) {
  const contiguous = _frameListToRanges(badFrameIds);
  if (!contiguous.length) return [];
  const bridged = _bridgeBadRanges(contiguous);
  return _mergeRepairRanges(bridged.map(([a, b]) => [a, b, null]));
}

function _resolveFreezeRepairs(frames) {
  const rows = [];
  for (const frame of (frames || [])) {
    if (!frame) continue;
    const fidRaw = Number(frame.fid);
    if (!Number.isFinite(fidRaw)) continue;
    rows.push({
      fid: Math.trunc(fidRaw),
      status: String(frame.status || ''),
    });
  }
  if (!rows.length) return new Map();

  const badFrameIds = rows.filter(r => r.status === 'bad').map(r => r.fid);
  const baseRanges = _localBadFramesToRepairs(badFrameIds);
  if (!baseRanges.length) return new Map();

  const badSet = new Set();
  for (const [aRaw, bRaw] of baseRanges) {
    const a = Number(aRaw);
    const b = Number(bRaw);
    for (let f = a; f <= b; f += 1) {
      badSet.add(f);
    }
  }

  const sourceCandidates = Array.from(
    new Set(
      rows
        .filter(r => r.status === 'good' && frameImageSrcForFid(r.fid))
        .map(r => r.fid)
    )
  ).sort((a, b) => a - b);
  if (!sourceCandidates.length) return new Map();

  const endFrameRaw = Number(state.loadSettings && state.loadSettings.end_frame);
  let maxSourceFrame = Number.isFinite(endFrameRaw) ? Math.max(0, Math.trunc(endFrameRaw) - 1) : null;
  if (maxSourceFrame === null) {
    maxSourceFrame = Math.max(...rows.map(r => r.fid));
  }

  const clearance = Math.max(0, Number(FREEZE_SOURCE_CLEARANCE) || 0);

  function sourceIsClear(src) {
    if (badSet.has(src)) return false;
    if (clearance <= 0) return true;
    for (let f = src - clearance; f <= src + clearance; f += 1) {
      if (badSet.has(f)) return false;
    }
    return true;
  }

  function chooseSourceAfter(b, extraSkip = 0) {
    const minSource = Number(b) + 1 + Math.max(0, Number(extraSkip) || 0);
    for (const src of sourceCandidates) {
      if (src < minSource) continue;
      if (maxSourceFrame !== null && src > maxSourceFrame) return null;
      if (sourceIsClear(src)) return src;
    }
    return null;
  }

  function chooseSourceBefore(a, extraSkip = 0) {
    const maxSource = Number(a) - 1 - Math.max(0, Number(extraSkip) || 0);
    for (let i = sourceCandidates.length - 1; i >= 0; i -= 1) {
      const src = sourceCandidates[i];
      if (src > maxSource) continue;
      if (src < 0) break;
      if (sourceIsClear(src)) return src;
    }
    return null;
  }

  const resolved = [];
  for (const [aRaw, bRaw] of baseRanges) {
    const a = Number(aRaw);
    const b = Number(bRaw);
    const span = b - a + 1;
    const sourceSkip = span === 1 ? FREEZE_SINGLE_FRAME_SOURCE_SKIP : 0;
    const nextSrc = chooseSourceAfter(b, sourceSkip);
    const src = nextSrc === null ? chooseSourceBefore(a, sourceSkip) : nextSrc;
    if (src === null) continue;
    resolved.push([a, b, src]);
  }

  const replacementMap = new Map();
  const merged = _mergeRepairRanges(resolved);
  for (const [aRaw, bRaw, srcRaw] of merged) {
    const a = Number(aRaw);
    const b = Number(bRaw);
    const src = Number(srcRaw);
    for (let f = a; f <= b; f += 1) {
      replacementMap.set(String(f), src);
    }
  }
  return replacementMap;
}

function buildFrameImageUrl(fidRaw) {
  const fid = Math.trunc(Number(fidRaw));
  if (!Number.isFinite(fid) || fid < 0) return '';
  const params = new URLSearchParams({ fid: String(fid) });
  const config = normalizeFrameSheetConfig(state.frameSheetConfig || null);
  if (config.rev) params.set('rev', String(config.rev));
  return `/api/frame_image?${params.toString()}`;
}

function frameImageSrcForFid(fidRaw, fallback = '') {
  const fidKey = String(fidRaw || '').trim();
  if (!fidKey) return String(fallback || '').trim();
  const remembered = String(state.frameImages.get(fidKey) || '').trim();
  if (remembered) return remembered;
  const fallbackSrc = String(fallback || '').trim();
  if (fallbackSrc) return fallbackSrc;
  return buildFrameImageUrl(fidKey);
}

function reviewFrameIndexByFid(fidRaw) {
  const target = Math.trunc(Number(fidRaw));
  if (!Number.isFinite(target)) return -1;
  const frames = currentReviewFrames();
  let lo = 0;
  let hi = frames.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const midFid = Math.trunc(Number(frames[mid] && frames[mid].fid));
    if (!Number.isFinite(midFid)) return -1;
    if (midFid === target) return mid;
    if (midFid < target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return -1;
}

function frameGridContactSheetPlan(metricsRaw = null) {
  const config = normalizeFrameSheetConfig(state.frameSheetConfig || null);
  const maxCount = Math.max(1, Math.trunc(Number(config.chunkSize || FRAME_SHEET_DEFAULT_CHUNK_SIZE)));
  const metrics = metricsRaw && typeof metricsRaw === 'object' ? metricsRaw : null;
  const columns = Math.max(
    1,
    Math.trunc(Number((metrics && metrics.columnCount) || config.columns || FRAME_SHEET_DEFAULT_COLUMNS))
  );
  const rowStride = Math.max(
    1,
    Number((metrics && metrics.rowStride) || (FRAME_GRID_CARD_HEIGHT_PX + FRAME_GRID_GAP_PX))
  );
  const maxRows = Math.max(1, Math.floor(maxCount / columns));
  // When metrics are provided (grid display), size the sheet to the visible viewport so
  // it covers just enough rows. When called without metrics (flipbook canvas path), always
  // use the full chunk size so the URL is stable across layout/viewport changes.
  const rows = metrics ? (() => {
    const viewportHeight = Math.max(rowStride, Number((frameGridEl && frameGridEl.clientHeight) || rowStride));
    const visibleRows = Math.max(1, Math.ceil(viewportHeight / rowStride));
    const targetRows = Math.max(FRAME_GRID_CONTACT_SHEET_MIN_ROWS, visibleRows + (FRAME_GRID_OVERSCAN_ROWS * 2));
    return Math.max(1, Math.min(targetRows, maxRows));
  })() : maxRows;
  const count = Math.max(columns, Math.min(maxCount, columns * rows));
  return {
    rev: String(config.rev || ''),
    columns,
    rows,
    count,
    thumbWidth: Math.max(1, Math.trunc(Number(config.thumbWidth || FRAME_SHEET_DEFAULT_THUMB_WIDTH))),
    thumbHeight: Math.max(1, Math.trunc(Number(config.thumbHeight || FRAME_SHEET_DEFAULT_THUMB_HEIGHT))),
  };
}

function frameContactSheetSpecForIndex(indexRaw, metricsRaw = null) {
  const index = Math.trunc(Number(indexRaw));
  const frames = currentReviewFrames();
  if (!Number.isFinite(index) || index < 0 || index >= frames.length) return null;
  const plan = frameGridContactSheetPlan(metricsRaw);
  const columns = Math.max(1, Math.trunc(Number(plan.columns || FRAME_SHEET_DEFAULT_COLUMNS)));
  const chunkSize = Math.max(columns, Math.trunc(Number(plan.count || FRAME_SHEET_DEFAULT_CHUNK_SIZE)));
  const sheetStart = Math.floor(index / chunkSize) * chunkSize;
  const sheet = frameContactSheetSpecForSheetStart(sheetStart, metricsRaw);
  if (!sheet) return null;
  const offset = index - sheet.start;
  return {
    ...sheet,
    col: offset % sheet.columns,
    row: Math.floor(offset / sheet.columns),
  };
}

function frameContactSheetSpecForSheetStart(startIndexRaw, metricsRaw = null) {
  const startIndex = Math.trunc(Number(startIndexRaw));
  const frames = currentReviewFrames();
  if (!Number.isFinite(startIndex) || startIndex < 0 || startIndex >= frames.length) return null;
  const plan = frameGridContactSheetPlan(metricsRaw);
  const columns = Math.max(1, Math.trunc(Number(plan.columns || FRAME_SHEET_DEFAULT_COLUMNS)));
  const chunkSize = Math.max(columns, Math.trunc(Number(plan.count || FRAME_SHEET_DEFAULT_CHUNK_SIZE)));
  const start = Math.floor(startIndex / chunkSize) * chunkSize;
  const count = Math.max(columns, Math.min(chunkSize, frames.length - start));
  const rows = Math.max(1, Math.ceil(count / columns));
  const params = new URLSearchParams({
    start: String(start),
    count: String(count),
    columns: String(columns),
  });
  if (plan.rev) params.set('rev', String(plan.rev));
  return {
    start,
    count,
    url: `/api/frame_contact_sheet?${params.toString()}`,
    columns,
    rows,
    thumbWidth: plan.thumbWidth,
    thumbHeight: plan.thumbHeight,
  };
}

function prefetchFrameContactSheet(urlRaw) {
  const url = String(urlRaw || '').trim();
  if (!url) return;
  if (frameSheetPrefetchDone.has(url) || frameSheetPrefetchPending.has(url)) return;
  const img = new Image();
  img.decoding = 'async';
  img.loading = 'eager';
  img.onload = () => {
    frameSheetPrefetchPending.delete(url);
    frameSheetPrefetchDone.add(url);
    frameSheetImageObjects.set(url, img); // keep alive for canvas drawImage
    try {
      const p = new URL(url, window.location.href).searchParams;
      frameSheetRanges.set(url, {
        start: Math.trunc(Number(p.get('start') || 0)),
        count: Math.trunc(Number(p.get('count') || 0)),
        columns: Math.max(1, Math.trunc(Number(p.get('columns') || FRAME_SHEET_DEFAULT_COLUMNS))),
      });
    } catch (_unused) {}
  };
  img.onerror = () => {
    frameSheetPrefetchPending.delete(url);
  };
  frameSheetPrefetchPending.set(url, img);
  img.src = url;
}

function prefetchVisibleFrameSheets(metricsRaw = null, rangeRaw = null) {
  performance.mark('prefetchVisibleFrameSheets:s');
  const frames = currentReviewFrames();
  if (!frames.length) {
    performance.mark('prefetchVisibleFrameSheets:e');
    try { performance.measure('prefetchVisibleFrameSheets', 'prefetchVisibleFrameSheets:s', 'prefetchVisibleFrameSheets:e'); } catch(_) {}
    return;
  }
  const metrics = metricsRaw && typeof metricsRaw === 'object' ? metricsRaw : frameGridMetrics(frames.length);
  const range = rangeRaw && typeof rangeRaw === 'object' ? rangeRaw : _computeVisibleIndexRange();
  if (!range) return;
  const startIndex = _clamp(Number(range.start || 0), 0, frames.length - 1);
  const endIndex = _clamp(Number(range.end || 0), 0, frames.length - 1);
  // Record visible range tiles for buffer indicator
  const startSpec = frameContactSheetSpecForIndex(startIndex, metrics);
  if (startSpec) _recordImageFetchedRange(startSpec.start, startSpec.start + startSpec.count - 1);
  const current = frameContactSheetSpecForIndex(endIndex, metrics);
  if (!current) return;
  _recordImageFetchedRange(current.start, current.start + current.count - 1);
  const stride = Math.max(1, Math.trunc(Number(current.count || 1)));
  // Look-ahead prefetch (metrics-based URLs for CSS grid scroll performance and flipbook canvas)
  for (let i = 1; i <= FRAME_GRID_CONTACT_SHEET_PREFETCH_AHEAD; i += 1) {
    const nextStart = current.start + (stride * i);
    const next = frameContactSheetSpecForSheetStart(nextStart, metrics);
    if (!next || !next.url) break;
    _recordImageFetchedRange(next.start, next.start + next.count - 1);
    prefetchFrameContactSheet(next.url);
  }
  performance.mark('prefetchVisibleFrameSheets:e');
  try { performance.measure('prefetchVisibleFrameSheets', 'prefetchVisibleFrameSheets:s', 'prefetchVisibleFrameSheets:e'); } catch(_) {}
}

function spriteAxisPercent(position, total) {
  const count = Math.max(1, Math.trunc(Number(total || 1)));
  const idx = Math.max(0, Math.trunc(Number(position || 0)));
  if (count <= 1) return '0%';
  return `${((idx / (count - 1)) * 100).toFixed(4)}%`;
}

function frameThumbContainSize(boxWidthRaw, thumbWidthRaw, thumbHeightRaw) {
  const boxWidth = Math.max(1, Number(boxWidthRaw || 1));
  const boxHeight = 112;
  const thumbWidth = Math.max(1, Number(thumbWidthRaw || FRAME_SHEET_DEFAULT_THUMB_WIDTH));
  const thumbHeight = Math.max(1, Number(thumbHeightRaw || FRAME_SHEET_DEFAULT_THUMB_HEIGHT));
  const ratio = thumbWidth / thumbHeight;
  let width = boxWidth;
  let height = width / ratio;
  if (height > boxHeight) {
    height = boxHeight;
    width = height * ratio;
  }
  return {
    width: Math.max(1, Math.round(width)),
    height: Math.max(1, Math.round(height)),
  };
}

function applyFrameThumbVisual(thumbEl, display, boxWidthRaw) {
  if (!thumbEl) return;
  const sprite = display && display.sprite ? display.sprite : null;
  const size = frameThumbContainSize(
    boxWidthRaw,
    sprite ? sprite.thumbWidth : FRAME_SHEET_DEFAULT_THUMB_WIDTH,
    sprite ? sprite.thumbHeight : FRAME_SHEET_DEFAULT_THUMB_HEIGHT,
  );
  thumbEl.style.width = `${size.width}px`;
  thumbEl.style.height = `${size.height}px`;
  thumbEl.style.backgroundRepeat = 'no-repeat';
  if (sprite && sprite.url) {
    thumbEl.style.backgroundImage = `url("${sprite.url}")`;
    thumbEl.style.backgroundSize = `${sprite.columns * 100}% ${sprite.rows * 100}%`;
    thumbEl.style.backgroundPosition = `${spriteAxisPercent(sprite.col, sprite.columns)} ${spriteAxisPercent(sprite.row, sprite.rows)}`;
  } else {
    const image = String(display && display.image || '').trim();
    thumbEl.style.backgroundImage = image ? `url("${image}")` : 'none';
    thumbEl.style.backgroundSize = 'contain';
    thumbEl.style.backgroundPosition = 'center';
  }
}

function frameDisplayModel(frame, frameIndex = -1, gridMetrics = null) {
  const fidKey = String((frame && frame.fid !== undefined && frame.fid !== null) ? frame.fid : '').trim();
  const sprite = frameContactSheetSpecForIndex(frameIndex, gridMetrics);
  if (!isReviewFrameLoaded(frame)) {
    return { image: '', sprite, replaced: false, note: '' };
  }
  const originalImage = frameImageSrcForFid(fidKey, frame && frame.image);
  let image = originalImage;
  let nextSprite = sprite;
  let replaced = false;
  let note = '';
  if (state.simulateFreezeFrame && frame && frame.status === 'bad') {
    const sourceFid = state.freezeReplacementMap.get(fidKey);
    if (sourceFid !== undefined && sourceFid !== null) {
      const sourceKey = String(sourceFid);
      const sourceImage = frameImageSrcForFid(sourceKey);
      const sourceIndex = reviewFrameIndexByFid(sourceFid);
      const sourceSprite = frameContactSheetSpecForIndex(sourceIndex, gridMetrics);
      if (sourceImage) {
        image = sourceImage;
        nextSprite = sourceSprite;
        replaced = true;
        note = `FreezeFrame sim source G:${sourceKey}`;
      } else {
        note = `FreezeFrame sim source G:${sourceKey} not loaded`;
      }
    } else {
      note = 'FreezeFrame sim has no nearby clean source';
    }
  }
  return { image, sprite: nextSprite, replaced, note };
}

function renderFrameGridWindow(force = false) {
  performance.mark('renderFrameGridWindow:s');
  const frames = currentReviewFrames();
  if (!frameGridEl) {
    performance.mark('renderFrameGridWindow:e');
    try { performance.measure('renderFrameGridWindow', 'renderFrameGridWindow:s', 'renderFrameGridWindow:e'); } catch(_) {}
    return;
  }
  if (!frames.length) {
    frameGridEl.innerHTML = 'No frames loaded.';
    frameGridSizerEl = null;
    frameGridItemsEl = null;
    frameGridRenderedStart = -1;
    frameGridRenderedEnd = -1;
    frameGridRenderedLayoutKey = '';
    return;
  }

  const nodes = ensureFrameGridVirtualElements();
  if (!nodes || !nodes.sizer || !nodes.items) return;
  const metrics = frameGridMetrics(frames.length);
  const layoutKey = `${metrics.columnCount}:${metrics.itemWidth}:${frames.length}`;
  const range = frameGridVisibleIndexRange(metrics, FRAME_GRID_OVERSCAN_ROWS);
  if (!range) return;

  nodes.sizer.style.height = `${metrics.totalHeight}px`;
  if (
    !force
    && frameGridRenderedStart === range.start
    && frameGridRenderedEnd === range.end
    && frameGridRenderedLayoutKey === layoutKey
  ) {
    return;
  }

  frameGridRenderedStart = range.start;
  frameGridRenderedEnd = range.end;
  frameGridRenderedLayoutKey = layoutKey;

  const fragment = document.createDocumentFragment();
  for (let i = range.start; i <= range.end; i += 1) {
    const frame = frames[i];
    if (!frame) continue;
    const loaded = isReviewFrameLoaded(frame);
    const col = i % metrics.columnCount;
    const row = Math.floor(i / metrics.columnCount);
    const left = FRAME_GRID_PADDING_PX + (col * (metrics.itemWidth + FRAME_GRID_GAP_PX));
    const top = FRAME_GRID_PADDING_PX + (row * metrics.rowStride);
    const formattedLabel = formatFrameCardLabel(frame);
    const display = frameDisplayModel(frame, i, metrics);
    const card = document.createElement('div');
    card.className = `frame-card ${loaded ? frame.status : 'loading'}`;
    if (display.replaced) card.classList.add('replaced');
    if (i === flipbookGridCursorIndex) card.classList.add('playback-current');
    card.dataset.index = String(i);
    card.dataset.fid = String(frame.fid);
    card.dataset.loaded = loaded ? '1' : '0';
    card.dataset.label = String(formattedLabel || '');
    card.dataset.timecode = timelineLabelFromFid(frame.fid);
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
    card.style.width = `${metrics.itemWidth}px`;
    card.style.height = `${metrics.itemHeight}px`;
    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'frame-thumb-wrap';
    const thumbEl = document.createElement('div');
    thumbEl.className = 'frame-thumb-sprite';
    applyFrameThumbVisual(thumbEl, display, Math.max(1, metrics.itemWidth - 12));
    thumbWrap.appendChild(thumbEl);
    const labelEl = document.createElement('div');
    labelEl.className = 'frame-label';
    labelEl.textContent = formattedLabel;
    const noteEl = document.createElement('div');
    noteEl.className = 'frame-replace-note';
    noteEl.textContent = loaded ? '' : display.note;
    card.appendChild(thumbWrap);
    card.appendChild(labelEl);
    card.appendChild(noteEl);
    if (loaded && isGammaStepActive()) {
      const gamma = gammaLevelForFrameId(frame.fid);
      if (Math.abs(gamma - 1.0) < 0.001) {
        thumbEl.style.removeProperty('filter');
      } else {
        thumbEl.style.filter = `brightness(${gamma.toFixed(3)})`;
      }
    }
    fragment.appendChild(card);
  }
  nodes.items.replaceChildren(fragment);
  performance.mark('renderFrameGridWindow:e');
  try { performance.measure('renderFrameGridWindow', 'renderFrameGridWindow:s', 'renderFrameGridWindow:e'); } catch(_) {}
  _scheduleClientPerfUpload();
}

function refreshFreezeSimulation() {
  const frames = currentReviewFrames();
  state.freezeReplacementMap = _resolveFreezeRepairs(frames);
  renderFrameGridWindow(true);
  ensureFlipbookReady(false);
}

function clearRenderedReviewFrames() {
  const mergedFrames = currentReviewFrames();
  if (!mergedFrames.length) {
    state.freezeReplacementMap = new Map();
    frameGridEl.textContent = 'No frames loaded.';
    frameGridSizerEl = null;
    frameGridItemsEl = null;
    frameGridRenderedStart = -1;
    frameGridRenderedEnd = -1;
    frameGridRenderedLayoutKey = '';
    flipbookGridCursorIndex = -1;
    renderActiveSparkline([], 0);
    updateTimelineScrubUi();
    return true;
  }
  return false;
}

function renderReviewFrames(frames, options = {}) {
  const suppressPlaceholderMerge = Boolean(options && options.suppressPlaceholderMerge);
  const review = suppressPlaceholderMerge
    ? replaceReviewState({
      ...(state.review || {}),
      frames: frames || [],
    })
    : setReviewState(
      {
        ...(state.review || {}),
        frames: frames || [],
      },
      state.review && Array.isArray(state.review.frames) ? state.review.frames : null
    );
  const mergedFrames = currentReviewFrames();
  if (clearRenderedReviewFrames()) return;
  rememberFrameImages(reviewLoadedFrames(mergedFrames));
  flipbookGridCursorIndex = -1;
  state.freezeReplacementMap = _resolveFreezeRepairs(mergedFrames);
  renderFrameGridWindow(true);
  renderActiveSparkline(mergedFrames, review ? review.threshold : 0);
  updatePeopleStepLayoutSizing();
  scheduleVisibleRangeRefresh();
  updateTimelineScrubUi();
  ensureFlipbookReady(true);
  updateReviewStatsDisplay();
}

function applyFrameUpdates(updates) {
  rememberFrameImages(updates);
  if (Array.isArray(updates) && updates.length) {
    patchReviewFramesIntoState(updates);
  }
  renderFrameGridWindow(true);
  ensureFlipbookReady(false);
  updateTimelineScrubUi();
}

function refreshFrameCardLabelsForCurrentMode() {
  if (!frameGridEl) return;
  renderFrameGridWindow(true);
}

// --- Client performance profiling ---
// Enabled automatically when the server has VHS_PROFILE_CLIENT set.
// Collects performance.measure() entries and resource timing for contact sheet
// fetches, then POSTs a snapshot to /api/perf_report 3s after the last render.

let _clientPerfUploadTimer = null;
const _clientPerfResourceLog = [];

if (typeof PerformanceObserver !== 'undefined') {
  try {
    const _perfObs = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name && entry.name.includes('/api/frame_contact_sheet')) {
          _clientPerfResourceLog.push({
            url: entry.name,
            start: Math.round(entry.startTime * 10) / 10,
            duration: Math.round(entry.duration * 10) / 10,
            responseStart: Math.round((entry.responseStart || 0) * 10) / 10,
            transferSize: entry.transferSize || 0,
          });
        }
      }
    });
    _perfObs.observe({ type: 'resource', buffered: true });
  } catch (_) {}
}

function _scheduleClientPerfUpload() {
  if (_clientPerfUploadTimer !== null) clearTimeout(_clientPerfUploadTimer);
  _clientPerfUploadTimer = setTimeout(_uploadClientPerf, 3000);
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'P' && e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) {
    if (_clientPerfUploadTimer !== null) { clearTimeout(_clientPerfUploadTimer); _clientPerfUploadTimer = null; }
    _uploadClientPerf();
    console.log('[perf] profile snapshot uploaded');
  }
});

function _uploadClientPerf() {
  _clientPerfUploadTimer = null;
  const measures = performance.getEntriesByType('measure')
    .filter(e => e.name === 'renderFrameGridWindow' || e.name === 'prefetchVisibleFrameSheets')
    .map(e => ({
      name: e.name,
      start: Math.round(e.startTime * 10) / 10,
      duration: Math.round(e.duration * 10) / 10,
    }));
  if (!measures.length && !_clientPerfResourceLog.length) return;
  const body = {
    ts: Date.now(),
    userAgent: navigator.userAgent,
    measures,
    resources: _clientPerfResourceLog.slice(),
  };
  fetch('/api/perf_report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => {});
  // Clear so next upload only contains new data
  performance.clearMeasures('renderFrameGridWindow');
  performance.clearMeasures('prefetchVisibleFrameSheets');
  _clientPerfResourceLog.length = 0;
}
