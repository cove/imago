function setLoadCancelUi(visible, disabled = false, label = 'Cancel') {
  if (!overlayCancelWrapEl || !overlayCancelBtnEl) return;
  overlayCancelWrapEl.style.display = visible ? 'flex' : 'none';
  overlayCancelBtnEl.disabled = Boolean(disabled);
  overlayCancelBtnEl.textContent = String(label || 'Cancel');
}

function setStatus(text, isError = false) {
  const msg = String(text || '').trim();
  statusEl.textContent = msg;
  statusEl.classList.toggle('error', Boolean(isError));
  statusEl.classList.toggle('hidden', msg.length === 0);
}

function reviewHasFrameFid(fid) {
  const target = Math.trunc(Number(fid));
  if (!Number.isFinite(target)) return false;
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  for (const frame of frames) {
    const fidVal = Math.trunc(Number(frame && frame.fid));
    if (Number.isFinite(fidVal) && fidVal === target && isReviewFrameLoaded(frame)) return true;
  }
  return false;
}

function trackPendingToggleRequest(promiseLike) {
  if (!promiseLike || typeof promiseLike.then !== 'function') return Promise.resolve();
  const p = Promise.resolve(promiseLike);
  pendingToggleRequests.add(p);
  p.finally(() => {
    pendingToggleRequests.delete(p);
  });
  return p;
}

async function flushPendingToggleRequests() {
  while (pendingToggleRequests.size > 0) {
    const batch = Array.from(pendingToggleRequests);
    if (!batch.length) break;
    await Promise.allSettled(batch);
  }
}

function sortFramesByFid(frames) {
  const arr = Array.isArray(frames) ? frames.slice() : [];
  arr.sort((a, b) => {
    const fa = Number(a && a.fid);
    const fb = Number(b && b.fid);
    if (!Number.isFinite(fa) && !Number.isFinite(fb)) return 0;
    if (!Number.isFinite(fa)) return 1;
    if (!Number.isFinite(fb)) return -1;
    return fa - fb;
  });
  return arr;
}

function isReviewFrameLoaded(frame) {
  if (!frame || frame.fid === undefined || frame.fid === null) return false;
  if (frame.loading === true || frame.placeholder === true) return false;
  const status = String(frame.status || '').trim().toLowerCase();
  return status === 'good' || status === 'bad';
}

function _recordImageFetchedRange(startRaw, endRaw) {
  const s = Math.max(0, Math.trunc(Number(startRaw)));
  const e = Math.max(s, Math.trunc(Number(endRaw)));
  if (imageFetchedRanges.some((r) => r.start <= s && r.end >= e)) return;
  imageFetchedRanges.push({ start: s, end: e });
  imageFetchedRanges.sort((a, b) => a.start - b.start);
  const merged = [];
  for (const r of imageFetchedRanges) {
    if (merged.length && r.start <= merged[merged.length - 1].end + 1) {
      merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, r.end);
    } else {
      merged.push({ start: r.start, end: r.end });
    }
  }
  imageFetchedRanges = merged;
  imageFetchVersion += 1;
}

function _isImageFetchedIndex(index) {
  let lo = 0;
  let hi = imageFetchedRanges.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const r = imageFetchedRanges[mid];
    if (index < r.start) { hi = mid - 1; }
    else if (index > r.end) { lo = mid + 1; }
    else { return true; }
  }
  return false;
}

function buildReviewPlaceholderFrame(fidRaw) {
  const fid = Math.max(0, Math.trunc(Number(fidRaw || 0)));
  return {
    fid,
    status: 'loading',
    source: '',
    score: null,
    loading: true,
    placeholder: true,
  };
}

function reviewLoadedFrames(framesRaw = null) {
  const frames = Array.isArray(framesRaw)
    ? framesRaw
    : ((state.review && Array.isArray(state.review.frames)) ? state.review.frames : []);
  const out = [];
  for (const frame of frames) {
    if (isReviewFrameLoaded(frame)) out.push(frame);
  }
  return out;
}

function reviewLoadedFrameCount(framesRaw = null) {
  if (!framesRaw && state.review && Number.isFinite(Number(state.review.loadedCount))) {
    return Math.max(0, Math.trunc(Number(state.review.loadedCount)));
  }
  return reviewLoadedFrames(framesRaw).length;
}

function mergeReviewFramesIntoChapterSpan(loadedFramesRaw, existingFramesRaw = null) {
  const loadedFrames = sortFramesByFid(reviewLoadedFrames(loadedFramesRaw));
  const span = chapterFrameSpan();
  const start = Math.max(0, Math.trunc(Number(span.start || 0)));
  const end = Math.max(start, Math.trunc(Number(span.end || start)));
  const totalCount = Math.max(0, end - start);
  if (totalCount <= 0) {
    return {
      frames: loadedFrames,
      loadedCount: loadedFrames.length,
      totalCount: loadedFrames.length,
    };
  }

  // Fast path: if all frames are loaded, skip building a 65k placeholder array.
  if (loadedFrames.length === totalCount) {
    return { frames: loadedFrames, loadedCount: totalCount, totalCount };
  }

  const existingFrames = Array.isArray(existingFramesRaw) ? existingFramesRaw : null;
  const frames = (existingFrames && existingFrames.length === totalCount)
    ? existingFrames.slice()
    : Array.from({ length: totalCount }, (_unused, index) => buildReviewPlaceholderFrame(start + index));

  for (let index = 0; index < totalCount; index += 1) {
    const current = frames[index];
    if (!current || !Number.isFinite(Number(current.fid))) {
      frames[index] = buildReviewPlaceholderFrame(start + index);
      continue;
    }
    if (!isReviewFrameLoaded(current)) {
      frames[index] = buildReviewPlaceholderFrame(start + index);
    }
  }

  for (const frame of loadedFrames) {
    const fid = Math.trunc(Number(frame && frame.fid));
    if (!Number.isFinite(fid)) continue;
    const index = fid - start;
    if (index < 0 || index >= totalCount) continue;
    frames[index] = {
      ...frame,
      loading: false,
      placeholder: false,
    };
  }

  return {
    frames,
    loadedCount: loadedFrames.length,
    totalCount,
  };
}

function normalizeReviewState(reviewRaw, existingFramesRaw = null) {
  const review = reviewRaw && typeof reviewRaw === 'object' ? reviewRaw : {};
  const merged = mergeReviewFramesIntoChapterSpan(review.frames || [], existingFramesRaw);
  return {
    ...review,
    frames: merged.frames,
    loadedCount: merged.loadedCount,
    totalCount: merged.totalCount,
  };
}

function setReviewState(reviewRaw, existingFramesRaw = null) {
  const normalized = normalizeReviewState(reviewRaw, existingFramesRaw);
  state.review = {
    ...(state.review || {}),
    ...normalized,
    frames: normalized.frames,
    loadedCount: normalized.loadedCount,
    totalCount: normalized.totalCount,
  };
  invalidateReviewSparklineCache();
  return state.review;
}

function replaceReviewState(reviewRaw) {
  const review = reviewRaw && typeof reviewRaw === 'object' ? reviewRaw : {};
  const frames = sortFramesByFid(Array.isArray(review.frames) ? review.frames : []);
  state.review = {
    ...(state.review || {}),
    ...review,
    frames,
    loadedCount: reviewLoadedFrameCount(frames),
    totalCount: frames.length,
  };
  invalidateReviewSparklineCache();
  return state.review;
}

function patchReviewFramesIntoState(updatesRaw) {
  const updates = sortFramesByFid(updatesRaw);
  const span = chapterFrameSpan();
  const start = Math.max(0, Math.trunc(Number(span.start || 0)));
  const end = Math.max(start, Math.trunc(Number(span.end || start)));
  const totalCount = Math.max(0, end - start);
  let frames = (state.review && Array.isArray(state.review.frames))
    ? state.review.frames.slice()
    : [];
  if (!frames.length && totalCount > 0) {
    frames = Array.from({ length: totalCount }, (_unused, index) => buildReviewPlaceholderFrame(start + index));
  }
  if (totalCount > 0 && frames.length !== totalCount) {
    frames = mergeReviewFramesIntoChapterSpan(reviewLoadedFrames(frames), frames).frames;
  }
  if (!frames.length && !updates.length) {
    return setReviewState({ ...(state.review || {}), frames: [] }, null);
  }
  if (!frames.length) {
    frames = updates.slice();
  }
  for (const update of updates) {
    const fid = Math.trunc(Number(update && update.fid));
    if (!Number.isFinite(fid)) continue;
    const index = totalCount > 0 ? (fid - start) : frames.findIndex((frame) => Number(frame && frame.fid) === fid);
    if (!Number.isFinite(index) || index < 0) continue;
    if (index >= frames.length) continue;
    frames[index] = {
      ...update,
      loading: false,
      placeholder: false,
    };
  }
  return setReviewState(
    {
      ...(state.review || {}),
      frames,
      loadedCount: reviewLoadedFrameCount(frames),
      totalCount: frames.length,
    },
    frames,
  );
}

function normalizeFrameSheetConfig(rawConfig) {
  const raw = rawConfig && typeof rawConfig === 'object' ? rawConfig : {};
  const chunkSize = Math.max(1, Math.trunc(Number(raw.chunk_size || raw.chunkSize || FRAME_SHEET_DEFAULT_CHUNK_SIZE)));
  const columns = Math.max(1, Math.trunc(Number(raw.columns || FRAME_SHEET_DEFAULT_COLUMNS)));
  const thumbWidth = Math.max(1, Math.trunc(Number(raw.thumb_width || raw.thumbWidth || FRAME_SHEET_DEFAULT_THUMB_WIDTH)));
  const thumbHeight = Math.max(1, Math.trunc(Number(raw.thumb_height || raw.thumbHeight || FRAME_SHEET_DEFAULT_THUMB_HEIGHT)));
  return {
    rev: String(raw.rev || ''),
    chunkSize,
    columns,
    thumbWidth,
    thumbHeight,
  };
}

function setLoading(active, message = 'Working...') {
  overlayEl.classList.toggle('active', Boolean(active));
  overlayMsgEl.textContent = message;
  if (!active) {
    if (overlayProgressFillEl) overlayProgressFillEl.style.width = '0%';
    if (overlayProgressTextEl) overlayProgressTextEl.textContent = '';
    if (overlayEtaTextEl) overlayEtaTextEl.textContent = '';
    setLoadCancelUi(false);
  }
}

function openHelpModal() {
  if (!helpModalEl) return;
  helpModalEl.classList.add('active');
  helpModalEl.setAttribute('aria-hidden', 'false');
  if (helpCloseBtnEl && typeof helpCloseBtnEl.focus === 'function') {
    helpCloseBtnEl.focus();
  }
}

function closeHelpModal() {
  if (!helpModalEl) return;
  helpModalEl.classList.remove('active');
  helpModalEl.setAttribute('aria-hidden', 'true');
  if (helpBtnEl && typeof helpBtnEl.focus === 'function') {
    helpBtnEl.focus();
  }
}

function formatClockTime(tsMs) {
  const d = new Date(Number(tsMs));
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function formatDuration(ms) {
  const total = Math.max(0, Math.round(Number(ms || 0) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function renderReadyAtFromSamples(sampleDone, sampleTotal) {
  if (!overlayEtaTextEl) return;
  const done = Math.max(0, Number(sampleDone || 0));
  const total = Math.max(0, Number(sampleTotal || 0));
  const now = Date.now();
  const elapsedMs = overlayProgressStartedAt > 0 ? Math.max(0, now - overlayProgressStartedAt) : 0;
  const etText = elapsedMs > 0 ? `ET ${formatDuration(elapsedMs)}` : 'ET 00:00';
  if (total <= 0) {
    overlayEtaTextEl.textContent = etText;
    return;
  }
  if (done >= total) {
    overlayEtaTextEl.textContent = `${etText} | ETA ${formatClockTime(now)}`;
    return;
  }
  if (done <= 0) {
    overlayEtaTextEl.textContent = `${etText} | Estimating ETA...`;
    return;
  }
  if (!loadSampleTimingStartAt) {
    loadSampleTimingStartAt = now;
    loadSampleTimingStartDone = done;
  }
  const observedFrames = Math.max(0, done - loadSampleTimingStartDone);
  if (observedFrames < 3) {
    overlayEtaTextEl.textContent = `${etText} | Estimating ETA... (${observedFrames}/3 sample)`;
    return;
  }
  const observedElapsedMs = now - loadSampleTimingStartAt;
  if (!(observedElapsedMs > 0)) {
    overlayEtaTextEl.textContent = etText;
    return;
  }
  const framesPerSec = observedFrames / (observedElapsedMs / 1000);
  if (!(framesPerSec > 0)) {
    overlayEtaTextEl.textContent = `${etText} | Estimating ETA...`;
    return;
  }
  const remaining = Math.max(0, total - done);
  const etaMs = now + (remaining / framesPerSec) * 1000;
  overlayEtaTextEl.textContent = `${etText} | ETA ${formatClockTime(etaMs)}`;
}

function setOverlayProgress(value) {
  const p = Math.max(0, Math.min(100, Number(value) || 0));
  if (overlayProgressFillEl) overlayProgressFillEl.style.width = `${p.toFixed(1)}%`;
  if (overlayProgressTextEl) overlayProgressTextEl.textContent = `${Math.round(p)}%`;
}

function maybeDismissLoadOverlayForProgressiveFrames(frameCountRaw) {
  const frameCount = Math.max(0, Math.trunc(Number(frameCountRaw || 0)));
  if (!isChapterLoadInFlight) return;
  if (progressiveLoadOverlayDismissed) return;
  if (frameCount <= 0) return;
  progressiveLoadOverlayDismissed = true;
  setLoading(false);
  updateReviewStatsDisplay();
}

function clearLoadProgressPollTimer() {
  if (!loadProgressPollTimer) return;
  window.clearTimeout(loadProgressPollTimer);
  loadProgressPollTimer = null;
}

function scheduleLoadProgressPoll(delayMs = LOAD_PROGRESS_POLL_MS) {
  if (!isChapterLoadInFlight) return;
  clearLoadProgressPollTimer();
  loadProgressPollTimer = window.setTimeout(() => {
    pollLoadProgressOnce();
  }, Math.max(0, Math.trunc(Number(delayMs) || 0)));
}

async function pollLoadProgressOnce() {
  if (!isChapterLoadInFlight || loadProgressPollInFlight) return;
  loadProgressPollInFlight = true;
  try {
    const p = await api('/api/load_progress');
    const done = Number(p.sample_done || 0);
    const total = Number(p.sample_total || 0);
    const msg = String(p.message || '');
    loadProgressDone = Math.max(0, Math.trunc(done));
    loadProgressTotal = Math.max(0, Math.trunc(total));
    loadProgressMessage = msg;
    if (msg) {
      if (total > 0) {
        overlayMsgEl.textContent = `${msg} (${done}/${total})`;
      } else {
        overlayMsgEl.textContent = msg;
      }
    }
    setOverlayProgress(Number(p.progress || 0));
    renderReadyAtFromSamples(done, total);
    updateReviewStatsDisplay();

    if (p.people_profile && state.peopleProfile && state.peopleProfile.source === 'default') {
      state.peopleProfile = normalizePeopleProfile(p.people_profile);
      refreshPeopleEditorFromState();
    }
    if (p.subtitles_profile && state.subtitlesProfile && state.subtitlesProfile.source === 'default') {
      state.subtitlesProfile = normalizeSubtitlesProfile(p.subtitles_profile);
      refreshSubtitlesEditorFromState();
    }

    const now = Date.now();
    const sampleDelta = done - lastLoadReviewSampleDone;
    const shouldRefreshLiveReview = done > 0 && (
      progressiveReviewFrameCount <= 0 ||
      done >= total ||
      lastLoadReviewSampleDone < 0 ||
      sampleDelta >= LOAD_REVIEW_POLL_MIN_FRAME_DELTA ||
      (now - lastLoadReviewPollAt) >= LOAD_REVIEW_POLL_MIN_MS
    );
    if (shouldRefreshLiveReview) {
      const live = await api('/api/load_review');
      lastLoadReviewPollAt = now;
      lastLoadReviewSampleDone = done;
      const liveReview = live.review || null;
      const liveFrames = sortFramesByFid((liveReview && Array.isArray(liveReview.frames)) ? liveReview.frames : []);
      if (liveFrames.length > 0) {
        state.frameSheetConfig = normalizeFrameSheetConfig(live.contact_sheet || state.frameSheetConfig);
        setReviewState(
          {
            ...(state.review || {}),
            threshold: Number(liveReview.threshold || 0),
            stats: liveReview.stats || {},
            frames: liveFrames,
          },
          state.review && Array.isArray(state.review.frames) ? state.review.frames : null
        );
        syncForceAllFramesGoodFromReview(liveReview);
        updateReviewStatsDisplay();
        if (liveFrames.length !== progressiveReviewFrameCount) {
          renderReviewFrames(liveFrames);
          progressiveReviewFrameCount = liveFrames.length;
          maybeDismissLoadOverlayForProgressiveFrames(liveFrames.length);
        }
      }
    }
  } catch (_err) {
    // Ignore transient polling errors while main request is in-flight.
  } finally {
    loadProgressPollInFlight = false;
    if (isChapterLoadInFlight) {
      scheduleLoadProgressPoll(LOAD_PROGRESS_POLL_MS);
    }
  }
}

function setSimFreezeFrame(enabled) {
  const next = isGammaStepActive() ? true : Boolean(enabled);
  state.simulateFreezeFrame = next;
  if (simulateFreezeFrameEl) {
    simulateFreezeFrameEl.checked = state.simulateFreezeFrame;
  }
  if (flipbookSimFreezeFrameEl) {
    flipbookSimFreezeFrameEl.checked = state.simulateFreezeFrame;
  }
  refreshFreezeSimulation();
}

function setForceAllFramesGoodUi(enabled) {
  const next = Boolean(enabled);
  state.forceAllFramesGood = next;
  if (forceAllFramesGoodEl) {
    forceAllFramesGoodEl.checked = next;
  }
}

function syncForceAllFramesGoodFromReview(review) {
  if (!review || typeof review !== 'object') return;
  if (!Object.prototype.hasOwnProperty.call(review, 'force_all_frames_good')) return;
  setForceAllFramesGoodUi(Boolean(review.force_all_frames_good));
}

function updateActionLocks() {
  const locked = Boolean(isChapterLoadInFlight || isPreviewRenderInFlight || isSubtitlesGenerateInFlight);
  const forceGoodLocked = Boolean(isPreviewRenderInFlight || isSubtitlesGenerateInFlight);
  navActionButtons.forEach((el) => {
    if (el) el.disabled = locked;
  });
  lockExtraButtons.forEach((el) => {
    if (!el) return;
    if (el === forceAllFramesGoodEl) {
      el.disabled = forceGoodLocked;
      return;
    }
    el.disabled = locked;
  });
  if (stepPillsEl) {
    stepPillsEl.querySelectorAll('.step-pill').forEach((el) => {
      const step = stepDef(Number(el.dataset.step || 0));
      el.disabled = isStepPillDisabled(step);
    });
  }
}

function startLoadProgress(message) {
  clearLoadProgressPollTimer();
  setLoading(true, message || 'Loading chapter...');
  if (loadSpinnerEl) loadSpinnerEl.classList.add('active');
  isChapterLoadInFlight = true;
  loadProgressPollInFlight = false;
  updateActionLocks();
  loadSampleTimingStartAt = 0;
  loadSampleTimingStartDone = 0;
  overlayProgressStartedAt = Date.now();
  activeCancelableTask = 'load';
  setLoadCancelUi(true, false, 'Cancel');
  setOverlayProgress(1);
  renderReadyAtFromSamples(0, 0);
  progressiveReviewFrameCount = 0;
  progressiveLoadOverlayDismissed = false;
  imageFetchedRanges = [];
  imageFetchVersion = 0;
  loadProgressMessage = '';
  loadProgressDone = 0;
  loadProgressTotal = 0;
  lastLoadReviewPollAt = 0;
  lastLoadReviewSampleDone = -1;
  pollLoadProgressOnce();
}

function finishLoadProgress(success = true) {
  clearLoadProgressPollTimer();
  if (loadSpinnerEl) loadSpinnerEl.classList.remove('active');
  setLoadCancelUi(false);
  if (success) {
    setOverlayProgress(100);
    renderReadyAtFromSamples(1, 1);
    window.setTimeout(() => {
      setLoading(false);
    }, 180);
  } else {
    setLoading(false);
  }
  loadSampleTimingStartAt = 0;
  loadSampleTimingStartDone = 0;
  overlayProgressStartedAt = 0;
  activeCancelableTask = '';
  progressiveReviewFrameCount = 0;
  progressiveLoadOverlayDismissed = false;
  imageFetchedRanges = [];
  imageFetchVersion = 0;
  loadProgressMessage = '';
  loadProgressDone = 0;
  loadProgressTotal = 0;
  isChapterLoadInFlight = false;
  loadProgressPollInFlight = false;
  lastLoadReviewPollAt = 0;
  lastLoadReviewSampleDone = -1;
  updateReviewStatsDisplay();
  updateActionLocks();
}

async function api(path, method = 'GET', body = null, timeoutMs = 30000) {
  const timeoutValue = Number(timeoutMs);
  const useTimeout = Number.isFinite(timeoutValue) && timeoutValue > 0;
  const effectiveTimeout = useTimeout ? Math.max(1000, timeoutValue) : 0;
  const controller = useTimeout ? new AbortController() : null;
  const timer = useTimeout ? window.setTimeout(() => {
    try {
      if (controller) controller.abort();
    } catch (_err) {}
  }, effectiveTimeout) : null;
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  if (controller) {
    options.signal = controller.signal;
  }
  let res;
  try {
    res = await fetch(path, options);
  } catch (err) {
    if (err && err.name === 'AbortError') {
      throw new Error(`Request timed out after ${Math.round(effectiveTimeout / 1000)}s: ${method} ${path}`);
    }
    throw err;
  } finally {
    if (timer) {
      window.clearTimeout(timer);
    }
  }

  let payload = null;
  try {
    payload = await res.json();
  } catch (_err) {
    throw new Error(`Invalid server response for ${method} ${path} (HTTP ${res.status}).`);
  }
  if (!res.ok || !payload || !payload.ok) {
    throw new Error((payload && payload.error) || `HTTP ${res.status}`);
  }
  return payload;
}

async function pollPreviewProgressOnce() {
  try {
    const p = await api('/api/preview_progress');
    const done = Number(p.frame_done || 0);
    const total = Number(p.frame_total || 0);
    const msg = String(p.message || '');
    if (msg) {
      if (total > 0) {
        overlayMsgEl.textContent = `${msg} (${done}/${total})`;
      } else {
        overlayMsgEl.textContent = msg;
      }
    }
    setOverlayProgress(Number(p.progress || 0));
    renderReadyAtFromSamples(done, total);
  } catch (_err) {
    // Ignore transient polling errors while main request is in-flight.
  }
}

function startPreviewProgress(message) {
  if (previewProgressPollTimer) {
    window.clearInterval(previewProgressPollTimer);
    previewProgressPollTimer = null;
  }
  setLoading(true, message || 'Rendering chapter preview...');
  isPreviewRenderInFlight = true;
  updateActionLocks();
  loadSampleTimingStartAt = 0;
  loadSampleTimingStartDone = 0;
  overlayProgressStartedAt = Date.now();
  activeCancelableTask = '';
  setOverlayProgress(1);
  renderReadyAtFromSamples(0, 0);
  pollPreviewProgressOnce();
  previewProgressPollTimer = window.setInterval(pollPreviewProgressOnce, 220);
}

function finishPreviewProgress(success = true) {
  if (previewProgressPollTimer) {
    window.clearInterval(previewProgressPollTimer);
    previewProgressPollTimer = null;
  }
  if (success) {
    setOverlayProgress(100);
    renderReadyAtFromSamples(1, 1);
    window.setTimeout(() => {
      setLoading(false);
    }, 180);
  } else {
    setLoading(false);
  }
  loadSampleTimingStartAt = 0;
  loadSampleTimingStartDone = 0;
  overlayProgressStartedAt = 0;
  activeCancelableTask = '';
  isPreviewRenderInFlight = false;
  updateActionLocks();
}

async function pollSubtitlesProgressOnce() {
  try {
    const p = await api('/api/subtitles_progress');
    const done = Number(p.segment_done || 0);
    const total = Number(p.segment_total || 0);
    const msg = String(p.message || '');
    if (msg) {
      if (total > 0) {
        overlayMsgEl.textContent = `${msg} (${done}/${total})`;
      } else {
        overlayMsgEl.textContent = msg;
      }
    }
    setOverlayProgress(Number(p.progress || 0));
    renderReadyAtFromSamples(done, total);
  } catch (_err) {
    // Ignore transient polling errors while main request is in-flight.
  }
}

function startSubtitlesProgress(message) {
  if (subtitlesProgressPollTimer) {
    window.clearInterval(subtitlesProgressPollTimer);
    subtitlesProgressPollTimer = null;
  }
  setLoading(true, message || 'Generating dialogue subtitles with Whisper...');
  isSubtitlesGenerateInFlight = true;
  updateActionLocks();
  loadSampleTimingStartAt = 0;
  loadSampleTimingStartDone = 0;
  overlayProgressStartedAt = Date.now();
  activeCancelableTask = 'subtitles';
  setLoadCancelUi(true, false, 'Cancel');
  setOverlayProgress(1);
  renderReadyAtFromSamples(0, 0);
  pollSubtitlesProgressOnce();
  subtitlesProgressPollTimer = window.setInterval(pollSubtitlesProgressOnce, 220);
}

function finishSubtitlesProgress(success = true) {
  if (subtitlesProgressPollTimer) {
    window.clearInterval(subtitlesProgressPollTimer);
    subtitlesProgressPollTimer = null;
  }
  if (success) {
    setOverlayProgress(100);
    renderReadyAtFromSamples(1, 1);
    window.setTimeout(() => {
      setLoading(false);
    }, 180);
  } else {
    setLoading(false);
  }
  loadSampleTimingStartAt = 0;
  loadSampleTimingStartDone = 0;
  overlayProgressStartedAt = 0;
  activeCancelableTask = '';
  setLoadCancelUi(false);
  isSubtitlesGenerateInFlight = false;
  updateActionLocks();
}

function stepDef(rawStep) {
  const n = Math.trunc(Number(rawStep));
  return STEP_BY_NUM.get(n) || STEP_FIRST;
}

function stepNumForMode(mode, fallbackStep = STEP_FIRST.num) {
  const key = String(mode || '').trim();
  const def = STEP_MODE_TO_FIRST.get(key);
  return def ? def.num : stepDef(fallbackStep).num;
}

function stepIndex(rawStep) {
  const def = stepDef(rawStep);
  const idx = Number(STEP_INDEX_BY_NUM.get(def.num));
  if (Number.isFinite(idx) && idx >= 0) return idx;
  return 0;
}

function setStepByMode(mode, fallbackStep = STEP_FIRST.num) {
  setStep(stepNumForMode(mode, fallbackStep));
}

function isReviewMode(mode) {
  return mode === 'review' || mode === 'audio_sync' || mode === 'gamma' || mode === 'people' || mode === 'subtitles' || mode === 'split';
}

function isNavigationLocked() {
  return Boolean(isPreviewRenderInFlight || isSubtitlesGenerateInFlight);
}

function isStepPillDisabled(step) {
  const target = stepDef(step);
  if (isNavigationLocked()) return true;
  if (!isChapterLoadInFlight) return false;
  return target.mode === 'gamma' || target.mode === 'summary';
}

function renderStepPills() {
  if (!stepPillsEl) return;
  stepPillsEl.innerHTML = '';
  const activeStepNum = stepDef(state.wizardStep).num;
  const visibleSteps = STEP_DEFS.filter((step) => step.mode !== 'load');
  visibleSteps.forEach((step, index) => {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.id = `stepPill${step.num}`;
    pill.className = 'step-pill';
    pill.dataset.step = String(step.num);
    pill.textContent = `${index + 1}/${visibleSteps.length} ${step.label}`;
    pill.disabled = isStepPillDisabled(step);
    if (step.num === activeStepNum) {
      pill.classList.add('active');
    }
    stepPillsEl.appendChild(pill);
  });
}

function updateStepPillActive(stepNum) {
  if (!stepPillsEl || !stepPillsEl.children.length) {
    renderStepPills();
    return;
  }
  const target = Number(stepNum);
  stepPillsEl.querySelectorAll('.step-pill').forEach((el) => {
    const n = Number(el.dataset.step || 0);
    el.classList.toggle('active', n === target);
  });
}

function setStep(step) {
  const previousStep = stepDef(state.wizardStep);
  const previousGammaMode = previousStep.mode === 'gamma';
  const targetStep = stepDef(step);
  state.wizardStep = targetStep.num;
  const reviewStage = isReviewMode(targetStep.mode);
  if (!reviewStage) {
    setFlipbookFocusMode(false);
    stopSparkWindowPlayback();
  }
  if (!reviewStage && isReviewFullscreenActive()) {
    exitReviewFullscreenIfActive();
  }
  STEP_PAGE_IDS.forEach((id) => {
    const page = document.getElementById(id);
    if (!page) return;
    page.classList.toggle('active', id === targetStep.pageId);
  });
  updateStepPillActive(targetStep.num);
  const isLoadStep = targetStep.mode === 'load';
  if (stepPillsEl) stepPillsEl.classList.toggle('hidden-ui', isLoadStep);
  if (backToChaptersBtnEl) backToChaptersBtnEl.classList.toggle('hidden-ui', isLoadStep);
  const audioSyncMode = targetStep.mode === 'audio_sync';
  const gammaMode = targetStep.mode === 'gamma';
  const peopleMode = targetStep.mode === 'people';
  const subtitlesMode = targetStep.mode === 'subtitles';
  const splitMode = targetStep.mode === 'split';
  const timelineMode = peopleMode || subtitlesMode || splitMode;
  if (!timelineMode) {
    peopleTimelineDraft = null;
    subtitleTimelineDraft = null;
    peopleTimelineDrag = null;
  }
  // Hide audio sync step if leaving it
  if (!audioSyncMode && typeof hideAudioSyncStep === 'function') hideAudioSyncStep();
  const audioSyncControlsEl2 = document.getElementById('audioSyncControls');
  const audioSyncPanelEl2 = document.getElementById('audioSyncPanel');
  if (audioSyncControlsEl2) audioSyncControlsEl2.classList.toggle('hidden-ui', !audioSyncMode);
  if (audioSyncPanelEl2) audioSyncPanelEl2.classList.toggle('hidden-ui', !audioSyncMode);
  if (reviewControlsEl) reviewControlsEl.classList.toggle('hidden-ui', audioSyncMode || gammaMode || timelineMode);
  if (gammaControlsEl) gammaControlsEl.classList.toggle('hidden-ui', !gammaMode);
  if (peopleControlsEl) peopleControlsEl.classList.toggle('hidden-ui', !peopleMode);
  if (subtitlesControlsEl) subtitlesControlsEl.classList.toggle('hidden-ui', !subtitlesMode);
  if (iqrSparkEl) iqrSparkEl.classList.toggle('hidden-ui', audioSyncMode || timelineMode);
  if (sparkMetaEl) sparkMetaEl.classList.toggle('hidden-ui', audioSyncMode || timelineMode);
  if (timelineScrubWrapEl) timelineScrubWrapEl.classList.toggle('hidden-ui', audioSyncMode || !timelineMode);
  if (peopleTimelineEl) peopleTimelineEl.classList.add('hidden-ui');
  if (gammaRangeMetaEl) gammaRangeMetaEl.classList.toggle('hidden-ui', !gammaMode);
  if (peopleMetaEl) peopleMetaEl.classList.toggle('hidden-ui', !peopleMode);
  if (peopleEditorEl) peopleEditorEl.classList.toggle('hidden-ui', !peopleMode);
  if (subtitlesMetaEl) subtitlesMetaEl.classList.toggle('hidden-ui', !subtitlesMode);
  if (subtitlesEditorEl) subtitlesEditorEl.classList.toggle('hidden-ui', !subtitlesMode);
  if (splitEditorEl) splitEditorEl.classList.toggle('hidden-ui', !splitMode);
  if (page2El) page2El.classList.toggle('bad-frame-step', targetStep.mode === 'review');
  if (page2El) page2El.classList.toggle('audio-sync-step', audioSyncMode);
  if (flipbookGammaControlsEl) flipbookGammaControlsEl.classList.toggle('hidden-ui', !gammaMode);
  if (flipbookSimFreezeWrapEl) flipbookSimFreezeWrapEl.classList.toggle('hidden-ui', gammaMode);

  if (gammaMode && !previousGammaMode) {
    state.simulateFreezeFrameReviewPref = Boolean(state.simulateFreezeFrame);
    setSimFreezeFrame(true);
  } else if (!gammaMode && previousGammaMode) {
    setSimFreezeFrame(Boolean(state.simulateFreezeFrameReviewPref));
  }

  if (gammaMode) {
    updateGammaControls();
    updateGammaRangeMeta();
  }
  if (peopleMode) {
    updatePeopleMeta();
    updateSubtitlesMeta();
    ensureTimelineAudioLoaded(false);
  } else if (splitMode) {
    updateSplitMeta();
    ensureTimelineAudioLoaded(false);
  } else if (timelineAudioEl && !timelineAudioEl.paused) {
    timelineAudioEl.pause();
  }
  updateFlipbookSubtitleRailMode();
  updatePeopleStepLayoutSizing();
  updateTimelineAudioPlayButton();
  updateFullscreenButton();
  refreshGammaVisuals();
  refreshFrameCardLabelsForCurrentMode();
  updateReviewStatsDisplay();
  if (reviewStage && state.review && Array.isArray(state.review.frames) && state.review.frames.length) {
    window.requestAnimationFrame(() => {
      updatePeopleStepLayoutSizing();
      scheduleVisibleRangeRefresh();
      ensureFlipbookReady(false);
    });
  }
}

function chapterByTitle(title) {
  return state.chapters.find(c => c.title === title) || null;
}

function isRequestedChapterAlreadyLoaded(archiveRaw = state.archive, chapterRaw = state.chapter) {
  const requestedArchive = String(archiveRaw || '').trim();
  const requestedChapter = String(chapterRaw || '').trim();
  const loadedArchive = String((state.loadSettings && state.loadSettings.archive) || '').trim();
  const loadedChapter = String((state.loadSettings && state.loadSettings.chapter) || '').trim();
  return Boolean(
    requestedArchive
    && requestedChapter
    && loadedArchive
    && loadedChapter
    && requestedArchive === loadedArchive
    && requestedChapter === loadedChapter
  );
}

function chapterBadMeta(ch) {
  const bad = Math.max(0, Number(ch.bad || 0));
  const total = Math.max(1, Number(ch.frames || 0));
  const pct = Math.max(0, Math.min(100, (100 * bad) / total));
  const pctText = `${Math.round(pct)}%`;
  return {
    bad,
    total,
    pct,
    text: `${ch.time} | BAD ${bad}/${total} (${pctText})`,
  };
}

function chapterBadSparkSvg(ch) {
  const badFrames = Array.isArray(ch.bad_frames) ? ch.bad_frames : [];
  const cacheKey = `${ch.title}:${ch.start_frame}:${ch.end_frame}:${badFrames.length}`;
  if (chapterSparkSvgCache.has(cacheKey)) return chapterSparkSvgCache.get(cacheKey);
  const start = Number(ch.start_frame || 0);
  const end = Number(ch.end_frame || (start + Math.max(1, Number(ch.frames || 1))));
  const total = Math.max(1, end - start);
  const bucketCount = 28;
  const buckets = new Array(bucketCount).fill(0);

  badFrames.forEach((fidRaw) => {
    const fid = Number(fidRaw);
    if (!Number.isFinite(fid) || fid < start || fid >= end) return;
    const local = fid - start;
    const idx = Math.min(bucketCount - 1, Math.max(0, Math.floor((local / total) * bucketCount)));
    buckets[idx] += 1;
  });

  const peak = Math.max(1, ...buckets);
  const w = 140;
  const h = 18;
  const baseY = h - 3;
  const topY = 2;
  const step = w / Math.max(1, bucketCount - 1);
  const points = buckets
    .map((v, i) => {
      const x = i * step;
      const y = baseY - ((v / peak) * (baseY - topY));
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
  const dotSvg = buckets
    .map((v, i) => {
      if (v <= 0) return '';
      const x = i * step;
      return `<circle cx="${x.toFixed(2)}" cy="${baseY.toFixed(2)}" r="1.35" fill="var(--bad)" opacity="0.98"></circle>`;
    })
    .join('');

  const svg = `
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" width="100%" height="100%" aria-hidden="true">
      <line x1="0" y1="${baseY}" x2="${w}" y2="${baseY}" stroke="var(--line)" stroke-width="1"></line>
      <polyline points="${points}" fill="none" stroke="var(--spark-base)" stroke-width="1.25" opacity="0.95"></polyline>
      ${dotSvg}
    </svg>
  `;
  chapterSparkSvgCache.set(cacheKey, svg);
  return svg;
}

function renderArchives() {
  archiveListEl.innerHTML = '';
  state.archives.forEach(name => {
    const btn = document.createElement('button');
    btn.className = `item-btn ${name === state.archive ? 'active' : ''}`;
    btn.textContent = name;
    btn.addEventListener('click', () => loadArchive(name, null));
    archiveListEl.appendChild(btn);
  });
}

function renderChapters() {
  const chapters = state.chapters || [];
  // Fast path: if the list hasn't changed, just toggle the active class
  if (chapterButtonCache.size === chapters.length
      && chapters.every(ch => chapterButtonCache.has(ch.title))) {
    chapterButtonCache.forEach((el, title) => {
      el.classList.toggle('active', title === state.chapter);
    });
    return;
  }
  // Full rebuild: chapter list changed (new archive loaded)
  chapterButtonCache.clear();
  const fragment = document.createDocumentFragment();
  chapters.forEach(ch => {
    const m = chapterBadMeta(ch);
    const row = document.createElement('div');
    row.className = `item-btn chapter-row ${ch.title === state.chapter ? 'active' : ''}`;
    row.setAttribute('role', 'button');
    row.setAttribute('tabindex', '0');

    const idxEl = document.createElement('div');
    idxEl.textContent = String(ch.index).padStart(2, '0');

    const titleEl = document.createElement('div');
    titleEl.className = 'title';
    titleEl.title = ch.title;
    titleEl.textContent = ch.title;

    const actionsEl = document.createElement('div');
    actionsEl.className = 'chapter-row-actions';

    const loadBtn = document.createElement('button');
    loadBtn.className = 'action chapter-load-btn';
    loadBtn.type = 'button';
    loadBtn.textContent = 'Load';
    loadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      state.chapter = ch.title;
      renderChapters();
      const result = navigateToStep(stepNumForMode('review'));
      if (result && typeof result.then === 'function') void result;
    });

    const renameBtn = document.createElement('button');
    renameBtn.className = 'action chapter-rename-btn';
    renameBtn.type = 'button';
    renameBtn.title = 'Rename chapter';
    renameBtn.textContent = '✎';
    renameBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      startRenameChapter(ch, titleEl, actionsEl, loadBtn, renameBtn);
    });

    actionsEl.appendChild(loadBtn);
    actionsEl.appendChild(renameBtn);

    row.appendChild(idxEl);
    row.appendChild(titleEl);
    row.appendChild(actionsEl);

    row.addEventListener('click', () => {
      state.chapter = ch.title;
      renderChapters();
    });
    row.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        state.chapter = ch.title;
        renderChapters();
      }
    });

    chapterButtonCache.set(ch.title, row);
    fragment.appendChild(row);
  });
  chapterListEl.replaceChildren(fragment);
}

function startRenameChapter(ch, titleEl, actionsEl, loadBtn, renameBtn) {
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'chapter-rename-input';
  input.value = ch.title;
  titleEl.textContent = '';
  titleEl.appendChild(input);
  input.focus();
  input.select();

  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.className = 'action chapter-save-rename-btn';
  saveBtn.title = 'Save rename';
  saveBtn.textContent = '✓';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'action warn chapter-cancel-rename-btn';
  cancelBtn.title = 'Cancel rename';
  cancelBtn.textContent = '✗';

  function cancelRename() {
    titleEl.textContent = '';
    titleEl.title = ch.title;
    titleEl.textContent = ch.title;
    actionsEl.replaceChildren(loadBtn, renameBtn);
  }

  async function doSave() {
    const newTitle = input.value.trim();
    if (!newTitle || newTitle === ch.title) { cancelRename(); return; }
    saveBtn.disabled = true;
    cancelBtn.disabled = true;
    await saveRenameChapter(state.archive, ch.title, newTitle);
  }

  saveBtn.addEventListener('click', (e) => { e.stopPropagation(); void doSave(); });
  cancelBtn.addEventListener('click', (e) => { e.stopPropagation(); cancelRename(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); void doSave(); }
    if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
  });

  actionsEl.replaceChildren(saveBtn, cancelBtn);
}

async function saveRenameChapter(archive, oldTitle, newTitle) {
  setLoading(true, `Renaming "${oldTitle}" → "${newTitle}"…`);
  try {
    const result = await api('/api/rename_chapter', 'POST', { archive, old_title: oldTitle, new_title: newTitle }, 120000);
    if (!result.ok) throw new Error(result.error || 'Rename failed.');
    const renamed = result.renamed_files || [];
    setStatus(renamed.length
      ? `Renamed ${renamed.length} file(s): ${renamed.join(', ')}`
      : 'Chapter renamed (no rendered files found).'
    );
    await loadArchive(archive, newTitle);
  } catch (err) {
    setStatus(String(err.message || 'Rename failed.'), true);
    setLoading(false);
  }
}

function statsText(stats, threshold) {
  const forceText = state.forceAllFramesGood ? ' | Force Good: ON' : '';
  return `Bad: ${stats.bad} (${Math.round((100 * stats.bad) / Math.max(1, stats.total))}%) | Good: ${stats.good} | Overrides: ${stats.overrides}${forceText}`;
}
