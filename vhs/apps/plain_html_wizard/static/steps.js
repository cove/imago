async function loadArchives() {
  const payload = await api('/api/archives', 'GET', null, 15000);
  state.archives = payload.archives || [];
  if (!state.archives.length) {
    setStatus('No archives found.', true);
    return;
  }
  state.archive = payload.selected || state.archives[0];
  renderArchives();
  await loadArchive(state.archive, null);
}

async function loadArchive(archive, chapter) {
  setLoading(true, `Loading chapter metadata for ${archive}...`);
  try {
    const q = new URLSearchParams({ archive });
    if (chapter) q.set('chapter', chapter);
    const payload = await api(`/api/archive_state?${q.toString()}`, 'GET', null, 20000);
    const st = payload.archive_state;
    state.archive = st.archive;
    state.chapter = st.chapter;
    state.chapters = st.chapters || [];
    state.peopleProfile = normalizePeopleProfile(null);
    state.subtitlesProfile = normalizeSubtitlesProfile(null);
    state.splitProfile = normalizeSplitProfile(null);
    state.frameSheetConfig = normalizeFrameSheetConfig(null);
    resetFrameSheetPrefetchState();
    peopleTimelineZoom = 1.0;
    resetTimelineAudioState();
    resetFlipbookAudioState();
    refreshPeopleEditorFromState();
    refreshSubtitlesEditorFromState();
    refreshSplitEditorFromState();
    renderArchives();
    renderChapters();
    setStatus(st.status || '');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setLoading(false);
  }
}

async function loadFrames() {
  if (isChapterLoadInFlight) {
    setStatus('A chapter load is already in progress. Cancel it first if you want to change chapters.', true);
    return false;
  }
  if (!state.archive || !state.chapter) {
    setStatus('Select an archive and chapter first.', true);
    return false;
  }

  const payload = {
    archive: state.archive,
    chapter: state.chapter,
    iqr_k: Number(iqrEl.value || 3.5),
    debug_extract: false,
    force_all_frames_good: Boolean(state.forceAllFramesGood),
  };
  const reloadingLoadedChapter = isRequestedChapterAlreadyLoaded(state.archive, state.chapter);

  if (autoIqrTimer) {
    window.clearTimeout(autoIqrTimer);
    autoIqrTimer = null;
  }
  state.iqrApplySeq += 1;

  let loadFailed = false;
  startLoadProgress('Extracting chapter and loading all frames...');
  if (!reloadingLoadedChapter) {
    // Switching chapter should blank the review surface immediately so old contact sheets never flash.
    state.frameImages = new Map();
    state.freezeReplacementMap = new Map();
    state.loadSettings = null;
    replaceGammaScores(new Map());
    state.gammaProfile = normalizeGammaProfile(null);
    state.peopleProfile = normalizePeopleProfile(null);
    state.subtitlesProfile = normalizeSubtitlesProfile(null);
    state.splitProfile = normalizeSplitProfile(null);
    state.frameSheetConfig = normalizeFrameSheetConfig(null);
    resetFrameSheetPrefetchState();
    peopleTimelineZoom = 1.0;
    resetTimelineAudioState();
    resetFlipbookAudioState();
    replaceReviewState({
      threshold: 0,
      stats: { total: 0, bad: 0, good: 0, shown: 0, overrides: 0 },
      frames: [],
    });
    updateReviewStatsDisplay();
    renderReviewFrames([], { suppressPlaceholderMerge: true });
    refreshPeopleEditorFromState();
    refreshSubtitlesEditorFromState();
    refreshSplitEditorFromState();
  }
  setStepByMode('review');
  try {
    const result = await api('/api/load_chapter', 'POST', payload, 0);
    state.frameImages = new Map();
    replaceGammaScores(new Map());
    state.freezeReplacementMap = new Map();
    const finalFrames = sortFramesByFid((result.review && result.review.frames) || []);
    setReviewState({
      ...result.review,
      frames: finalFrames,
    }, state.review && Array.isArray(state.review.frames) ? state.review.frames : null);
    syncForceAllFramesGoodFromReview(result.review || {});
    state.loadSettings = result.settings || null;
    state.frameSheetConfig = normalizeFrameSheetConfig(
      (state.loadSettings && state.loadSettings.contact_sheet) || null
    );
    resetFrameSheetPrefetchState();
    state.gammaProfile = normalizeGammaProfile(
      (state.loadSettings && state.loadSettings.gamma_profile) || null
    );
    state.peopleProfile = normalizePeopleProfile(
      (state.loadSettings && state.loadSettings.people_profile) || null
    );
    state.subtitlesProfile = normalizeSubtitlesProfile(
      (state.loadSettings && state.loadSettings.subtitles_profile) || null
    );
    state.autoTranscript = (state.loadSettings && state.loadSettings.auto_transcript) === 'on';
    if (subtitlesAutoTranscriptEl) subtitlesAutoTranscriptEl.checked = state.autoTranscript;
    state.splitProfile = normalizeSplitProfile(
      (state.loadSettings && state.loadSettings.split_profile) || null
    );
    const rawAudioSync = state.loadSettings && state.loadSettings.audio_sync_profile;
    state.audioSyncOffset = (rawAudioSync && typeof rawAudioSync.offset_seconds === 'number')
      ? rawAudioSync.offset_seconds
      : 0.0;
    if (typeof updateAudioSyncOffsetLabel === 'function') updateAudioSyncOffsetLabel();
    updateGammaControls();
    refreshPeopleEditorFromState();
    refreshSubtitlesEditorFromState();
    refreshSplitEditorFromState();
    updateReviewStatsDisplay();
    renderReviewFrames(state.review.frames);
    iqrEl.value = String(payload.iqr_k);
    iqrLabelEl.textContent = Number(payload.iqr_k).toFixed(2);
    const s = state.loadSettings;
    const loadedCount = s ? Number(s.loaded_count || 0) : Number(result.review.stats.total || 0);
    const chapterTotal = s ? Number(s.chapter_frame_count || 0) : Number(result.review.stats.total || 0);
    setStatus('');
    setStepByMode('review');
    void ensureTimelineAudioLoaded(false);
    return true;
  } catch (err) {
    loadFailed = true;
    const msg = String(err?.message || err || 'Load failed.');
    const cancelled = /cancel/i.test(msg);
    setStatus(msg, !cancelled);
    return false;
  } finally {
    finishLoadProgress(!loadFailed);
  }
}

async function applyIqr(isManual = true) {
  if (isChapterLoadInFlight && isManual) {
    setStatus('Wait until frame loading finishes before applying IQR manually.', true);
    return;
  }
  if (!state.review) return;
  if (!isManual && state.liveIqrInFlight) {
    state.liveIqrPending = true;
    return;
  }
  const k = Number(iqrEl.value || 3.5);
  iqrLabelEl.textContent = k.toFixed(2);
  const callSeq = ++state.iqrApplySeq;
  if (isManual) {
    setLoading(true, 'Applying IQR threshold...');
  } else {
    state.liveIqrInFlight = true;
    setStatus(`Reprocessing frame statuses (IQR k=${k.toFixed(2)})...`);
  }
  try {
    const result = await api('/api/apply_iqr', 'POST', {
      iqr_k: k,
      force_all_frames_good: Boolean(state.forceAllFramesGood),
    });
    if (callSeq !== state.iqrApplySeq) return;
    const review = {
      ...result.review,
      frames: sortFramesByFid((result.review && result.review.frames) || []),
    };
    syncForceAllFramesGoodFromReview(review);
    const previous = new Map((state.review.frames || []).map(f => [String(f.fid), f.status]));
    let changed = 0;
    for (const f of (review.frames || [])) {
      if (previous.get(String(f.fid)) !== f.status) changed += 1;
    }
    reviewStatsEl.textContent = statsText(review.stats, review.threshold);
    applyFrameUpdates(review.frames);
    setReviewState(
      {
        ...(state.review || {}),
        stats: review.stats,
        threshold: review.threshold,
        frames: review.frames || [],
      },
      state.review && Array.isArray(state.review.frames) ? state.review.frames : null
    );
    renderActiveSparkline(state.review.frames, review.threshold);
    replaceGammaScores(new Map());
    refreshFreezeSimulation();
    updateReviewStatsDisplay();
    setStatus(`Applied IQR k=${k.toFixed(2)} (${changed} frame(s) changed).`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (isManual) {
      setLoading(false);
    } else {
      state.liveIqrInFlight = false;
      if (state.liveIqrPending) {
        state.liveIqrPending = false;
        applyIqr(false);
      }
    }
  }
}

async function prefillPeopleFromCast() {
  if (isChapterLoadInFlight || isPreviewRenderInFlight || isSubtitlesGenerateInFlight) {
    setStatus('Wait until current processing finishes before running Cast prefill.', true);
    return;
  }
  if (!state.archive || !state.chapter) {
    setStatus('Load a chapter before running Cast prefill.', true);
    return;
  }
  if (!syncPeopleProfileFromEditor(true)) return;
  setLoading(true, 'Generating people subtitle draft from cast database...');
  try {
    const result = await api('/api/people_prefill_cast', 'POST', {
      mode: 'replace',
    });
    state.peopleProfile = normalizePeopleProfile(result.people_profile || null);
    peopleTimelineDraft = null;
    peopleTimelineDrag = null;
    refreshPeopleEditorFromState();
    updateReviewStatsDisplay();
    setStatus(result.message || 'Cast prefill complete.');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setLoading(false);
  }
}

async function generateSubtitlesFromWhisper() {
  if (isChapterLoadInFlight || isPreviewRenderInFlight || isSubtitlesGenerateInFlight) {
    setStatus('Wait until current processing finishes before generating subtitles.', true);
    return;
  }
  if (!state.archive || !state.chapter) {
    setStatus('Load a chapter before generating subtitles.', true);
    return;
  }
  if (!syncSubtitlesProfileFromEditor(true)) return;
  const hasExisting = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []).length > 0;
  const mode = hasExisting ? 'append' : 'replace';
  startSubtitlesProgress('Generating dialogue subtitles with Whisper...');
  let ok = false;
  try {
    const result = await api('/api/subtitles_generate', 'POST', { mode }, 0);
    state.subtitlesProfile = subtitlesProfileFromApiResult(result);
    refreshSubtitlesEditorFromState();
    updateReviewStatsDisplay();
    const appliedCount = canonicalizeSubtitlesEntries(
      (state.subtitlesProfile && state.subtitlesProfile.entries) || []
    ).length;
    if (appliedCount > 0) {
      setStatus(result.message || `Subtitle generation complete (${appliedCount} entries).`);
    } else {
      const generated = Math.max(0, Math.trunc(Number(result.generated_count || 0) || 0));
      if (generated > 0) {
        setStatus(
          `Generated ${generated} subtitle entr${generated === 1 ? 'y' : 'ies'}, but none were loaded into the editor. Try reloading the chapter.`,
          true
        );
      } else {
        setStatus(result.message || 'Subtitle generation complete.');
      }
    }
    ok = true;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    finishSubtitlesProgress(ok);
  }
}

function clearPeopleEntries() {
  peopleTimelineDraft = null;
  peopleTimelineDrag = null;
  state.peopleProfile = {
    ...(state.peopleProfile || {}),
    entries: [],
  };
  refreshPeopleEditorFromState();
  updateReviewStatsDisplay();
  setStatus('Cleared people rows for this chapter draft.');
}

function visibleGammaRangeFromView() {
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) return null;
  const range = state.visibleRange || _computeVisibleIndexRange();
  if (!range) return null;
  const startIdx = _clamp(Number(range.start || 0), 0, state.review.frames.length - 1);
  const endIdx = _clamp(Number(range.end || startIdx), startIdx, state.review.frames.length - 1);
  const startFid = Number(state.review.frames[startIdx] && state.review.frames[startIdx].fid);
  const endFid = Number(state.review.frames[endIdx] && state.review.frames[endIdx].fid);
  if (!Number.isFinite(startFid) || !Number.isFinite(endFid)) return null;
  return {
    start_frame: Math.trunc(Math.min(startFid, endFid)),
    end_frame: Math.trunc(Math.max(startFid, endFid)) + 1,
  };
}

function applyGammaVisibleRange() {
  if (!isGammaStepActive()) return;
  if (!state.gammaProfile || state.gammaProfile.mode !== 'regions') {
    setStatus('Set Gamma Correction mode to "Visible Range Regions" before applying ranges.', true);
    return;
  }
  const range = visibleGammaRangeFromView();
  if (!range) {
    setStatus('No visible range found to apply gamma.', true);
    return;
  }
  const gamma = normalizeGammaValue(gammaLevelEl && gammaLevelEl.value, state.gammaProfile.level || 1.0);
  const next = canonicalizeGammaRanges([...(state.gammaProfile.ranges || []), { ...range, gamma }]);
  state.gammaProfile = {
    ...(state.gammaProfile || {}),
    level: gamma,
    ranges: next,
  };
  invalidateGammaSparklineCache();
  updateGammaControls();
  refreshGammaVisuals();
  renderActiveSparkline(state.review ? state.review.frames : [], state.review ? state.review.threshold : 0);
  updateReviewStatsDisplay();
  setStatus(`Applied gamma ${gamma.toFixed(2)} to visible range ${range.start_frame}-${range.end_frame}.`);
}

function clearGammaRanges() {
  if (!state.gammaProfile) return;
  const mode = String(state.gammaProfile.mode || 'whole');
  if (mode === 'whole') {
    state.gammaProfile.level = 1.0;
  } else {
    state.gammaProfile.ranges = [];
  }
  invalidateGammaSparklineCache();
  updateGammaControls();
  refreshGammaVisuals();
  renderActiveSparkline(state.review ? state.review.frames : [], state.review ? state.review.threshold : 0);
  updateReviewStatsDisplay();
  setStatus('Cleared gamma adjustments for current mode.');
}

function setActiveGammaLevel(rawValue) {
  const level = normalizeGammaValue(rawValue, state.gammaProfile.level || 1.0);
  state.gammaProfile.level = level;
  invalidateGammaSparklineCache();
  updateGammaControls();
  if (isGammaStepActive()) {
    refreshGammaVisuals();
    renderActiveSparkline(state.review ? state.review.frames : [], state.review ? state.review.threshold : 0);
    updateReviewStatsDisplay();
    updateGammaRangeMeta();
  }
}

async function openGammaStep() {
  if (isChapterLoadInFlight) {
    setStatus('Wait until frame loading finishes before entering gamma step.', true);
    return false;
  }
  if (!reviewLoadedFrameCount()) {
    setStatus('Load and review frames before entering gamma step.', true);
    return false;
  }
  if (state.gammaProfile) {
    state.gammaProfile.level = normalizeGammaValue(
      gammaLevelEl ? gammaLevelEl.value : state.gammaProfile.level,
      state.gammaProfile.level || state.gammaProfile.defaultGamma || 1.0,
    );
  }
  if (pendingToggleRequests.size > 0) {
    setLoading(true, 'Syncing frame edits...');
    try {
      await flushPendingToggleRequests();
    } finally {
      setLoading(false);
    }
  }
  setStepByMode('gamma', stepNumForMode('review'));
  renderActiveSparkline(state.review.frames, state.review.threshold || 0);
  updateReviewStatsDisplay();
  ensureGammaScores();
  return true;
}

async function openPeopleStep() {
  if (!reviewLoadedFrameCount()) {
    setStatus('Load and review frames before entering people step.', true);
    return false;
  }
  if (!syncPeopleProfileFromEditor(true)) return false;
  refreshPeopleEditorFromState();
  setStepByMode('people', stepNumForMode('gamma', stepNumForMode('review')));
  updateReviewStatsDisplay();
  return true;
}

async function openSubtitlesStep() {
  if (!reviewLoadedFrameCount()) {
    setStatus('Load and review frames before entering subtitles step.', true);
    return false;
  }
  if (!syncSubtitlesProfileFromEditor(true)) return false;
  refreshSubtitlesEditorFromState();
  setStepByMode('subtitles', stepNumForMode('people', stepNumForMode('gamma', stepNumForMode('review'))));
  updateReviewStatsDisplay();
  return true;
}

async function openSplitStep() {
  if (!reviewLoadedFrameCount()) {
    setStatus('Load and review frames before entering chapter step.', true);
    return false;
  }
  if (!syncSplitProfileFromEditor(true)) return false;
  refreshSplitEditorFromState();
  setStepByMode('split', stepNumForMode('subtitles', stepNumForMode('people', stepNumForMode('gamma', stepNumForMode('review')))));
  updateReviewStatsDisplay();
  return true;
}

async function applyForceAllFramesGood(enabled, options = {}) {
  const next = Boolean(enabled);
  const previous = Boolean(state.forceAllFramesGood);
  if (isPreviewRenderInFlight || isSubtitlesGenerateInFlight) {
    setForceAllFramesGoodUi(previous);
    setStatus('Wait until active tasks finish before toggling Force All Good.', true);
    return;
  }
  setForceAllFramesGoodUi(next);

  const showOverlay = options && Object.prototype.hasOwnProperty.call(options, 'showOverlay')
    ? Boolean(options.showOverlay)
    : !isChapterLoadInFlight;
  if (showOverlay) {
    setLoading(true, next ? 'Forcing all loaded frames to good...' : 'Restoring frame classification...');
  }
  // Save a compact snapshot of current frame statuses before enabling force,
  // so we can restore them locally when disabling without a server round-trip.
  if (next && state.review && Array.isArray(state.review.frames)) {
    const snap = new Map();
    for (const f of state.review.frames) {
      if (f && !f.placeholder && !f.loading) snap.set(Number(f.fid), { status: f.status, source: f.source });
    }
    state._preForceFrameStatuses = snap;
  }
  try {
    const result = await api('/api/set_force_all_good', 'POST', { enabled: next });
    const reviewRaw = result.review || {};
    syncForceAllFramesGoodFromReview(reviewRaw);
    const serverFrames = (result.review != null && 'frames' in result.review) ? result.review.frames : undefined;
    if (serverFrames === null) {
      // Fast path: server skipped frame serialization; patch statuses locally.
      const existingFrames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
      let patchedFrames;
      if (next) {
        patchedFrames = existingFrames.map(f =>
          (!f || f.placeholder || f.loading) ? f : { ...f, status: 'good', source: 'FG' }
        );
      } else {
        const snapshot = state._preForceFrameStatuses;
        patchedFrames = existingFrames.map(f => {
          if (!f || f.placeholder || f.loading) return f;
          const saved = snapshot && snapshot.get(Number(f.fid));
          return saved ? { ...f, status: saved.status, source: saved.source } : f;
        });
        state._preForceFrameStatuses = null;
      }
      state.review = {
        ...(state.review || {}),
        stats: reviewRaw.stats || (state.review && state.review.stats) || { total: 0, bad: 0, good: 0, shown: 0, overrides: 0 },
        threshold: Number(reviewRaw.threshold != null ? reviewRaw.threshold : ((state.review && state.review.threshold) || 0)),
        frames: patchedFrames,
      };
      invalidateReviewSparklineCache();
      if (patchedFrames.length) {
        reviewStatsEl.textContent = statsText(state.review.stats, state.review.threshold);
        if (!isChapterLoadInFlight) renderActiveSparkline(state.review.frames, state.review.threshold);
        renderFrameGridWindow(true);
      }
    } else {
      const review = {
        ...reviewRaw,
        frames: sortFramesByFid(reviewRaw.frames || []),
      };
      setReviewState({
        ...(state.review || {}),
        stats: review.stats || { total: 0, bad: 0, good: 0, shown: 0, overrides: 0 },
        threshold: Number(review.threshold || 0),
        frames: review.frames || [],
      }, state.review && Array.isArray(state.review.frames) ? state.review.frames : null);
      if (Array.isArray(state.review.frames) && state.review.frames.length) {
        reviewStatsEl.textContent = statsText(state.review.stats, state.review.threshold);
        if (!isChapterLoadInFlight) {
          renderActiveSparkline(state.review.frames, state.review.threshold);
        }
        applyFrameUpdates(state.review.frames);
      } else {
        renderReviewFrames([]);
      }
    }
    replaceGammaScores(new Map());
    refreshFreezeSimulation();
    updateReviewStatsDisplay();
    setStatus(next ? 'Force All Good enabled for current loaded frames.' : 'Force All Good disabled.');
  } catch (err) {
    setForceAllFramesGoodUi(previous);
    setStatus(err.message, true);
  } finally {
    if (showOverlay) {
      setLoading(false);
    }
  }
}

async function toggleFrame(fid, options = {}) {
  if (state.forceAllFramesGood) {
    setStatus('Disable Force All Good before editing frame statuses.', true);
    return;
  }
  const showOverlay = options && options.showOverlay === false ? false : true;
  const optimistic = options && options.optimistic === true;
  const fidStr = String(fid);
  let optimisticSnapshot = null;
  if (optimistic && isChapterLoadInFlight && state.review && Array.isArray(state.review.frames)) {
    const idx = state.review.frames.findIndex(f => String(f && f.fid) === fidStr);
    if (idx >= 0) {
      const prev = state.review.frames[idx];
      optimisticSnapshot = { idx, prev: { ...prev } };
      const nextStatus = prev.status === 'bad' ? 'good' : 'bad';
      const nextSource = nextStatus === 'bad' ? 'MB' : 'MG';
      const next = { ...prev, status: nextStatus, source: nextSource };
      const match = String(prev.label || '').match(/\s(?:MG|MB|AG|AB)\s*$/);
      if (match) {
        next.label = String(prev.label || '').replace(/\s(?:MG|MB|AG|AB)\s*$/, ` ${nextSource}`);
      }
      state.review.frames[idx] = next;
      applyFrameUpdates([next]);
    }
  }
  if (showOverlay) {
    setLoading(true, `Updating ${chapterTimecodeFromFid(fid)}...`);
  }
  try {
    const result = await api('/api/toggle_frame', 'POST', { fid: Number(fid) });
    syncForceAllFramesGoodFromReview(result.review || {});
    const frame = result.frame;
    if (frame) applyFrameUpdates([frame]);
    reviewStatsEl.textContent = statsText(result.review.stats, result.review.threshold);
    if (!isChapterLoadInFlight || !optimistic) {
      setReviewState(
        {
          ...(state.review || {}),
          stats: result.review.stats,
          threshold: result.review.threshold,
          frames: result.review.frames || [],
        },
        state.review && Array.isArray(state.review.frames) ? state.review.frames : null
      );
      renderActiveSparkline(state.review.frames, result.review.threshold);
    } else if (state.review) {
      state.review = {
        ...state.review,
        stats: result.review.stats,
        threshold: result.review.threshold,
      };
    }
    replaceGammaScores(new Map());
    refreshFreezeSimulation();
    updateReviewStatsDisplay();
  } catch (err) {
    if (optimisticSnapshot && state.review && Array.isArray(state.review.frames)) {
      const { idx, prev } = optimisticSnapshot;
      if (idx >= 0 && idx < state.review.frames.length) {
        state.review.frames[idx] = prev;
        applyFrameUpdates([prev]);
      }
    }
    setStatus(err.message, true);
  } finally {
    if (showOverlay) {
      setLoading(false);
    }
  }
}

async function setFrameRangeStatus(startFid, endFid, status = 'bad', options = {}) {
  if (state.forceAllFramesGood) {
    setStatus('Disable Force All Good before editing frame statuses.', true);
    return;
  }
  const showOverlay = options && options.showOverlay === false ? false : true;
  const lo = Math.trunc(Math.min(Number(startFid), Number(endFid)));
  const hi = Math.trunc(Math.max(Number(startFid), Number(endFid)));
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    setStatus('Invalid frame range.', true);
    return;
  }
  if (showOverlay) {
    setLoading(true, `Updating range ${chapterTimecodeFromFid(lo)}-${chapterTimecodeFromFid(hi)}...`);
  }
  try {
    const result = await api('/api/set_frame_range', 'POST', {
      start_fid: lo,
      end_fid: hi,
      status: String(status || 'bad'),
    });
    const review = {
      ...result.review,
      frames: sortFramesByFid((result.review && result.review.frames) || []),
    };
    syncForceAllFramesGoodFromReview(review);
    reviewStatsEl.textContent = statsText(review.stats, review.threshold);
    applyFrameUpdates(review.frames);
    setReviewState(
      {
        ...(state.review || {}),
        stats: review.stats,
        threshold: review.threshold,
        frames: review.frames || [],
      },
      state.review && Array.isArray(state.review.frames) ? state.review.frames : null
    );
    if (!isChapterLoadInFlight) {
      renderActiveSparkline(state.review.frames, review.threshold);
    }
    replaceGammaScores(new Map());
    refreshFreezeSimulation();
    updateReviewStatsDisplay();
    setStatus(
      `Marked ${Number(result.updated_count || 0)} frame(s) as ${String(result.status || status)} in ${chapterTimecodeFromFid(lo)}-${chapterTimecodeFromFid(hi)}.`
    );
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    if (showOverlay) {
      setLoading(false);
    }
  }
}

async function loadSummary() {
  if (isChapterLoadInFlight) {
    setStatus('Wait until the current chapter load finishes before building summary.', true);
    return false;
  }
  if (!syncPeopleProfileFromEditor(true)) return false;
  if (pendingToggleRequests.size > 0) {
    setLoading(true, 'Syncing frame edits...');
    try {
      await flushPendingToggleRequests();
    } finally {
      setLoading(false);
    }
  }
  if (!syncSubtitlesProfileFromEditor(true)) return false;
  if (!syncSplitProfileFromEditor(true)) return false;
  const saved = await saveProgress({ showOverlay: true, loadingText: 'Saving wizard progress...', silent: true });
  if (!saved) return false;
  setLoading(true, 'Building summary...');
  try {
    const result = await api('/api/summary');
    summaryBoxEl.textContent = String(result.summary || '').trim();
    setStepByMode('summary', stepNumForMode('split', stepNumForMode('subtitles', stepNumForMode('people', stepNumForMode('gamma', stepNumForMode('review'))))));
    return true;
  } catch (err) {
    setStatus(err.message, true);
    return false;
  } finally {
    setLoading(false);
  }
}

async function toggleCurrentFlipbookFrame() {
  if (state.forceAllFramesGood) {
    setStatus('Disable Force All Good before editing frame statuses.', true);
    return;
  }
  if (sparkPlayTimer) return;
  if (!flipbookPreviewEl || !flipbookPreviewEl.classList.contains('active')) return;
  if (flipbookToggleInFlight) return;
  if (!sparkPlayFrames.length) return;
  const current = sparkPlayFrames[sparkPlayIndex];
  const fid = Number(current && current.fid);
  if (!Number.isFinite(fid)) return;
  flipbookToggleInFlight = true;
  updateFlipbookControls();
  try {
    await trackPendingToggleRequest(toggleFrame(fid, { showOverlay: false }));
    syncSparkPlayFramesFromGrid();
    renderSparkPlaybackFrame(sparkPlayIndex, { keepVisible: false });
  } finally {
    flipbookToggleInFlight = false;
    updateFlipbookControls();
  }
}

async function previewRender() {
  if (isChapterLoadInFlight || isPreviewRenderInFlight || isSubtitlesGenerateInFlight) {
    setStatus('Wait until frame loading finishes before preview render.', true);
    return;
  }
  if (!reviewLoadedFrameCount()) {
    setStatus('Load and review a chapter before running preview render.', true);
    return;
  }
  startPreviewProgress('Rendering chapter preview from current frame selections...');
  let ok = false;
  try {
    const result = await api('/api/preview_render', 'POST', {
      preview_mode: stepDef(state.wizardStep).mode,
      force_all_frames_good: Boolean(state.forceAllFramesGood),
      gamma_profile: {
        default_gamma: normalizeGammaValue(state.gammaProfile && state.gammaProfile.defaultGamma, 1.0),
        ranges: gammaRangesForSave(),
      },
      audio_sync_profile: {
        offset_seconds: typeof state.audioSyncOffset === 'number' ? state.audioSyncOffset : 0.0,
      },
    }, 0);
    const target = result.preview_page_url || result.preview_url || '';
    if (target) {
      window.open(target, '_blank', 'noopener');
    }
    setStatus(result.message || 'Preview render ready.');
    ok = true;
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    finishPreviewProgress(ok);
  }
}

async function saveProgress(options = {}) {
  if (isChapterLoadInFlight) {
    setStatus('Wait until frame loading finishes before saving progress.', true);
    return false;
  }
  if (!reviewLoadedFrameCount()) {
    setStatus('Load a chapter before saving progress.', true);
    return false;
  }
  if (pendingToggleRequests.size > 0) {
    await flushPendingToggleRequests();
  }
  if (!syncPeopleProfileFromEditor(Boolean(options.showErrors !== false))) {
    return false;
  }
  if (!syncSubtitlesProfileFromEditor(Boolean(options.showErrors !== false))) {
    return false;
  }
  if (!syncSplitProfileFromEditor(Boolean(options.showErrors !== false))) {
    return false;
  }
  const peopleEntries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  const subtitleEntries = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  const splitEntries = canonicalizeSplitEntries((state.splitProfile && state.splitProfile.entries) || []);
  const showOverlay = options.showOverlay !== false;
  const loadingText = String(options.loadingText || 'Saving wizard progress...');
  const silent = Boolean(options.silent);
  if (showOverlay) {
    setLoading(true, loadingText);
  }
  try {
    const result = await api('/api/save_progress', 'POST', {
      force_all_frames_good: Boolean(state.forceAllFramesGood),
      gamma_profile: {
        default_gamma: normalizeGammaValue(state.gammaProfile && state.gammaProfile.defaultGamma, 1.0),
        ranges: gammaRangesForSave(),
      },
      audio_sync_profile: {
        offset_seconds: typeof state.audioSyncOffset === 'number' ? state.audioSyncOffset : 0.0,
      },
      people_profile: {
        entries: peopleEntries,
      },
      subtitles_profile: {
        entries: subtitleEntries,
      },
      split_profile: {
        entries: splitEntries,
      },
    });
    if (!silent) {
      setStatus(result.message || 'Progress saved.');
    }
    return true;
  } catch (err) {
    if (!silent) {
      setStatus(err.message, true);
    }
    return false;
  } finally {
    if (showOverlay) {
      setLoading(false);
    }
  }
}

async function saveAndReturn() {
  if (isChapterLoadInFlight) {
    setStatus('Wait until frame loading finishes before saving.', true);
    return;
  }
  if (pendingToggleRequests.size > 0) {
    await flushPendingToggleRequests();
  }
  if (!syncPeopleProfileFromEditor(true)) return;
  if (!syncSubtitlesProfileFromEditor(true)) return;
  if (!syncSplitProfileFromEditor(true)) return;
  const peopleEntries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  const subtitleEntries = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  const splitEntries = canonicalizeSplitEntries((state.splitProfile && state.splitProfile.entries) || []);
  setLoading(true, 'Saving BAD frame, gamma, people, and subtitle selections...');
  try {
    const result = await api('/api/save', 'POST', {
      force_all_frames_good: Boolean(state.forceAllFramesGood),
      gamma_profile: {
        default_gamma: normalizeGammaValue(state.gammaProfile && state.gammaProfile.defaultGamma, 1.0),
        ranges: gammaRangesForSave(),
      },
      audio_sync_profile: {
        offset_seconds: typeof state.audioSyncOffset === 'number' ? state.audioSyncOffset : 0.0,
      },
      people_profile: {
        entries: peopleEntries,
      },
      subtitles_profile: {
        entries: subtitleEntries,
      },
      split_profile: {
        entries: splitEntries,
      },
    });
    setStatus(result.message || 'Saved.');

    const st = result.archive_state;
    state.archive = st.archive;
    state.chapter = st.chapter;
    state.chapters = st.chapters || [];
    state.peopleProfile = normalizePeopleProfile(null);
    state.subtitlesProfile = normalizeSubtitlesProfile(null);
    state.splitProfile = normalizeSplitProfile(null);
    refreshPeopleEditorFromState();
    refreshSubtitlesEditorFromState();
    refreshSplitEditorFromState();

    renderArchives();
    renderChapters();
    setStepByMode('load');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setLoading(false);
  }
}

function escapeHtml(input) {
  return String(input)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function navigateToStep(rawStep) {
  const targetStep = stepDef(rawStep);
  const activeStep = stepDef(state.wizardStep);
  if (targetStep.num === activeStep.num) return true;

  if (targetStep.mode === 'load') {
    stopSparkWindowPlayback();
    setStep(targetStep.num);
    return true;
  }

  let currentStep = stepDef(state.wizardStep);
  if (currentStep.mode === 'load') {
    if (isChapterLoadInFlight) {
      setStepByMode('review');
    } else {
      const loaded = await loadFrames();
      if (!loaded) return false;
    }
    currentStep = stepDef(state.wizardStep);
  }

  if (targetStep.mode === 'review') {
    stopSparkWindowPlayback();
    setStep(targetStep.num);
    return true;
  }

  if (targetStep.mode === 'audio_sync') {
    const movingForward = stepIndex(targetStep.num) > stepIndex(currentStep.num);
    if (movingForward) {
      const saved = await saveProgress({
        showOverlay: true,
        loadingText: 'Saving review progress...',
        silent: true,
      });
      if (!saved) return false;
    }
    setStep(targetStep.num);
    if (typeof openAudioSyncStep === 'function') await openAudioSyncStep();
    return true;
  }

  if (targetStep.mode === 'gamma') {
    const movingForward = stepIndex(targetStep.num) > stepIndex(currentStep.num);
    if (movingForward) {
      const saved = await saveProgress({
        showOverlay: true,
        loadingText: currentStep.mode === 'audio_sync' ? 'Saving audio sync...' : 'Saving review progress...',
        silent: true,
      });
      if (!saved) return false;
    }
    return openGammaStep();
  }

  if (targetStep.mode === 'people') {
    const movingForward = stepIndex(targetStep.num) > stepIndex(currentStep.num);
    if (movingForward && !isChapterLoadInFlight) {
      const loadingText = currentStep.mode === 'gamma'
        ? 'Saving gamma progress...'
        : (currentStep.mode === 'audio_sync' ? 'Saving audio sync...' : 'Saving review progress...');
      const saved = await saveProgress({
        showOverlay: true,
        loadingText,
        silent: true,
      });
      if (!saved) return false;
    }
    return openPeopleStep();
  }

  if (targetStep.mode === 'subtitles') {
    const movingForward = stepIndex(targetStep.num) > stepIndex(currentStep.num);
    if (movingForward && !isChapterLoadInFlight) {
      const loadingText = currentStep.mode === 'people'
        ? 'Saving people progress...'
        : (currentStep.mode === 'gamma' ? 'Saving gamma progress...'
          : (currentStep.mode === 'audio_sync' ? 'Saving audio sync...' : 'Saving review progress...'));
      const saved = await saveProgress({
        showOverlay: true,
        loadingText,
        silent: true,
      });
      if (!saved) return false;
    }
    return openSubtitlesStep();
  }

  if (targetStep.mode === 'split') {
    const movingForward = stepIndex(targetStep.num) > stepIndex(currentStep.num);
    if (movingForward && !isChapterLoadInFlight) {
      const loadingText = currentStep.mode === 'subtitles'
        ? 'Saving subtitles progress...'
        : (currentStep.mode === 'people'
          ? 'Saving people progress...'
          : (currentStep.mode === 'gamma' ? 'Saving gamma progress...'
            : (currentStep.mode === 'audio_sync' ? 'Saving audio sync...' : 'Saving review progress...')));
      const saved = await saveProgress({
        showOverlay: true,
        loadingText,
        silent: true,
      });
      if (!saved) return false;
    }
    return openSplitStep();
  }

  if (targetStep.mode === 'summary') {
    return loadSummary();
  }

  return false;
}

if (stepPillsEl) {
  stepPillsEl.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const pill = target.closest('.step-pill');
    if (!(pill instanceof HTMLButtonElement) || pill.disabled) return;
    const stepNum = Number(pill.dataset.step || 0);
    if (!Number.isFinite(stepNum)) return;
    const result = navigateToStep(stepNum);
    if (result && typeof result.then === 'function') {
      void result;
    }
  });
}

const saveAndReturnBtnEl = document.getElementById('saveAndReturn');
if (saveAndReturnBtnEl) {
  navActionButtons.set('saveAndReturn', saveAndReturnBtnEl);
  saveAndReturnBtnEl.addEventListener('click', () => {
    const result = saveAndReturn();
    if (result && typeof result.then === 'function') {
      void result;
    }
  });
}

if (loadChapterBtnEl) {
  loadChapterBtnEl.addEventListener('click', () => {
    const result = navigateToStep(stepNumForMode('review'));
    if (result && typeof result.then === 'function') void result;
  });
}

if (backToChaptersBtnEl) {
  backToChaptersBtnEl.addEventListener('click', () => {
    const result = navigateToStep(stepNumForMode('load'));
    if (result && typeof result.then === 'function') void result;
  });
}

previewRenderBtnEl.addEventListener('click', previewRender);
fullscreenBtnEl.addEventListener('click', toggleReviewFullscreen);
document.addEventListener('fullscreenchange', () => {
  if (page2El && page2El.classList.contains('flipbook-focus') && !isReviewFullscreenActive()) {
    stopSparkWindowPlayback();
    setFlipbookFocusMode(false);
  }
  updateFullscreenButton();
  scheduleVisibleRangeRefresh();
});
iqrEl.addEventListener('input', () => {
  iqrLabelEl.textContent = Number(iqrEl.value || 0).toFixed(2);
  if (isBadFrameStepActive()) {
    scheduleAutoApplyIqr();
  }
});
iqrEl.addEventListener('change', () => {
  if (isBadFrameStepActive()) {
    applyIqr(true);
  }
});
if (gammaLevelEl) {
  gammaLevelEl.addEventListener('input', () => {
    setActiveGammaLevel(gammaLevelEl.value);
  });
}
if (flipbookGammaLevelEl) {
  flipbookGammaLevelEl.addEventListener('input', () => {
    setActiveGammaLevel(flipbookGammaLevelEl.value);
  });
}
if (gammaModeEl) {
  gammaModeEl.addEventListener('change', () => {
    const mode = String(gammaModeEl.value || 'whole');
    state.gammaProfile.mode = mode === 'regions' ? 'regions' : 'whole';
    invalidateGammaSparklineCache();
    updateGammaControls();
    refreshGammaVisuals();
    if (isGammaStepActive()) {
      renderActiveSparkline(state.review ? state.review.frames : [], state.review ? state.review.threshold : 0);
    }
    updateReviewStatsDisplay();
  });
}
if (gammaApplyVisibleBtnEl) {
  gammaApplyVisibleBtnEl.addEventListener('click', applyGammaVisibleRange);
}
if (gammaClearBtnEl) {
  gammaClearBtnEl.addEventListener('click', clearGammaRanges);
}
if (peoplePrefillCastBtnEl) {
  peoplePrefillCastBtnEl.addEventListener('click', prefillPeopleFromCast);
}
if (subtitlesAutoTranscriptEl) {
  subtitlesAutoTranscriptEl.addEventListener('change', async () => {
    const enabled = subtitlesAutoTranscriptEl.checked;
    state.autoTranscript = enabled;
    try {
      await api('/api/set_auto_transcript', 'POST', { auto_transcript: enabled ? 'on' : 'off' });
    } catch (err) {
      setStatus('Failed to update Auto Transcript setting: ' + String(err));
      subtitlesAutoTranscriptEl.checked = !enabled;
      state.autoTranscript = !enabled;
    }
  });
}
if (subtitlesGenerateBtnEl) {
  subtitlesGenerateBtnEl.addEventListener('click', generateSubtitlesFromWhisper);
}
if (peopleClearBtnEl) {
  peopleClearBtnEl.addEventListener('click', clearPeopleEntries);
}
if (peopleEditorEl) {
  peopleEditorEl.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const deleteBtn = target.closest('[data-people-row-delete]');
    if (!deleteBtn) return;
    syncPeopleProfileFromEditor(false);
    const idx = Number(deleteBtn.getAttribute('data-people-row-delete'));
    const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
    if (Number.isFinite(idx) && idx >= 0 && idx < entries.length) {
      entries.splice(idx, 1);
      state.peopleProfile = {
        ...(state.peopleProfile || {}),
        entries: canonicalizePeopleEntries(entries),
      };
      refreshPeopleEditorFromState();
      updateReviewStatsDisplay();
    }
    event.preventDefault();
  });
  peopleEditorEl.addEventListener('keydown', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    if (event.key !== 'Enter') return;
    target.blur();
    event.preventDefault();
  });
  peopleEditorEl.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    setTimeout(() => { syncPeopleProfileFromEditor(false); updateReviewStatsDisplay(); }, 0);
  });
  peopleEditorEl.addEventListener('focusout', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    setTimeout(() => { syncPeopleProfileFromEditor(false); }, 0);
  });
}
if (subtitlesEditorEl) {
  subtitlesEditorEl.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const deleteBtn = target.closest('[data-subtitle-row-delete]');
    if (!deleteBtn) return;
    syncSubtitlesProfileFromEditor(false);
    const idx = Number(deleteBtn.getAttribute('data-subtitle-row-delete'));
    if (Number.isFinite(idx)) {
      deleteSubtitleTimelineEntry(idx);
    }
    event.preventDefault();
  });
  subtitlesEditorEl.addEventListener('keydown', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    if (event.key !== 'Enter') return;
    target.blur();
    event.preventDefault();
  });
  subtitlesEditorEl.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    setTimeout(() => { syncSubtitlesProfileFromEditor(false); }, 0);
  });
  subtitlesEditorEl.addEventListener('focusout', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    setTimeout(() => { syncSubtitlesProfileFromEditor(false); }, 0);
  });
  subtitlesEditorEl.addEventListener('scroll', () => {
    if (subtitleEditorProgrammaticScroll) {
      subtitleEditorProgrammaticScroll = false;
      return;
    }
    subtitleEditorUserScrolling = true;
    clearTimeout(_subtitleEditorUserScrollTimer);
    _subtitleEditorUserScrollTimer = setTimeout(() => { subtitleEditorUserScrolling = false; }, 600);
    const rows = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
    const rowEls = Array.from(subtitlesEditorEl.querySelectorAll('tr[data-sub-row="1"]'));
    if (!rows.length || !rowEls.length) return;
    const headerEl = subtitlesEditorEl.querySelector('thead');
    const headerHeight = headerEl ? Number(headerEl.getBoundingClientRect().height || 0) : 0;
    const viewTop = Number(subtitlesEditorEl.scrollTop || 0) + headerHeight;
    let visibleIdx = 0;
    for (let i = 0; i < rowEls.length; i += 1) {
      if (Number(rowEls[i].offsetTop || 0) >= viewTop - 1) {
        visibleIdx = i;
        break;
      }
      visibleIdx = i;
    }
    const row = rows[visibleIdx];
    if (!row) return;
    const sec = Number(row.start_seconds);
    if (!Number.isFinite(sec)) return;
    scrubTimelineToIndex(scrubIndexFromChapterSeconds(sec), { scrollGrid: true });
  });
}
if (splitEditorEl) {
  splitEditorEl.addEventListener('keydown', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    if (!target.hasAttribute('data-split-field')) return;
    if (event.key !== 'Enter') return;
    const rowEl = target.closest('tr[data-split-row="1"]');
    const rowIdx = rowEl ? Math.max(0, Math.trunc(Number(rowEl.getAttribute('data-split-row-idx') || 0))) : 0;
    const field = String(target.getAttribute('data-split-field') || '');
    if (field === 'start' && rowEl) {
      const endInput = rowEl.querySelector('[data-split-field="end"]');
      if (endInput instanceof HTMLInputElement && !String(endInput.value || '').trim()) {
        endInput.focus();
        endInput.select();
        event.preventDefault();
        return;
      }
    }
    const ok = syncSplitProfileFromEditor(true);
    if (ok) {
      refreshSplitEditorFromState();
      const nextStart = splitEditorEl.querySelector(
        `tr[data-split-row="1"][data-split-row-idx="${rowIdx + 1}"] [data-split-field="start"]`
      );
      if (nextStart instanceof HTMLInputElement) {
        nextStart.focus();
        nextStart.select();
      }
    }
    event.preventDefault();
  });
  splitEditorEl.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    if (!target.hasAttribute('data-split-field')) return;
    syncSplitProfileFromEditor(false);
  });
  splitEditorEl.addEventListener('focusout', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains('subtitles-editor-cell')) return;
    if (!target.hasAttribute('data-split-field')) return;
    syncSplitProfileFromEditor(false);
  });
}
if (saveProgressDraftBtnEl) {
  saveProgressDraftBtnEl.addEventListener('click', async () => {
    await saveProgress({ showOverlay: true, loadingText: 'Saving wizard progress...', silent: false });
  });
}
if (simulateFreezeFrameEl) {
  simulateFreezeFrameEl.addEventListener('change', () => {
    setSimFreezeFrame(Boolean(simulateFreezeFrameEl.checked));
  });
}
if (forceAllFramesGoodEl) {
  forceAllFramesGoodEl.addEventListener('change', async () => {
    await applyForceAllFramesGood(Boolean(forceAllFramesGoodEl.checked));
  });
}
if (flipbookSimFreezeFrameEl) {
  flipbookSimFreezeFrameEl.addEventListener('change', () => {
    setSimFreezeFrame(Boolean(flipbookSimFreezeFrameEl.checked));
  });
}
if (timelineScrubEl) {
  timelineScrubEl.addEventListener('input', () => {
    scrubTimelineToIndex(timelineScrubEl.value);
  });
  timelineScrubEl.addEventListener('change', () => {
    scrubTimelineToIndex(timelineScrubEl.value);
  });
}
if (timelineAudioPlayBtnEl) {
  timelineAudioPlayBtnEl.addEventListener('click', async () => {
    await toggleTimelineAudioPlayback();
  });
}
if (timelineAudioTrackEl) {
  timelineAudioTrackEl.addEventListener('pointerdown', (event) => {
    beginTimelineAudioScrub(event);
  });
  timelineAudioTrackEl.addEventListener('pointermove', (event) => {
    moveTimelineAudioScrub(event);
  });
  timelineAudioTrackEl.addEventListener('pointerup', (event) => {
    endTimelineAudioScrub(event);
  });
  timelineAudioTrackEl.addEventListener('pointercancel', (event) => {
    endTimelineAudioScrub(event);
  });
  timelineAudioTrackEl.addEventListener('lostpointercapture', () => {
    endTimelineAudioScrub();
  });
}
if (timelineAudioEl) {
  timelineAudioEl.addEventListener('play', () => {
    updateTimelineAudioPlayButton();
  });
  timelineAudioEl.addEventListener('pause', () => {
    updateTimelineAudioPlayButton();
  });
  timelineAudioEl.addEventListener('ended', () => {
    updateTimelineAudioPlayButton();
  });
  timelineAudioEl.addEventListener('timeupdate', () => {
    if (!isTimelineStepActive()) return;
    if (timelineAudioScrubActive) return;
    const sec = Number(timelineAudioEl.currentTime || 0);
    if (!Number.isFinite(sec)) return;
    const idx = scrubIndexFromChapterSeconds(sec);
    scrubTimelineToIndex(idx, { scrollGrid: true, forceMeta: true });
  });
}
if (peopleTimelineEl) {
  peopleTimelineEl.addEventListener('click', (event) => {
    if (!isTimelineStepActive()) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const zoomBtn = target.closest('.people-timeline-zoom');
    if (zoomBtn) {
      const zoomIn = target.closest('[data-zoom="in"]');
      const zoomOut = target.closest('[data-zoom="out"]');
      const zoomReset = target.closest('[data-zoom="reset"]');
      if (zoomIn) {
        nudgePeopleTimelineZoom(1);
      } else if (zoomOut) {
        nudgePeopleTimelineZoom(-1);
      } else if (zoomReset) {
        setPeopleTimelineZoom(1.0);
      }
      event.preventDefault();
      return;
    }
    if (isSplitStepActive()) {
      return;
    }
    const subtitleDeleteBtn = target.closest('.subtitle-timeline-delete');
    if (subtitleDeleteBtn) {
      const bar = subtitleDeleteBtn.closest('.subtitle-timeline-bar');
      const idx = bar ? Number(bar.dataset.subtitleIndex) : NaN;
      if (Number.isFinite(idx)) {
        deleteSubtitleTimelineEntry(idx);
      }
      event.preventDefault();
      return;
    }
    const subtitleBar = target.closest('.subtitle-timeline-bar');
    if (subtitleBar) {
      // Some browsers/input devices fail to emit dblclick reliably here.
      // Fall back to the second click signal so subtitle rename still works.
      if (Number(event.detail || 0) >= 2) {
        const idx = Number(subtitleBar.dataset.subtitleIndex);
        if (Number.isFinite(idx)) {
          event.preventDefault();
          editSubtitleTimelineEntry(idx);
        }
      }
      return;
    }
    const deleteBtn = target.closest('.people-timeline-delete');
    if (deleteBtn) {
      const bar = deleteBtn.closest('.people-timeline-bar');
      const idx = bar ? Number(bar.dataset.entryIndex) : NaN;
      if (Number.isFinite(idx)) {
        deletePeopleTimelineEntry(idx);
      }
      event.preventDefault();
      return;
    }
    if (target.closest('.people-timeline-input-wrap')) return;
    if (target.closest('.people-timeline-bar')) return;
    if (target.closest('[data-subtitle-region="1"]')) return;
    if (peopleTimelineDrag) return;
    if (peopleTimelineDraft) {
      commitPeopleTimelineDraft();
    }
    beginPeopleTimelineDraftFromPoint(event.clientX, event.clientY);
  });

  peopleTimelineEl.addEventListener('dblclick', (event) => {
    if (!isTimelineStepActive()) return;
    if (isSplitStepActive()) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.closest('.subtitle-timeline-delete')) return;
    const subtitleBar = target.closest('.subtitle-timeline-bar');
    if (subtitleBar) {
      const idx = Number(subtitleBar.dataset.subtitleIndex);
      if (Number.isFinite(idx)) {
        event.preventDefault();
        editSubtitleTimelineEntry(idx);
      }
      return;
    }
    const bar = target.closest('.people-timeline-bar');
    if (!bar) return;
    const idx = Number(bar.dataset.entryIndex);
    if (!Number.isFinite(idx)) return;
    event.preventDefault();
    peopleTimelineDrag = null;
    openPeopleTimelineEdit(idx);
  });

  peopleTimelineEl.addEventListener('pointerdown', (event) => {
    if (!isPeopleStepActive()) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (
      target.closest('.people-timeline-input-wrap')
      || target.closest('.people-timeline-delete')
      || target.closest('.people-timeline-zoom')
      || target.closest('.subtitle-timeline-bar')
      || target.closest('.subtitle-timeline-delete')
    ) return;
    const bar = target.closest('.people-timeline-bar');
    if (!bar) return;
    const idx = Number(bar.dataset.entryIndex);
    if (!Number.isFinite(idx)) return;
    const grip = target.closest('.people-timeline-grip');
    const resizeKind = grip ? String(grip.dataset.resize || '') : '';
    const kind = resizeKind === 'start' ? 'resize-start' : (resizeKind === 'end' ? 'resize-end' : 'move');
    startPeopleTimelineDrag(event, kind, idx);
  });

  peopleTimelineEl.addEventListener('pointermove', (event) => {
    updatePeopleTimelineDrag(event);
  });

  peopleTimelineEl.addEventListener('pointerup', (event) => {
    finishPeopleTimelineDrag(event);
  });

  peopleTimelineEl.addEventListener('pointercancel', (event) => {
    finishPeopleTimelineDrag(event);
  });

  peopleTimelineEl.addEventListener('lostpointercapture', () => {
    finishPeopleTimelineDrag();
  });

  peopleTimelineEl.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (target.classList.contains('subtitle-timeline-input')) {
      if (!subtitleTimelineDraft) return;
      subtitleTimelineDraft.text = target.value || '';
      return;
    }
    if (!target.classList.contains('people-timeline-input')) return;
    if (!peopleTimelineDraft) return;
    peopleTimelineDraft.text = target.value || '';
  });

  peopleTimelineEl.addEventListener('keydown', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (target.classList.contains('subtitle-timeline-input')) {
      if (!subtitleTimelineDraft) return;
      if (event.key === 'Enter') {
        event.preventDefault();
        subtitleTimelineDraft.text = target.value || '';
        commitSubtitleTimelineDraft();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        clearSubtitleTimelineDraft();
      }
      return;
    }
    if (!target.classList.contains('people-timeline-input')) return;
    if (!peopleTimelineDraft) return;
    if (event.key === 'Enter') {
      event.preventDefault();
      peopleTimelineDraft.text = target.value || '';
      commitPeopleTimelineDraft();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      clearPeopleTimelineDraft();
    }
  });

  peopleTimelineEl.addEventListener('focusout', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains('subtitle-timeline-input')) return;
    if (!subtitleTimelineDraft) return;
    subtitleTimelineDraft.text = target.value || '';
    commitSubtitleTimelineDraft();
  });
}

frameGridEl.addEventListener('click', (event) => {
  const card = event.target.closest('.frame-card');
  if (!card) return;
  const fid = String(card.dataset.fid || '').trim();
  const fidNum = Math.trunc(Number(fid));
  if (!Number.isFinite(fidNum)) return;
  const loaded = String(card.dataset.loaded || '') === '1';
  const flipbookOn = Boolean(flipbookPreviewEl && flipbookPreviewEl.classList.contains('active'));
  if (sparkPlayTimer) {
    _pauseSparkWindowPlayback();
  }
  if (flipbookOn && fid && loaded) {
    seekFlipbookToFrameId(fid, { keepVisible: false });
  }
  if (isTimelineStepActive()) {
    if (isPeopleStepActive() || isSubtitlesStepActive()) {
      const timecode = chapterTimecodeFromFid(fidNum);
      const editorEl = isPeopleStepActive() ? peopleEditorEl : subtitlesEditorEl;
      const rowSelector = isPeopleStepActive() ? 'tr[data-people-row="1"]' : 'tr[data-sub-row="1"]';
      const fieldAttr = isPeopleStepActive() ? 'data-people-field' : 'data-sub-field';
      if (editorEl && timecode) {
        const rows = Array.from(editorEl.querySelectorAll(rowSelector));
        let targetRow = null;
        for (let i = rows.length - 1; i >= 0; i--) {
          const startInput = rows[i].querySelector(`[${fieldAttr}="start"]`);
          const endInput = rows[i].querySelector(`[${fieldAttr}="end"]`);
          if ((startInput && !startInput.value.trim()) || (endInput && !endInput.value.trim())) {
            targetRow = rows[i];
            break;
          }
        }
        if (targetRow) {
          const startInput = targetRow.querySelector(`[${fieldAttr}="start"]`);
          const endInput = targetRow.querySelector(`[${fieldAttr}="end"]`);
          if (startInput && !startInput.value.trim()) {
            startInput.value = timecode;
            startInput.dispatchEvent(new Event('change', { bubbles: true }));
          } else if (endInput && !endInput.value.trim()) {
            endInput.value = timecode;
            endInput.dispatchEvent(new Event('change', { bubbles: true }));
          }
        }
      }
    }
    const idx = (state.review && Array.isArray(state.review.frames))
      ? state.review.frames.findIndex(f => String(f && f.fid) === String(fidNum))
      : -1;
    if (idx >= 0) {
      scrubTimelineToIndex(idx, { scrollGrid: false, forceMeta: true });
    }
    return;
  }
  if (!isBadFrameStepActive()) {
    return;
  }
  if (!loaded) {
    setStatus(`Frame ${chapterTimecodeFromFid(fidNum)} has not loaded yet.`);
    return;
  }
  if (state.forceAllFramesGood) {
    setStatus('Disable Force All Good before editing frame statuses.', true);
    return;
  }
  if (event.shiftKey) {
    if (shiftRangeAnchorFid === null || !reviewHasFrameFid(shiftRangeAnchorFid)) {
      shiftRangeAnchorFid = fidNum;
      setStatus(`Range anchor set at ${chapterTimecodeFromFid(fidNum)}. Shift+click another frame to mark the range bad.`);
      return;
    }
    const anchor = Math.trunc(Number(shiftRangeAnchorFid));
    shiftRangeAnchorFid = fidNum;
    const showOverlay = !isChapterLoadInFlight && !(page2El && page2El.classList.contains('flipbook-focus'));
    trackPendingToggleRequest(setFrameRangeStatus(anchor, fidNum, 'bad', { showOverlay }));
    return;
  }
  shiftRangeAnchorFid = fidNum;
  const showOverlay = !isChapterLoadInFlight && !(page2El && page2El.classList.contains('flipbook-focus'));
  trackPendingToggleRequest(toggleFrame(fid, { showOverlay, optimistic: isChapterLoadInFlight }));
});
frameGridEl.addEventListener('scroll', () => {
  scheduleVisibleRangeRefresh();
});
if (flipbookSubtitleRailEl) {
  flipbookSubtitleRailEl.addEventListener('wheel', () => {
    markFlipbookSubtitleRailManual(1400);
  }, { passive: true });
  flipbookSubtitleRailEl.addEventListener('pointerdown', () => {
    markFlipbookSubtitleRailManual(1400);
  });
  flipbookSubtitleRailEl.addEventListener('touchstart', () => {
    markFlipbookSubtitleRailManual(1400);
  }, { passive: true });
  flipbookSubtitleRailEl.addEventListener('scroll', () => {
    if (flipbookSubtitleRailProgrammaticScroll) return;
    markFlipbookSubtitleRailManual(900);
  }, { passive: true });
  flipbookSubtitleRailEl.addEventListener('click', (e) => {
    const row = e.target.closest('[data-subtitle-start-seconds]');
    if (!row) return;
    const startSeconds = parseFloat(row.getAttribute('data-subtitle-start-seconds'));
    if (!Number.isFinite(startSeconds)) return;
    const frameIdx = flipbookIndexFromChapterSeconds(startSeconds);
    renderSparkPlaybackFrame(frameIdx, { keepVisible: true });
  });
}
overlayCancelBtnEl.addEventListener('click', async () => {
  setLoadCancelUi(true, true, 'Cancelling...');
  try {
    let result = null;
    if (activeCancelableTask === 'subtitles' || isSubtitlesGenerateInFlight) {
      result = await api('/api/cancel_subtitles', 'POST', {});
    } else if (activeCancelableTask === 'load' || Boolean(loadProgressPollTimer)) {
      result = await api('/api/cancel_load', 'POST', {});
    } else {
      setLoadCancelUi(false);
      return;
    }
    if (result && result.message) {
      overlayMsgEl.textContent = String(result.message);
    } else {
      overlayMsgEl.textContent = 'Cancelling...';
    }
  } catch (_err) {
    overlayMsgEl.textContent = 'Cancelling...';
  }
});
if (helpBtnEl) {
  helpBtnEl.addEventListener('click', openHelpModal);
}
if (helpCloseBtnEl) {
  helpCloseBtnEl.addEventListener('click', closeHelpModal);
}
if (helpModalEl) {
  helpModalEl.addEventListener('click', (event) => {
    if (event.target === helpModalEl) {
      closeHelpModal();
    }
  });
}
iqrSparkEl.addEventListener('pointerdown', (event) => {
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) return;
  event.preventDefault();
  sparkDragPointerId = event.pointerId;
  iqrSparkEl.classList.add('dragging');
  if (iqrSparkEl.setPointerCapture) {
    try {
      iqrSparkEl.setPointerCapture(event.pointerId);
    } catch (_err) {}
  }
  queueSparkDragSeek(event.clientX);
});
iqrSparkEl.addEventListener('pointermove', (event) => {
  if (sparkDragPointerId === null || event.pointerId !== sparkDragPointerId) return;
  event.preventDefault();
  queueSparkDragSeek(event.clientX);
});
iqrSparkEl.addEventListener('pointerup', (event) => {
  if (sparkDragPointerId === null || event.pointerId !== sparkDragPointerId) return;
  event.preventDefault();
  queueSparkDragSeek(event.clientX);
  endSparkDrag();
});
iqrSparkEl.addEventListener('pointercancel', () => {
  endSparkDrag();
});
iqrSparkEl.addEventListener('lostpointercapture', () => {
  endSparkDrag();
});

window.addEventListener('load', async () => {
  iqrLabelEl.textContent = Number(iqrEl.value).toFixed(2);
  if (gammaLabelEl) gammaLabelEl.textContent = Number(gammaLevelEl.value || 1).toFixed(2);
  renderStepPills();
  setStep(state.wizardStep || STEP_FIRST.num);
  updateSparkPlayButton();
  updateFullscreenButton();
  updateActionLocks();
  updateGammaControls();
  resetTimelineAudioState();
  resetFlipbookAudioState();
  applyFlipbookVolume();
  refreshPeopleEditorFromState();
  refreshSubtitlesEditorFromState();
  refreshSplitEditorFromState();
  updateTimelineScrubUi();
  setSimFreezeFrame(state.simulateFreezeFrame);
  setForceAllFramesGoodUi(state.forceAllFramesGood);
  setStatus('Loading archives...');
  try {
    await loadArchives();
  } catch (err) {
    setStatus(err.message, true);
  }
});

window.addEventListener('resize', () => {
  updatePeopleStepLayoutSizing();
  drawTimelineAudioWaveform();
  if (state.review && Array.isArray(state.review.frames) && state.review.frames.length) {
    // Skip the grid re-render while the flipbook is in focus mode: the grid is not visible
    // and the resize is typically from entering/exiting fullscreen. The grid will be
    // re-rendered correctly when the flipbook closes (exitReviewFullscreenIfActive fires
    // another resize at that point).
    if (!isFlipbookFocusModeActive()) {
      scheduleVisibleRangeRefresh();
    }
  }
});
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && helpModalEl && helpModalEl.classList.contains('active')) {
    event.preventDefault();
    closeHelpModal();
  }
});

sparkPlayBtnEl.addEventListener('click', openFlipbookPanel);
if (flipbookPlayBtnEl) {
  flipbookPlayBtnEl.addEventListener('click', () => {
    toggleSparkWindowPlayback();
  });
}
if (flipbookRevBtnEl) {
  flipbookRevBtnEl.addEventListener('click', () => {
    if (!sparkPlayFrames.length) {
      sparkPlayFrames = _collectFlipbookFramesFromGrid();
      sparkPlayIndex = 0;
    }
    if (!sparkPlayFrames.length) return;
    setFlipbookVisible(true);
    _pauseSparkWindowPlayback();
    stepSparkWindowLeft();
  });
}
if (flipbookFwdBtnEl) {
  flipbookFwdBtnEl.addEventListener('click', () => {
    if (!sparkPlayFrames.length) {
      sparkPlayFrames = _collectFlipbookFramesFromGrid();
      sparkPlayIndex = 0;
    }
    if (!sparkPlayFrames.length) return;
    setFlipbookVisible(true);
    _pauseSparkWindowPlayback();
    stepSparkWindowRight();
  });
}
if (flipbookCloseBtnEl) {
  flipbookCloseBtnEl.addEventListener('click', () => {
    closeFlipbookPanel();
  });
}
if (flipbookFrameEl) {
  flipbookFrameEl.addEventListener('click', () => {
    toggleCurrentFlipbookFrame();
  });
}
if (flipbookVolumeEl) {
  flipbookVolumeEl.addEventListener('input', () => {
    applyFlipbookVolume();
  });
}
if (flipbookAudioEl) {
  flipbookAudioEl.addEventListener('loadedmetadata', () => {
    if (!sparkPlayTimer) {
      syncFlipbookAudioToCurrentFrame();
    }
  });
  flipbookAudioEl.addEventListener('ended', () => {
    if (!sparkPlayTimer) return;
    if (!flipbookAudioEl) return;
    try {
      flipbookAudioEl.currentTime = 0;
    } catch (_err) {}
    flipbookAudioEl.play().then(() => {
      sparkPlayUseAudioClock = true;
    }).catch(() => {
      sparkPlayUseAudioClock = false;
    });
  });
  flipbookAudioEl.addEventListener('error', () => {
    sparkPlayUseAudioClock = false;
  });
}

function normalizeWheelToPixels(event) {
  const mode = Number(event.deltaMode || 0);
  const dy = Number(event.deltaY || 0);
  const dx = Number(event.deltaX || 0);
  const dominant = Math.abs(dy) >= Math.abs(dx) ? dy : dx;
  if (!dominant) return 0;
  if (mode === 1) return dominant * 16;
  if (mode === 2) return dominant * Math.max(320, window.innerHeight || 800);
  return dominant;
}

function relayWheelToFrameGrid(event) {
  if (!isReviewStepActive()) return;
  if (!frameGridEl) return;
  if (event.target && event.target.closest('input, textarea, select, button')) return;
  if (event.target && event.target.closest('.subtitles-editor')) return;

  const deltaPx = normalizeWheelToPixels(event);
  if (deltaPx === 0) return;
  if (event.cancelable) event.preventDefault();
  frameGridEl.scrollBy({ top: deltaPx * 1.75, behavior: 'auto' });
  scheduleVisibleRangeRefresh();
}

window.addEventListener('wheel', relayWheelToFrameGrid, { passive: false, capture: true });

function handleFlipbookArrowStep(event) {
  if (!isReviewStepActive()) return false;
  if (!flipbookPreviewEl || !flipbookPreviewEl.classList.contains('active')) return false;
  if (event.target && event.target.closest('input, textarea, select, button, [contenteditable="true"]')) return false;

  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return false;

  if (!sparkPlayFrames.length) {
    sparkPlayFrames = _collectFlipbookFramesFromGrid();
    sparkPlayIndex = 0;
  }
  if (!sparkPlayFrames.length) return false;

  event.preventDefault();
  _pauseSparkWindowPlayback();
  if (event.key === 'ArrowLeft') {
    stepSparkWindowLeft();
  } else {
    stepSparkWindowRight();
  }
  return true;
}

window.addEventListener('keydown', (event) => {
  handleFlipbookArrowStep(event);
});
