function chapterAudioCacheKey() {
  const span = chapterFrameSpan();
  const offset = Number(state.audioSyncOffset || 0);
  return [
    String(state.archive || '').trim(),
    String(state.chapter || '').trim(),
    String(span.start),
    String(span.end),
    Number.isFinite(offset) ? offset.toFixed(4) : '0.0000',
  ].join('|');
}

function chapterSecondsFromScrubIndex(rawIndex) {
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (!frames.length) return 0;
  const idx = _clamp(Number(rawIndex || 0), 0, frames.length - 1);
  return chapterLocalSecondsFromFid(frames[idx] && frames[idx].fid);
}

function scrubIndexFromChapterSeconds(rawSeconds) {
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (!frames.length) return 0;
  if (frames.length === 1) return 0;
  const target = _clamp(Number(rawSeconds || 0), 0, chapterDurationSeconds());
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const midSec = chapterLocalSecondsFromFid(frames[mid] && frames[mid].fid);
    if (midSec < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  let idx = _clamp(lo, 0, frames.length - 1);
  if (idx > 0) {
    const prevSec = chapterLocalSecondsFromFid(frames[idx - 1] && frames[idx - 1].fid);
    const curSec = chapterLocalSecondsFromFid(frames[idx] && frames[idx].fid);
    if (Math.abs(prevSec - target) <= Math.abs(curSec - target)) {
      idx -= 1;
    }
  }
  return idx;
}

function updateTimelineAudioPlayButton() {
  if (!timelineAudioPlayBtnEl) return;
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  const canUse = isTimelineStepActive() && frames.length > 0;
  timelineAudioPlayBtnEl.disabled = !canUse;
  if (!canUse) {
    timelineAudioPlayBtnEl.textContent = '▶';
    return;
  }
  const playing = Boolean(timelineAudioEl && !timelineAudioEl.paused);
  timelineAudioPlayBtnEl.textContent = playing ? '||' : '▶';
  timelineAudioPlayBtnEl.title = playing ? 'Pause timeline audio' : 'Play timeline audio';
}

function updateTimelineAudioPlayheadFromIndex(rawIndex) {
  if (!timelineAudioPlayheadEl || !timelineAudioTrackEl) return;
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (!frames.length) {
    timelineAudioPlayheadEl.style.left = '0%';
    timelineAudioTrackEl.setAttribute('aria-valuemin', '0');
    timelineAudioTrackEl.setAttribute('aria-valuemax', '0');
    timelineAudioTrackEl.setAttribute('aria-valuenow', '0');
    return;
  }
  const idx = _clamp(Number(rawIndex || 0), 0, frames.length - 1);
  const ratio = frames.length <= 1 ? 0 : (idx / Math.max(1, frames.length - 1));
  timelineAudioPlayheadEl.style.left = `${(ratio * 100).toFixed(3)}%`;
  timelineAudioTrackEl.setAttribute('aria-valuemin', '1');
  timelineAudioTrackEl.setAttribute('aria-valuemax', String(frames.length));
  timelineAudioTrackEl.setAttribute('aria-valuenow', String(idx + 1));
}

function sampleTimelineAudioPeak(rawSeconds) {
  const peaks = Array.isArray(timelineAudioWavePeaks) ? timelineAudioWavePeaks : [];
  if (!peaks.length) return 0;
  const duration = Math.max(0.001, Number(timelineAudioWaveDurationSec || chapterDurationSeconds() || 0.001));
  const sec = _clamp(Number(rawSeconds || 0), 0, duration);
  const pos = (sec / duration) * Math.max(0, peaks.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(peaks.length - 1, lo + 1);
  const t = pos - lo;
  const a = Number(peaks[lo] || 0);
  const b = Number(peaks[hi] || a);
  return _clamp(a + ((b - a) * t), 0, 1);
}

function drawTimelineAudioWaveform() {
  if (!timelineAudioWaveEl) return;
  const canvas = timelineAudioWaveEl;
  const widthCss = Math.max(1, Math.floor(canvas.clientWidth || 1));
  const heightCss = Math.max(1, Math.floor(canvas.clientHeight || 1));
  const dpr = Math.max(1, Number(window.devicePixelRatio || 1));
  const width = Math.max(1, Math.floor(widthCss * dpr));
  const height = Math.max(1, Math.floor(heightCss * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, widthCss, heightCss);
  ctx.fillStyle = 'rgba(6, 11, 18, 0.68)';
  ctx.fillRect(0, 0, widthCss, heightCss);

  const midY = heightCss / 2;
  ctx.strokeStyle = 'rgba(149, 166, 188, 0.28)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(widthCss, midY);
  ctx.stroke();

  const peaks = Array.isArray(timelineAudioWavePeaks) ? timelineAudioWavePeaks : [];
  if (!peaks.length) return;
  const maxAmp = Math.max(2, Math.floor((heightCss * 0.5) - 2));
  ctx.strokeStyle = 'rgba(102, 198, 245, 0.85)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < widthCss; x += 1) {
    const frac = widthCss <= 1 ? 0 : (x / (widthCss - 1));
    const idx = Math.floor(frac * Math.max(0, peaks.length - 1));
    const amp = _clamp(Number(peaks[idx] || 0), 0, 1);
    const h = Math.max(1, amp * maxAmp);
    ctx.moveTo(x + 0.5, midY - h);
    ctx.lineTo(x + 0.5, midY + h);
  }
  ctx.stroke();
}

function computeWaveformPeaks(samples, bucketCount) {
  const out = [];
  const arr = samples instanceof Float32Array ? samples : new Float32Array(0);
  const buckets = Math.max(64, Math.min(4096, Math.trunc(Number(bucketCount || 0)) || 1024));
  if (!arr.length) {
    for (let i = 0; i < buckets; i += 1) out.push(0);
    return out;
  }
  const step = Math.max(1, Math.floor(arr.length / buckets));
  for (let i = 0; i < buckets; i += 1) {
    const from = i * step;
    const to = Math.min(arr.length, from + step);
    let peak = 0;
    for (let j = from; j < to; j += 1) {
      const v = Math.abs(Number(arr[j] || 0));
      if (v > peak) peak = v;
    }
    out.push(_clamp(peak, 0, 1));
  }
  return out;
}

function resetTimelineAudioState() {
  if (timelineAudioEl) {
    try {
      timelineAudioEl.pause();
    } catch (_err) {}
    timelineAudioEl.removeAttribute('src');
  }
  timelineAudioWavePeaks = [];
  timelineAudioWaveDurationSec = 0;
  timelineAudioWaveKey = '';
  timelineAudioWaveLoading = false;
  timelineAudioScrubPointerId = null;
  timelineAudioScrubActive = false;
  timelineAudioScrubWasPlaying = false;
  timelineAudioLastScrubSeconds = 0;
  timelineAudioLastScrubAt = 0;
  drawTimelineAudioWaveform();
  updateTimelineAudioPlayButton();
  updateTimelineAudioPlayheadFromIndex(0);
}

async function ensureTimelineAudioLoaded(force = false) {
  if (!timelineAudioEl || !timelineAudioTrackEl || !state.archive || !state.chapter) return false;
  const key = chapterAudioCacheKey();
  if (!force && timelineAudioWaveKey === key && timelineAudioWavePeaks.length) {
    drawTimelineAudioWaveform();
    return true;
  }
  if (timelineAudioWaveLoading) return false;
  timelineAudioWaveLoading = true;
  const audioUrl = `/api/chapter_audio?key=${encodeURIComponent(key)}`;
  try {
    const absoluteAudioUrl = new URL(audioUrl, window.location.href).toString();
    if (String(timelineAudioEl.src || '') !== absoluteAudioUrl) {
      timelineAudioEl.src = absoluteAudioUrl;
      timelineAudioEl.preload = 'auto';
    }
    const resp = await fetch(audioUrl, { cache: 'no-store' });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const audioBytes = await resp.arrayBuffer();
    if (!timelineAudioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (Ctx) timelineAudioCtx = new Ctx();
    }
    if (!timelineAudioCtx) {
      throw new Error('AudioContext unavailable');
    }
    try {
      if (timelineAudioCtx.state === 'suspended') {
        await timelineAudioCtx.resume();
      }
    } catch (_err) {}
    const decoded = await timelineAudioCtx.decodeAudioData(audioBytes.slice(0));
    const channel = decoded.numberOfChannels > 0 ? decoded.getChannelData(0) : new Float32Array(0);
    const bucketCount = Math.max(256, Math.min(2800, Math.floor((timelineAudioTrackEl.clientWidth || 400) * 4)));
    timelineAudioWavePeaks = computeWaveformPeaks(channel, bucketCount);
    timelineAudioWaveDurationSec = Math.max(0, Number(decoded.duration || 0));
    timelineAudioWaveKey = key;
    drawTimelineAudioWaveform();
    if (isTimelineStepActive()) {
      renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
    }
    return true;
  } catch (_err) {
    timelineAudioWavePeaks = [];
    timelineAudioWaveDurationSec = 0;
    timelineAudioWaveKey = '';
    drawTimelineAudioWaveform();
    return false;
  } finally {
    timelineAudioWaveLoading = false;
  }
}

function peopleTimelineAudioUnderlayHtml(
  windowStart,
  windowSpan,
  topPad,
  laneHeight,
  peopleLaneCount,
  subtitleTopPad,
  subtitleLaneCount
) {
  if (!timelineAudioWavePeaks.length || windowSpan <= 0) return '';
  const regionTop = subtitleLaneCount > 0 ? subtitleTopPad : topPad;
  const regionHeight = subtitleLaneCount > 0
    ? Math.max(laneHeight, subtitleLaneCount * laneHeight)
    : Math.max(laneHeight, peopleLaneCount * laneHeight);
  const center = regionTop + (regionHeight / 2);
  const bars = 180;
  const barHtml = [];
  for (let i = 0; i < bars; i += 1) {
    const ratio = bars <= 1 ? 0 : (i / (bars - 1));
    const sec = Number(windowStart) + (ratio * Number(windowSpan));
    const amp = sampleTimelineAudioPeak(sec);
    const halfHeight = Math.max(1, Math.round(amp * (regionHeight * 0.46)));
    const top = Math.max(regionTop, Math.round(center - halfHeight));
    const fullHeight = Math.min(regionHeight, Math.max(1, halfHeight * 2));
    barHtml.push(
      `<span class="people-timeline-audio-bar" style="left:${(ratio * 100).toFixed(3)}%;top:${top}px;height:${fullHeight}px;"></span>`
    );
  }
  return `<div class="people-timeline-audio-underlay" style="top:${regionTop}px;height:${regionHeight}px;">${barHtml.join('')}</div>`;
}

function timelineAudioSecondsFromClientX(clientX) {
  if (!timelineAudioTrackEl) return null;
  const rect = timelineAudioTrackEl.getBoundingClientRect();
  if (!rect.width || rect.width <= 0) return null;
  const x = _clamp(Number(clientX) - rect.left, 0, rect.width);
  const ratio = rect.width <= 1 ? 0 : (x / rect.width);
  const duration = Math.max(0, chapterDurationSeconds());
  return _clamp(ratio * duration, 0, duration);
}

function previewTimelineScrubAudioAt(seconds) {
  if (!timelineAudioEl) return;
  const secNum = Number(seconds);
  if (!Number.isFinite(secNum)) return;
  const duration = Number(timelineAudioEl.duration || timelineAudioWaveDurationSec || chapterDurationSeconds());
  const maxSec = duration > 0 ? Math.max(0, duration - 0.01) : Math.max(0, chapterDurationSeconds() - 0.01);
  const target = _clamp(secNum, 0, maxSec);
  const now = performance.now();
  if (Math.abs(Number(timelineAudioEl.currentTime || 0) - target) > 0.01) {
    timelineAudioEl.currentTime = target;
  }
  const dt = Math.max(1, now - Number(timelineAudioLastScrubAt || 0));
  const delta = target - Number(timelineAudioLastScrubSeconds || 0);
  const speed = _clamp(Math.abs(delta) / (dt / 1000), 0.5, 3.0);
  timelineAudioEl.playbackRate = speed;
  timelineAudioLastScrubAt = now;
  timelineAudioLastScrubSeconds = target;
  if (timelineAudioEl.paused) {
    timelineAudioEl.play().catch(() => {});
  }
}

function scrubFromTimelineAudioClientX(clientX, options = {}) {
  const sec = timelineAudioSecondsFromClientX(clientX);
  if (!Number.isFinite(sec)) return;
  const idx = scrubIndexFromChapterSeconds(sec);
  scrubTimelineToIndex(idx, { scrollGrid: true, forceMeta: true });
  if (options && options.audible) {
    previewTimelineScrubAudioAt(chapterSecondsFromScrubIndex(idx));
  }
}

function beginTimelineAudioScrub(event) {
  if (!timelineAudioTrackEl || !isTimelineStepActive()) return;
  timelineAudioScrubPointerId = event.pointerId;
  timelineAudioScrubActive = true;
  timelineAudioScrubWasPlaying = Boolean(timelineAudioEl && !timelineAudioEl.paused);
  timelineAudioTrackEl.classList.add('scrubbing');
  timelineAudioLastScrubSeconds = chapterSecondsFromScrubIndex(Number(timelineScrubEl && timelineScrubEl.value || 0));
  timelineAudioLastScrubAt = performance.now();
  if (timelineAudioTrackEl.setPointerCapture) {
    try {
      timelineAudioTrackEl.setPointerCapture(event.pointerId);
    } catch (_err) {}
  }
  scrubFromTimelineAudioClientX(event.clientX, { audible: true });
  event.preventDefault();
}

function moveTimelineAudioScrub(event) {
  if (!timelineAudioScrubActive || timelineAudioScrubPointerId === null) return;
  if (event.pointerId !== timelineAudioScrubPointerId) return;
  scrubFromTimelineAudioClientX(event.clientX, { audible: true });
  event.preventDefault();
}

function endTimelineAudioScrub(event = null) {
  if (event && timelineAudioScrubPointerId !== null && Number(event.pointerId) !== Number(timelineAudioScrubPointerId)) {
    return;
  }
  if (timelineAudioTrackEl) {
    timelineAudioTrackEl.classList.remove('scrubbing');
  }
  timelineAudioScrubActive = false;
  timelineAudioScrubPointerId = null;
  if (timelineAudioEl) {
    timelineAudioEl.playbackRate = 1.0;
  }
  if (!timelineAudioScrubWasPlaying && timelineAudioEl) {
    timelineAudioEl.pause();
  }
  timelineAudioScrubWasPlaying = false;
}

async function toggleTimelineAudioPlayback() {
  if (!timelineAudioEl || !isTimelineStepActive()) return;
  await ensureTimelineAudioLoaded(false);
  const targetSec = chapterSecondsFromScrubIndex(Number(timelineScrubEl && timelineScrubEl.value || 0));
  const duration = Number(timelineAudioEl.duration || timelineAudioWaveDurationSec || chapterDurationSeconds());
  const maxSec = duration > 0 ? Math.max(0, duration - 0.01) : Math.max(0, chapterDurationSeconds() - 0.01);
  const seekTo = _clamp(targetSec, 0, maxSec);
  if (timelineAudioEl.paused) {
    try {
      timelineAudioEl.currentTime = seekTo;
      timelineAudioEl.playbackRate = 1.0;
      await timelineAudioEl.play();
    } catch (_err) {}
  } else {
    timelineAudioEl.pause();
  }
  updateTimelineAudioPlayButton();
}

function setPeopleTimelineZoom(nextZoom) {
  const z = _clamp(Number(nextZoom || 1), 0.25, 12.0);
  peopleTimelineZoom = Number.isFinite(z) ? z : 1.0;
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
}

function nudgePeopleTimelineZoom(direction) {
  const dir = Number(direction);
  if (!Number.isFinite(dir) || dir === 0) return;
  const factor = 1.2;
  const base = Number(peopleTimelineZoom || 1.0);
  const next = dir > 0 ? (base * factor) : (base / factor);
  const clamped = _clamp(next, 0.25, 12.0);
  setPeopleTimelineZoom(clamped);
}

function peopleTimelineSecondsFromClientX(clientX) {
  if (!peopleTimelineEl || !peopleTimelineRenderState) return null;
  const rect = peopleTimelineEl.getBoundingClientRect();
  if (!rect.width || rect.width <= 0) return null;
  const x = _clamp(Number(clientX) - rect.left, 0, rect.width);
  const ratio = rect.width <= 1 ? 0 : (x / rect.width);
  return Number(peopleTimelineRenderState.windowStart || 0) + (ratio * Number(peopleTimelineRenderState.windowSpan || 0));
}

function peopleTimelineLaneFromClientY(clientY) {
  if (!peopleTimelineEl || !peopleTimelineRenderState) return 0;
  const rect = peopleTimelineEl.getBoundingClientRect();
  const topPad = Number(peopleTimelineRenderState.topPad || 20);
  const laneHeight = Math.max(1, Number(peopleTimelineRenderState.laneHeight || 22));
  const rel = Number(clientY) - rect.top - topPad;
  if (!Number.isFinite(rel) || rel <= 0) return 0;
  return Math.max(0, Math.trunc(rel / laneHeight));
}

function openPeopleTimelineDraft(draft) {
  if (!draft || typeof draft !== 'object') return;
  const duration = Math.max(chapterDurationSeconds(), timelineFrameStepSeconds());
  const minDur = timelineFrameStepSeconds();
  const maxStart = Math.max(0, duration - minDur);
  const laneNum = Number(draft.lane);
  const lane = Number.isFinite(laneNum) ? Math.max(0, Math.trunc(laneNum)) : 0;
  const start = _clamp(snapTimelineSeconds(draft.start), 0, maxStart);
  const end = _clamp(
    snapTimelineSeconds(Math.max(Number(draft.end || 0), start + minDur)),
    start + minDur,
    duration
  );
  peopleTimelineDraft = {
    mode: String(draft.mode || 'new') === 'edit' ? 'edit' : 'new',
    entryIndex: Number.isFinite(Number(draft.entryIndex)) ? Math.trunc(Number(draft.entryIndex)) : null,
    lane,
    start,
    end,
    text: String(draft.text || '').trim(),
  };
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
}

function clearPeopleTimelineDraft() {
  if (!peopleTimelineDraft) return;
  peopleTimelineDraft = null;
  renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
}

function commitPeopleTimelineDraft() {
  if (!peopleTimelineDraft) return false;
  const text = String(peopleTimelineDraft.text || '').replace(/\s+/g, ' ').trim();
  const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  if (!text) {
    peopleTimelineDraft = null;
    renderPeopleTimeline(currentPeopleTimelineCursorSeconds());
    return false;
  }

  if (peopleTimelineDraft.mode === 'edit') {
    const idx = Number(peopleTimelineDraft.entryIndex);
    if (Number.isFinite(idx) && idx >= 0 && idx < entries.length) {
      entries[idx] = {
        ...entries[idx],
        people: text,
        lane: Math.max(0, Math.trunc(Number(entries[idx].lane || 0))),
      };
    }
  } else {
    entries.push({
      start_seconds: peopleTimelineDraft.start,
      end_seconds: peopleTimelineDraft.end,
      start: formatTimestampSeconds(peopleTimelineDraft.start),
      end: formatTimestampSeconds(peopleTimelineDraft.end),
      people: text,
      lane: peopleTimelineDraft.lane,
    });
  }
  state.peopleProfile = {
    ...(state.peopleProfile || {}),
    entries: canonicalizePeopleEntries(entries),
  };
  peopleTimelineDraft = null;
  refreshPeopleEditorFromState();
  updateReviewStatsDisplay();
  return true;
}

function renderPeopleTimeline(cursorSeconds = null) {
  if (!peopleTimelineEl) return;
  if (!isTimelineStepActive()) {
    peopleTimelineEl.innerHTML = '';
    peopleTimelineEl.style.height = '';
    peopleTimelineRenderState = null;
    return;
  }

  const splitMode = isSplitStepActive();
  const entries = splitMode
    ? []
    : canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
  const subtitleEntries = splitMode
    ? splitEntriesForTimeline()
    : canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  const duration = Math.max(
    timelineFrameStepSeconds(),
    chapterDurationSeconds(),
    entries.reduce((mx, row) => Math.max(mx, Number(row.end_seconds || 0)), 0),
    subtitleEntries.reduce((mx, row) => Math.max(mx, Number(row.end_seconds || 0)), 0)
  );

  let currentSec = Number(cursorSeconds);
  if (!Number.isFinite(currentSec)) {
    if (frames.length && timelineScrubEl) {
      const idx = _clamp(Number(timelineScrubEl.value || 0), 0, frames.length - 1);
      currentSec = chapterLocalSecondsFromFid(frames[idx] && frames[idx].fid);
    } else {
      currentSec = 0;
    }
  }
  currentSec = _clamp(currentSec, 0, duration);

  const baseViewportSeconds = Math.max(20, Math.min(180, Math.max(30, duration * 0.14)));
  const minViewportSeconds = Math.min(duration, Math.max(timelineFrameStepSeconds() * 3, 0.2));
  const maxViewportSeconds = Math.max(minViewportSeconds, duration);
  let viewportSeconds = baseViewportSeconds / Math.max(0.25, Number(peopleTimelineZoom || 1));
  viewportSeconds = _clamp(viewportSeconds, minViewportSeconds, maxViewportSeconds);
  if (!Number.isFinite(viewportSeconds) || viewportSeconds <= 0) {
    viewportSeconds = Math.max(minViewportSeconds, 0.2);
  }

  const maxWindowStart = Math.max(0, duration - viewportSeconds);
  const priorWindowStart = Number(peopleTimelineRenderState && peopleTimelineRenderState.windowStart);
  let windowStart = Number.isFinite(priorWindowStart)
    ? _clamp(priorWindowStart, 0, maxWindowStart)
    : _clamp(currentSec - (viewportSeconds * 0.35), 0, maxWindowStart);
  let windowEnd = windowStart + viewportSeconds;
  const leadMargin = viewportSeconds * 0.1;
  const tailMargin = viewportSeconds * 0.9;
  if (currentSec < (windowStart + leadMargin)) {
    windowStart = _clamp(currentSec - leadMargin, 0, maxWindowStart);
    windowEnd = windowStart + viewportSeconds;
  } else if (currentSec > (windowStart + tailMargin)) {
    windowStart = _clamp(currentSec - tailMargin, 0, maxWindowStart);
    windowEnd = windowStart + viewportSeconds;
  }
  const windowSpan = Math.max(0.001, windowEnd - windowStart);

  const maxLane = entries.reduce((mx, row) => {
    const lane = Math.max(0, Math.trunc(Number(row.lane || 0)));
    return Math.max(mx, lane);
  }, 0);
  const draftLane = peopleTimelineDraft ? Math.max(0, Math.trunc(Number(peopleTimelineDraft.lane || 0))) : 0;
  const peopleLaneCount = splitMode ? 0 : Math.max(1, maxLane + 1, draftLane + 1);
  const laneHeight = 22;
  const topPad = 20;
  const timelineHeight = Math.max(
    88,
    topPad + peopleLaneCount * laneHeight + 6
  );
  peopleTimelineEl.style.height = `${timelineHeight}px`;

  const tracksHtml = Array.from({ length: peopleLaneCount }, (_, lane) => {
    const top = topPad + lane * laneHeight;
    const laneNum = lane + 1;
    const bgClass = lane % 2 === 1 ? ' alt' : '';
    const labelTop = top + 4;
    return `
      <div class="people-timeline-lane-bg${bgClass}" style="top:${top}px;"></div>
      <div class="people-timeline-track" style="top:${top}px;"></div>
      <div class="people-timeline-lane-label" style="top:${labelTop}px;">L${laneNum}</div>
    `;
  }).join('');
  const audioUnderlayHtml = peopleTimelineAudioUnderlayHtml(
    windowStart,
    windowSpan,
    topPad,
    laneHeight,
    peopleLaneCount,
    topPad + peopleLaneCount * laneHeight,
    0
  );

  const barsHtml = entries.map((row, idx) => {
    const startSec = Number(row.start_seconds);
    const endSec = Number(row.end_seconds);
    if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || endSec <= startSec) return '';
    const clipStart = Math.max(startSec, windowStart);
    const clipEnd = Math.min(endSec, windowEnd);
    if (clipEnd <= clipStart) return '';
    const lane = Math.max(0, Math.trunc(Number(row.lane || 0)));
    const leftPct = _clamp(((clipStart - windowStart) / windowSpan) * 100, 0, 100);
    const rightPct = _clamp(((clipEnd - windowStart) / windowSpan) * 100, leftPct, 100);
    const widthPct = Math.min(100 - leftPct, Math.max(0.8, rightPct - leftPct));
    const top = topPad + (lane * laneHeight) + 2;
    const label = escapeHtml(String(row.people || ''));
    const title = escapeHtml(`${formatTimestampSeconds(startSec)}-${formatTimestampSeconds(endSec)}  ${row.people}`);
    const dragging = peopleTimelineDrag && Number(peopleTimelineDrag.entryIndex) === idx ? ' dragging' : '';
    return `
      <div class="people-timeline-bar${dragging}" data-entry-index="${idx}" data-lane="${lane}" style="left:${leftPct.toFixed(3)}%;width:${widthPct.toFixed(3)}%;top:${top}px;" title="${title}">
        <span class="people-timeline-grip left" data-resize="start" aria-hidden="true"></span>
        <span class="people-timeline-name">${label}</span>
        <button class="people-timeline-delete" type="button" data-delete="1" title="Delete subtitle range">x</button>
        <span class="people-timeline-grip right" data-resize="end" aria-hidden="true"></span>
      </div>
    `;
  }).join('');
  let draftHtml = '';
  if (peopleTimelineDraft) {
    const draftStart = Math.max(windowStart, Number(peopleTimelineDraft.start || 0));
    const draftEnd = Math.min(windowEnd, Number(peopleTimelineDraft.end || 0));
    if (draftEnd > draftStart) {
      const draftLeft = _clamp(((draftStart - windowStart) / windowSpan) * 100, 0, 100);
      const draftRight = _clamp(((draftEnd - windowStart) / windowSpan) * 100, draftLeft, 100);
      const draftWidth = Math.min(100 - draftLeft, Math.max(1.2, draftRight - draftLeft));
      const draftLaneTop = topPad + Math.max(0, Math.trunc(Number(peopleTimelineDraft.lane || 0))) * laneHeight + 1;
      draftHtml = `
        <div class="people-timeline-input-wrap" style="left:${draftLeft.toFixed(3)}%;width:${draftWidth.toFixed(3)}%;top:${draftLaneTop}px;">
          <input class="people-timeline-input" type="text" spellcheck="false" value="${escapeHtml(String(peopleTimelineDraft.text || ''))}" placeholder="Type name(s) and press Enter">
        </div>
      `;
    }
  }

  let dragTooltipHtml = '';
  if (peopleTimelineDrag) {
    const dragIdx = Number(peopleTimelineDrag.entryIndex);
    if (Number.isFinite(dragIdx) && dragIdx >= 0 && dragIdx < entries.length) {
      const dragRow = entries[dragIdx];
      const dragStart = Number(dragRow.start_seconds || 0);
      const dragEnd = Number(dragRow.end_seconds || 0);
      if (Number.isFinite(dragStart) && Number.isFinite(dragEnd) && dragEnd > dragStart) {
        const midSec = _clamp((dragStart + dragEnd) / 2, windowStart, windowEnd);
        const leftPct = _clamp(((midSec - windowStart) / windowSpan) * 100, 0, 100);
        const lane = Math.max(0, Math.trunc(Number(dragRow.lane || 0)));
        const laneTop = topPad + lane * laneHeight;
        const tooltipTop = Math.max(0, laneTop - 16);
        const tooltipText = `${formatTimestampSeconds(dragStart)}-${formatTimestampSeconds(dragEnd)} (${formatTimestampSeconds(dragEnd - dragStart)})`;
        dragTooltipHtml = `<div class="people-timeline-drag-tooltip" style="left:${leftPct.toFixed(3)}%;top:${tooltipTop}px;">${escapeHtml(tooltipText)}</div>`;
      }
    }
  }

  const hintHtml = splitMode
    ? (!subtitleEntries.length
      ? '<div class="people-timeline-empty">Add chapter rows in the table below to create chapter ranges.</div>'
      : '')
    : (!entries.length
      ? '<div class="people-timeline-empty">Click timeline to add a name (Enter to save, drag/resize after).</div>'
      : '');

  const playheadPct = _clamp(((currentSec - windowStart) / windowSpan) * 100, 0, 100);
  const headLeft = formatTimelineSecondsLabel(windowStart);
  const headRight = formatTimelineSecondsLabel(windowEnd);
  const headNow = formatTimelineSecondsLabel(currentSec);
  const zoomValue = `x${Number(peopleTimelineZoom || 1).toFixed(2)}`;

  peopleTimelineEl.innerHTML = `
    <div class="people-timeline-head">
      <span>${headLeft}</span>
      <span>${headRight}</span>
    </div>
    <div class="people-timeline-playhead" style="left:${playheadPct.toFixed(3)}%;"></div>
    <div class="people-timeline-playhead-label" style="left:${playheadPct.toFixed(3)}%;">${headNow}</div>
    ${tracksHtml}
    ${audioUnderlayHtml}
    ${barsHtml}
    ${draftHtml}
    ${dragTooltipHtml}
    ${hintHtml}
    <div class="people-timeline-zoom" role="group" aria-label="Timeline zoom controls">
      <button class="people-timeline-zoom-btn" type="button" data-zoom="in" title="Zoom in">+</button>
      <button class="people-timeline-zoom-readout" type="button" data-zoom="reset" title="Reset zoom">${zoomValue}</button>
      <button class="people-timeline-zoom-btn" type="button" data-zoom="out" title="Zoom out">-</button>
    </div>
  `;

  peopleTimelineRenderState = {
    windowStart,
    windowEnd,
    windowSpan,
    duration,
    zoom: Number(peopleTimelineZoom || 1),
    laneHeight,
    topPad,
    laneCount: peopleLaneCount,
  };

  if (isPeopleStepActive()) {
    syncPeopleEditorToCursor(currentSec, { force: true });
  } else if (splitMode) {
    syncSplitEditorToCursor(currentSec);
  } else {
    syncSubtitlesEditorToCursor(currentSec, { force: true });
  }

  if (peopleTimelineDraft) {
    const input = peopleTimelineEl.querySelector('.people-timeline-input');
    if (input && document.activeElement !== input) {
      try {
        input.focus({ preventScroll: true });
        input.select();
      } catch (_err) {
        input.focus();
      }
    }
  }
}

function gammaRangesForSave() {
  const span = chapterFrameSpan();
  if (!state.gammaProfile || typeof state.gammaProfile !== 'object') return [];
  const mode = String(state.gammaProfile.mode || 'whole');
  if (mode === 'whole') {
    const g = normalizeGammaValue(state.gammaProfile.level, state.gammaProfile.defaultGamma || 1.0);
    if (Math.abs(g - 1.0) < 0.001) return [];
    return [{ start_frame: span.start, end_frame: span.end, gamma: Number(g.toFixed(4)) }];
  }
  return canonicalizeGammaRanges(state.gammaProfile.ranges || []);
}

function updateGammaRangeMeta() {
  if (!gammaRangeMetaEl) return;
  if (!isGammaStepActive()) {
    gammaRangeMetaEl.textContent = '';
    return;
  }
  const mode = String((state.gammaProfile && state.gammaProfile.mode) || 'whole');
  const ranges = gammaRangesForSave();
  if (mode === 'whole') {
    const g = normalizeGammaValue(state.gammaProfile.level, 1.0);
    gammaRangeMetaEl.textContent = `Whole chapter gamma ${g.toFixed(3)} (${ranges.length ? 'active' : 'no-op at 1.0'})`;
    return;
  }
  if (!ranges.length) {
    gammaRangeMetaEl.textContent = 'Region mode: no ranges set.';
    return;
  }
  const preview = ranges
    .slice(0, 3)
    .map(r => `${r.start_frame}-${r.end_frame}:${Number(r.gamma).toFixed(2)}`)
    .join(' | ');
  const suffix = ranges.length > 3 ? ` | +${ranges.length - 3} more` : '';
  gammaRangeMetaEl.textContent = `Region mode: ${ranges.length} range(s) | ${preview}${suffix}`;
}

function updateGammaControls() {
  if (!gammaLevelEl || !gammaLabelEl || !gammaModeEl) return;
  const profile = state.gammaProfile || {};
  const level = normalizeGammaValue(profile.level, profile.defaultGamma || 1.0);
  gammaLevelEl.value = level.toFixed(2);
  gammaLabelEl.textContent = level.toFixed(2);
  if (flipbookGammaLevelEl) {
    flipbookGammaLevelEl.value = level.toFixed(2);
  }
  if (flipbookGammaLabelEl) {
    flipbookGammaLabelEl.textContent = level.toFixed(2);
  }
  gammaModeEl.value = String(profile.mode || 'whole');
  const regionMode = gammaModeEl.value === 'regions';
  if (gammaApplyVisibleBtnEl) gammaApplyVisibleBtnEl.disabled = !regionMode || isChapterLoadInFlight;
  if (gammaClearBtnEl) gammaClearBtnEl.disabled = isChapterLoadInFlight;
  updateGammaRangeMeta();
}

function reviewLoadProgressSuffix() {
  if (!isChapterLoadInFlight) return '';
  const done = Math.max(0, Math.trunc(Number(loadProgressDone || 0)));
  const total = Math.max(0, Math.trunc(Number(loadProgressTotal || 0)));
  if (total > 0) return ` | loading ${done}/${total}`;
  const msg = String(loadProgressMessage || '').trim();
  return msg ? ` | ${msg}` : ' | loading...';
}

function updateReviewStatsDisplay() {
  if (!reviewStatsEl) return;
  const frames = currentReviewFrames();
  const totalCount = frames.length;
  const loadedCount = reviewLoadedFrameCount(frames);
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) {
    const suffix = reviewLoadProgressSuffix();
    reviewStatsEl.textContent = suffix ? `Loading frames progressively${suffix}` : 'No frames loaded.';
    return;
  }
  const loadSuffix = reviewLoadProgressSuffix();
  if (isPeopleStepActive()) {
    const entries = canonicalizePeopleEntries((state.peopleProfile && state.peopleProfile.entries) || []);
    reviewStatsEl.textContent = `People step | rows ${entries.length} | loaded ${loadedCount}/${totalCount}${loadSuffix}`;
    return;
  }
  if (isSubtitlesStepActive()) {
    const subtitleEntries = canonicalizeSubtitlesEntries((state.subtitlesProfile && state.subtitlesProfile.entries) || []);
    reviewStatsEl.textContent = `Subtitles step | dialogue ${subtitleEntries.length} | loaded ${loadedCount}/${totalCount}${loadSuffix}`;
    return;
  }
  if (isSplitStepActive()) {
    const splitEntries = canonicalizeSplitEntries((state.splitProfile && state.splitProfile.entries) || []);
    reviewStatsEl.textContent = `Chapter step | range ${splitEntries.length} | loaded ${loadedCount}/${totalCount}${loadSuffix}`;
    return;
  }
  if (isGammaStepActive()) {
    reviewStatsEl.textContent = '';
    return;
  }
  reviewStatsEl.textContent = `${statsText(state.review.stats, state.review.threshold)}${loadSuffix}`;
}

async function estimateGammaScoreFromImage(src) {
  const imageSrc = String(src || '').trim();
  if (!imageSrc) return 1.0;
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const w = Math.max(1, Math.min(96, Number(img.naturalWidth || img.width || 96)));
        const h = Math.max(1, Math.min(72, Number(img.naturalHeight || img.height || 72)));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
          resolve(1.0);
          return;
        }
        ctx.drawImage(img, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h).data || [];
        let sum = 0;
        let count = 0;
        for (let i = 0; i < data.length; i += 4) {
          const r = Number(data[i] || 0);
          const g = Number(data[i + 1] || 0);
          const b = Number(data[i + 2] || 0);
          const y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
          sum += y;
          count += 1;
        }
        const avgLuma = count > 0 ? (sum / count) : 0.55;
        const gammaScore = normalizeGammaValue(0.55 / Math.max(0.05, avgLuma), 1.0);
        resolve(gammaScore);
      } catch (_err) {
        resolve(1.0);
      }
    };
    img.onerror = () => resolve(1.0);
    img.decoding = 'async';
    img.src = imageSrc;
  });
}

async function ensureGammaScores() {
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) return;
  const pending = [];
  for (const frame of state.review.frames) {
    if (!isReviewFrameLoaded(frame)) continue;
    const fidKey = String(frame && frame.fid);
    if (!fidKey) continue;
    if (state.gammaScores.has(fidKey)) continue;
    let src = frameImageSrcForFid(fidKey, frame && frame.image);
    if (frame && frame.status === 'bad') {
      const replacementFid = state.freezeReplacementMap.get(fidKey);
      if (replacementFid !== undefined && replacementFid !== null) {
        const replacementSrc = frameImageSrcForFid(replacementFid);
        if (replacementSrc) src = replacementSrc;
      }
    }
    if (!src) continue;
    pending.push({ fidKey, src });
  }
  if (!pending.length) return;
  const CONCURRENCY = 16;
  for (let i = 0; i < pending.length; i += CONCURRENCY) {
    const batch = pending.slice(i, i + CONCURRENCY);
    const scores = await Promise.all(batch.map(item => estimateGammaScoreFromImage(item.src)));
    scores.forEach((score, j) => state.gammaScores.set(batch[j].fidKey, score));
    invalidateGammaSparklineCache();
    if (isGammaStepActive() && state.review) {
      renderActiveSparkline(state.review.frames, state.review.threshold || 0);
    }
    await new Promise(resolve => window.setTimeout(resolve, 0));
  }
}

function gammaScoreForFrame(frame) {
  const fidKey = String(frame && frame.fid);
  if (!fidKey) return 1.0;
  const cached = Number(state.gammaScores.get(fidKey));
  if (Number.isFinite(cached) && cached > 0) return cached;
  return 1.0;
}

function gammaLevelForFrameId(fidRaw) {
  const fid = Number(fidRaw);
  if (!Number.isFinite(fid)) return 1.0;
  const profile = state.gammaProfile || {};
  const mode = String(profile.mode || 'whole');
  const defaultGamma = normalizeGammaValue(profile.defaultGamma, 1.0);
  if (mode === 'whole') {
    return normalizeGammaValue(profile.level, defaultGamma);
  }
  const ranges = canonicalizeGammaRanges(profile.ranges || []);
  for (const range of ranges) {
    const start = Number(range.start_frame);
    const end = Number(range.end_frame);
    if (fid >= start && fid < end) {
      return normalizeGammaValue(range.gamma, defaultGamma);
    }
  }
  return defaultGamma;
}

function refreshGammaVisuals() {
  const gammaOn = isGammaStepActive();
  renderFrameGridWindow(true);

  if (!flipbookImageEl) return;
  if (!gammaOn) {
    flipbookImageEl.style.removeProperty('filter');
    return;
  }
  const frame = sparkPlayFrames[sparkPlayIndex] || null;
  const fid = Number(frame && frame.fid);
  const gamma = gammaLevelForFrameId(fid);
  if (Math.abs(gamma - 1.0) < 0.001) {
    flipbookImageEl.style.removeProperty('filter');
  } else {
    flipbookImageEl.style.filter = `brightness(${gamma.toFixed(3)})`;
  }
}

function themeVar(name, fallback) {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name);
  const value = String(raw || '').trim();
  return value || fallback;
}
