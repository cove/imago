function resetSparklineCache() {
  state.sparklineCache = {
    mode: '',
    key: '',
    viewBox: '0 0 320 68',
    defsSvg: '',
    baseSvg: '',
    metaBase: '',
    totalCount: 0,
    width: 320,
    height: 68,
    pad: 6,
  };
}

function invalidateSparklineCache(mode = '') {
  const targetMode = String(mode || '').trim();
  if (!targetMode || String(state.sparklineCache.mode || '') === targetMode) {
    resetSparklineCache();
  }
}

function invalidateReviewSparklineCache() {
  state.reviewSparklineVersion = Math.max(0, Math.trunc(Number(state.reviewSparklineVersion || 0))) + 1;
  state.gammaSparklineVersion = Math.max(0, Math.trunc(Number(state.gammaSparklineVersion || 0))) + 1;
  invalidateSparklineCache();
}

function invalidateGammaSparklineCache() {
  state.gammaSparklineVersion = Math.max(0, Math.trunc(Number(state.gammaSparklineVersion || 0))) + 1;
  invalidateSparklineCache('gamma');
}

function replaceGammaScores(nextMap = null) {
  state.gammaScores = nextMap instanceof Map ? nextMap : new Map();
  invalidateGammaSparklineCache();
}

function sparklineViewportOverlaySvg(cache, range) {
  const totalCount = Math.max(0, Math.trunc(Number(cache && cache.totalCount || 0)));
  if (!range || totalCount <= 0) return '';
  const w = Math.max(1, Number(cache && cache.width || 320));
  const h = Math.max(1, Number(cache && cache.height || 68));
  const pad = Math.max(0, Number(cache && cache.pad || 6));
  const sparkWindow = themeVar('--spark-window', '#6ca0ff');
  const sparkWindowMid = themeVar('--spark-window-mid', '#9dc6ff');
  const xFor = (i) => {
    if (totalCount <= 1) return w / 2;
    return pad + (i * (w - pad * 2)) / (totalCount - 1);
  };
  const startIdx = Math.max(0, Math.min(totalCount - 1, Number(range.start)));
  const endIdx = Math.max(startIdx, Math.min(totalCount - 1, Number(range.end)));
  const xStart = xFor(startIdx);
  const xEnd = xFor(endIdx);
  const xMid = (xStart + xEnd) / 2;
  return `
    <rect x="${Math.min(xStart, xEnd).toFixed(2)}" y="${pad}" width="${Math.max(1, Math.abs(xEnd - xStart)).toFixed(2)}" height="${(h - pad * 2).toFixed(2)}" fill="${sparkWindow}" opacity="0.15"></rect>
    <line x1="${xStart.toFixed(2)}" y1="${pad}" x2="${xStart.toFixed(2)}" y2="${(h - pad).toFixed(2)}" stroke="${sparkWindow}" stroke-width="1.3" opacity="0.95"></line>
    <line x1="${xEnd.toFixed(2)}" y1="${pad}" x2="${xEnd.toFixed(2)}" y2="${(h - pad).toFixed(2)}" stroke="${sparkWindow}" stroke-width="1.3" opacity="0.95"></line>
    <line x1="${xMid.toFixed(2)}" y1="${pad}" x2="${xMid.toFixed(2)}" y2="${(h - pad).toFixed(2)}" stroke="${sparkWindowMid}" stroke-width="1.0" opacity="0.9"></line>
  `;
}

function sparklineMetaText(cache, frames) {
  const base = String(cache && cache.metaBase || '').trim();
  if (!base) return '';
  if (!state.visibleRange) return base;
  const data = Array.isArray(frames) ? frames : [];
  const totalCount = data.length;
  if (!totalCount) return base;
  const start = Math.max(0, Number(state.visibleRange.start)) + 1;
  const end = Math.max(start, Number(state.visibleRange.end) + 1);
  const viewTc = visibleRangeTimeText(state.visibleRange, data);
  return `${base} | view ${viewTc || `${start}-${end}/${totalCount}`}`;
}

function applySparklineCache(cache, frames) {
  if (!iqrSparkEl || !sparkMetaEl) return;
  const nextCache = cache && typeof cache === 'object' ? cache : null;
  if (!nextCache) {
    iqrSparkEl.innerHTML = '';
    sparkMetaEl.textContent = '';
    resetSparklineCache();
    return;
  }
  if (String(iqrSparkEl.getAttribute('viewBox') || '') !== String(nextCache.viewBox || '')) {
    iqrSparkEl.setAttribute('viewBox', String(nextCache.viewBox || '0 0 320 68'));
  }
  const overlaySvg = sparklineViewportOverlaySvg(nextCache, state.visibleRange);
  const nextMarkup = `
    ${String(nextCache.defsSvg || '').trim() ? `<defs>${nextCache.defsSvg}</defs>` : ''}
    <g data-spark-base="1">${nextCache.baseSvg || ''}</g>
    <g data-spark-overlay="1">${overlaySvg}</g>
  `;
  if (
    String(state.sparklineCache.key || '') !== String(nextCache.key || '')
    || iqrSparkEl.querySelector('[data-spark-overlay="1"]') === null
  ) {
    iqrSparkEl.innerHTML = nextMarkup;
  } else {
    const overlayLayer = iqrSparkEl.querySelector('[data-spark-overlay="1"]');
    if (overlayLayer) {
      overlayLayer.innerHTML = overlaySvg;
    } else {
      iqrSparkEl.innerHTML = nextMarkup;
    }
  }
  sparkMetaEl.textContent = sparklineMetaText(nextCache, frames);
  state.sparklineCache = nextCache;
}

function buildReviewSparklineCache(frames, threshold) {
  const data = Array.isArray(frames) ? frames : [];
  const totalCount = data.length;
  const w = Math.max(320, Math.floor(iqrSparkEl.clientWidth || 320));
  const h = 68;
  const pad = 6;
  const key = `review:${state.reviewSparklineVersion}:${imageFetchVersion}:${w}`;
  const loaded = [];
  for (let i = 0; i < data.length; i += 1) {
    const frame = data[i];
    if (!isReviewFrameLoaded(frame)) continue;
    loaded.push({ index: i, frame });
  }
  const sparkHot = themeVar('--spark-hot', '#ff9154');
  const sparkBase = themeVar('--spark-base', '#48a5bc');
  const sparkThreshold = themeVar('--spark-threshold', '#ff646e');
  const sparkBad = themeVar('--bad', '#e16f7a');
  const sparkGood = themeVar('--good', '#4ac28a');
  const bucketCount = Math.max(48, Math.floor(w - (pad * 2)));
  const xFor = (i) => {
    if (totalCount <= 1) return w / 2;
    return pad + (i * (w - pad * 2)) / (totalCount - 1);
  };
  if (!loaded.length) {
    return {
      mode: 'review',
      key,
      viewBox: `0 0 ${w} ${h}`,
      defsSvg: '',
      baseSvg: `<rect x="${pad}" y="${pad}" width="${Math.max(1, w - (pad * 2)).toFixed(2)}" height="${(h - pad * 2).toFixed(2)}" fill="${sparkBase}" opacity="0.06"></rect>`,
      metaBase: `Scores: loading 0/${totalCount}`,
      totalCount,
      width: w,
      height: h,
      pad,
    };
  }

  const scores = loaded.map(({ frame }) => Number(frame.score || 0));
  const badCount = loaded.filter(({ frame }) => frame.status === 'bad').length;
  const minV = Math.min(...scores, Number(threshold || 0));
  const maxV = Math.max(...scores, Number(threshold || 0));
  const span = Math.max(1e-9, maxV - minV);
  const yFor = (v) => h - pad - ((v - minV) / span) * (h - pad * 2);
  const points = downsampleSparkSeries(scores, bucketCount)
    .map((point) => `${xFor(loaded[point.index].index).toFixed(2)},${yFor(point.value).toFixed(2)}`)
    .join(' ');
  const yThr = yFor(Number(threshold || 0));
  const yThrClamped = _clamp(yThr, 0, h);
  const clipId = 'sparkAboveThresholdClipReview';
  // Pixel-accurate frame bar: map every frame to its x pixel, bad wins over good wins over fetched.
  // Run-length encoded into rects — O(w) elements, no frames missed.
  const barY = h - 3;
  const barH = 3;
  const pixelColor = new Uint8Array(w); // 0=none, 1=fetched, 2=good, 3=bad
  for (let i = 0; i < data.length; i += 1) {
    const px = Math.round(xFor(i));
    if (px < 0 || px >= w) continue;
    const frame = data[i];
    if (isReviewFrameLoaded(frame)) {
      if (frame.status === 'bad') { pixelColor[px] = 3; }
      else if (pixelColor[px] < 2) { pixelColor[px] = 2; }
    } else if (_isImageFetchedIndex(i) && pixelColor[px] === 0) {
      pixelColor[px] = 1;
    }
  }
  let frameBarSvg = '';
  for (let i = 0; i < w;) {
    const c = pixelColor[i];
    if (!c) { i += 1; continue; }
    let j = i + 1;
    while (j < w && pixelColor[j] === c) j += 1;
    const fill = c === 3 ? sparkBad : c === 2 ? sparkGood : '#839496';
    const opacity = c === 3 ? '0.95' : c === 2 ? '0.82' : '0.55';
    frameBarSvg += `<rect x="${i}" y="${barY}" width="${j - i}" height="${barH}" fill="${fill}" opacity="${opacity}"></rect>`;
    i = j;
  }
  let unloadedSvg = '';
  if (loaded.length < totalCount) {
    const firstUnloadedIndex = Math.max(0, loaded[loaded.length - 1].index + 1);
    const xStart = xFor(firstUnloadedIndex);
    unloadedSvg = `<rect x="${xStart.toFixed(2)}" y="${pad}" width="${Math.max(1, (w - pad) - xStart).toFixed(2)}" height="${(h - pad * 2).toFixed(2)}" fill="${sparkBase}" opacity="0.05"></rect>`;
  }
  return {
    mode: 'review',
    key,
    viewBox: `0 0 ${w} ${h}`,
    defsSvg: `<clipPath id="${clipId}"><rect x="0" y="0" width="${w}" height="${yThrClamped.toFixed(2)}"></rect></clipPath>`,
    baseSvg: `
      ${unloadedSvg}
      <polyline points="${points}" fill="none" stroke="${sparkBase}" stroke-width="1.6" opacity="0.95"></polyline>
      <polyline points="${points}" fill="none" stroke="${sparkHot}" stroke-width="1.9" opacity="0.98" clip-path="url(#${clipId})"></polyline>
      <line x1="${pad}" y1="${yThr.toFixed(2)}" x2="${(w - pad).toFixed(2)}" y2="${yThr.toFixed(2)}" stroke="${sparkThreshold}" stroke-width="1.4" opacity="0.95"></line>
      ${frameBarSvg}
    `,
    metaBase: `Scores: min ${minV.toFixed(2)} | max ${maxV.toFixed(2)} | threshold ${Number(threshold || 0).toFixed(2)} | auto bad ${badCount} | loaded ${loaded.length}/${totalCount}`,
    totalCount,
    width: w,
    height: h,
    pad,
  };
}

function buildGammaSparklineCache(frames, gammaLevel) {
  const data = Array.isArray(frames) ? frames : [];
  const totalCount = data.length;
  const w = Math.max(320, Math.floor(iqrSparkEl.clientWidth || 320));
  const h = 68;
  const pad = 6;
  const level = normalizeGammaValue(gammaLevel, 1.0);
  const key = `gamma:${state.gammaSparklineVersion}:${w}:${level.toFixed(3)}`;
  const loaded = [];
  for (let i = 0; i < data.length; i += 1) {
    const frame = data[i];
    if (!isReviewFrameLoaded(frame)) continue;
    loaded.push({ index: i, frame });
  }
  const sparkHot = themeVar('--spark-hot', '#ff9154');
  const sparkBase = themeVar('--spark-base', '#48a5bc');
  const sparkThreshold = themeVar('--spark-threshold', '#ff646e');
  const bucketCount = Math.max(48, Math.floor(w - (pad * 2)));
  const xFor = (i) => {
    if (totalCount <= 1) return w / 2;
    return pad + (i * (w - pad * 2)) / (totalCount - 1);
  };
  if (!loaded.length) {
    return {
      mode: 'gamma',
      key,
      viewBox: `0 0 ${w} ${h}`,
      defsSvg: '',
      baseSvg: `<rect x="${pad}" y="${pad}" width="${Math.max(1, w - (pad * 2)).toFixed(2)}" height="${(h - pad * 2).toFixed(2)}" fill="${sparkBase}" opacity="0.06"></rect>`,
      metaBase: `Gamma scores: loading 0/${totalCount}`,
      totalCount,
      width: w,
      height: h,
      pad,
    };
  }

  const scores = loaded.map(({ frame }) => gammaScoreForFrame(frame));
  const aboveCount = scores.filter((value) => Number(value) > level).length;
  const minV = Math.min(...scores, level);
  const maxV = Math.max(...scores, level);
  const span = Math.max(1e-9, maxV - minV);
  const yFor = (v) => h - pad - ((v - minV) / span) * (h - pad * 2);
  const points = downsampleSparkSeries(scores, bucketCount)
    .map((point) => `${xFor(loaded[point.index].index).toFixed(2)},${yFor(point.value).toFixed(2)}`)
    .join(' ');
  const yLevel = yFor(level);
  const yLevelClamped = _clamp(yLevel, 0, h);
  const clipId = 'sparkAboveThresholdClipGamma';
  // Pixel-accurate gamma bar: hot (above level) wins over normal.
  const barY = h - 3;
  const barH = 3;
  const pixelColor = new Uint8Array(w); // 0=none, 1=normal, 2=hot
  for (let i = 0; i < loaded.length; i += 1) {
    const { index, frame } = loaded[i];
    const px = Math.round(xFor(index));
    if (px < 0 || px >= w) continue;
    const hot = gammaScoreForFrame(frame) > level;
    if (hot) { pixelColor[px] = 2; }
    else if (pixelColor[px] === 0) { pixelColor[px] = 1; }
  }
  let gammaBarSvg = '';
  for (let i = 0; i < w;) {
    const c = pixelColor[i];
    if (!c) { i += 1; continue; }
    let j = i + 1;
    while (j < w && pixelColor[j] === c) j += 1;
    const fill = c === 2 ? sparkHot : sparkBase;
    const opacity = c === 2 ? '0.95' : '0.45';
    gammaBarSvg += `<rect x="${i}" y="${barY}" width="${j - i}" height="${barH}" fill="${fill}" opacity="${opacity}"></rect>`;
    i = j;
  }
  let unloadedSvg = '';
  if (loaded.length < totalCount) {
    const firstUnloadedIndex = Math.max(0, loaded[loaded.length - 1].index + 1);
    const xStart = xFor(firstUnloadedIndex);
    unloadedSvg = `<rect x="${xStart.toFixed(2)}" y="${pad}" width="${Math.max(1, (w - pad) - xStart).toFixed(2)}" height="${(h - pad * 2).toFixed(2)}" fill="${sparkBase}" opacity="0.05"></rect>`;
  }
  return {
    mode: 'gamma',
    key,
    viewBox: `0 0 ${w} ${h}`,
    defsSvg: `<clipPath id="${clipId}"><rect x="0" y="0" width="${w}" height="${yLevelClamped.toFixed(2)}"></rect></clipPath>`,
    baseSvg: `
      ${unloadedSvg}
      <polyline points="${points}" fill="none" stroke="${sparkBase}" stroke-width="1.6" opacity="0.95"></polyline>
      <polyline points="${points}" fill="none" stroke="${sparkHot}" stroke-width="1.9" opacity="0.98" clip-path="url(#${clipId})"></polyline>
      <line x1="${pad}" y1="${yLevel.toFixed(2)}" x2="${(w - pad).toFixed(2)}" y2="${yLevel.toFixed(2)}" stroke="${sparkThreshold}" stroke-width="1.4" opacity="0.95"></line>
      ${gammaBarSvg}
    `,
    metaBase: `Gamma scores: min ${minV.toFixed(2)} | max ${maxV.toFixed(2)} | level ${level.toFixed(2)} | above level ${aboveCount} | loaded ${loaded.length}/${totalCount}`,
    totalCount,
    width: w,
    height: h,
    pad,
  };
}

function downsampleSparkSeries(values, maxBuckets) {
  const nums = Array.isArray(values) ? values.map((value) => Number(value || 0)) : [];
  const bucketCount = Math.max(1, Math.trunc(Number(maxBuckets || 1)));
  if (nums.length <= Math.max(2, bucketCount * 2)) {
    return nums.map((value, index) => ({ index, value }));
  }
  const points = [];
  let lastIndex = -1;
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = Math.floor((bucket * nums.length) / bucketCount);
    const end = Math.max(start + 1, Math.floor(((bucket + 1) * nums.length) / bucketCount));
    let minVal = nums[start];
    let maxVal = nums[start];
    let minIdx = start;
    let maxIdx = start;
    for (let i = start + 1; i < end; i += 1) {
      const value = nums[i];
      if (value < minVal) {
        minVal = value;
        minIdx = i;
      }
      if (value > maxVal) {
        maxVal = value;
        maxIdx = i;
      }
    }
    const ordered = minIdx <= maxIdx
      ? [{ index: minIdx, value: minVal }, { index: maxIdx, value: maxVal }]
      : [{ index: maxIdx, value: maxVal }, { index: minIdx, value: minVal }];
    for (const point of ordered) {
      if (point.index === lastIndex) continue;
      points.push(point);
      lastIndex = point.index;
    }
  }
  return points;
}

function downsampleMarkerIndexes(flags, maxBuckets) {
  const bools = Array.isArray(flags) ? flags.map(Boolean) : [];
  const bucketCount = Math.max(1, Math.trunc(Number(maxBuckets || 1)));
  if (bools.length <= bucketCount) {
    return bools.map((on, index) => (on ? index : null)).filter((index) => index !== null);
  }
  const out = [];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = Math.floor((bucket * bools.length) / bucketCount);
    const end = Math.max(start + 1, Math.floor(((bucket + 1) * bools.length) / bucketCount));
    let hit = false;
    for (let i = start; i < end; i += 1) {
      if (!bools[i]) continue;
      hit = true;
      break;
    }
    if (hit) {
      out.push(Math.floor((start + Math.max(start, end - 1)) / 2));
    }
  }
  return out;
}

function renderSparkline(frames, threshold) {
  if (!iqrSparkEl || !sparkMetaEl) return;
  const data = Array.isArray(frames) ? frames : [];
  if (!data.length) {
    applySparklineCache(null, []);
    sparkMetaEl.textContent = 'No score data loaded.';
    state.visibleRange = null;
    return;
  }
  const width = Math.max(320, Math.floor(iqrSparkEl.clientWidth || 320));
  const cacheKey = `review:${state.reviewSparklineVersion}:${imageFetchVersion}:${width}`;
  const nextCache = (
    String(state.sparklineCache.mode || '') === 'review'
    && String(state.sparklineCache.key || '') === cacheKey
  )
    ? state.sparklineCache
    : buildReviewSparklineCache(data, threshold);
  applySparklineCache(nextCache, data);
}

function renderGammaSparkline(frames, gammaLevel) {
  if (!iqrSparkEl || !sparkMetaEl) return;
  const data = Array.isArray(frames) ? frames : [];
  if (!data.length) {
    applySparklineCache(null, []);
    sparkMetaEl.textContent = 'No gamma score data loaded.';
    state.visibleRange = null;
    return;
  }
  const width = Math.max(320, Math.floor(iqrSparkEl.clientWidth || 320));
  const level = normalizeGammaValue(gammaLevel, 1.0);
  const cacheKey = `gamma:${state.gammaSparklineVersion}:${width}:${level.toFixed(3)}`;
  const nextCache = (
    String(state.sparklineCache.mode || '') === 'gamma'
    && String(state.sparklineCache.key || '') === cacheKey
  )
    ? state.sparklineCache
    : buildGammaSparklineCache(data, gammaLevel);
  applySparklineCache(nextCache, data);
}

function renderActiveSparkline(frames, threshold) {
  if (isGammaStepActive()) {
    renderGammaSparkline(frames, state.gammaProfile && state.gammaProfile.level);
  } else {
    renderSparkline(frames, threshold);
  }
}

function updateTimelineScrubUi() {
  if (!timelineScrubEl || !timelineScrubMetaEl) return;
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (!frames.length) {
    timelineScrubEl.min = '0';
    timelineScrubEl.max = '0';
    timelineScrubEl.step = '1';
    timelineScrubEl.value = '0';
    timelineScrubEl.disabled = true;
    timelineScrubMetaEl.textContent = 'Timeline scrub: no loaded frames available.';
    updateTimelineAudioPlayheadFromIndex(0);
    updateTimelineAudioPlayButton();
    renderPeopleTimeline(0);
    return;
  }
  timelineScrubEl.disabled = false;
  timelineScrubEl.min = '0';
  timelineScrubEl.max = String(Math.max(0, frames.length - 1));
  timelineScrubEl.step = '1';

  const range = state.visibleRange || { start: 0, end: 0 };
  const startIdx = _clamp(Number(range.start || 0), 0, frames.length - 1);
  const endIdx = _clamp(Number(range.end || startIdx), startIdx, frames.length - 1);
  const centerIdx = _clamp(Math.round((startIdx + endIdx) / 2), 0, frames.length - 1);
  const scrubActive = document.activeElement === timelineScrubEl;
  if (!scrubActive) {
    timelineScrubEl.value = String(centerIdx);
  }
  let cursorIdx = centerIdx;
  if (scrubActive) {
    const activeIdx = Number(timelineScrubEl.value);
    if (Number.isFinite(activeIdx)) {
      cursorIdx = _clamp(Math.round(activeIdx), 0, frames.length - 1);
    }
  }

  const firstTc = timelineLabelFromFid(frames[0] && frames[0].fid);
  const lastTc = timelineLabelFromFid(frames[frames.length - 1] && frames[frames.length - 1].fid);
  const cursorTc = timelineLabelFromFid(frames[cursorIdx] && frames[cursorIdx].fid);
  const viewTc = visibleRangeTimeText({ start: startIdx, end: endIdx }, frames);
  timelineScrubMetaEl.textContent = `Timeline: ${cursorTc} (${cursorIdx + 1}/${frames.length}) | view ${viewTc || '-'} | span ${firstTc}-${lastTc}`;
  updateTimelineAudioPlayheadFromIndex(cursorIdx);
  updateTimelineAudioPlayButton();
  renderPeopleTimeline(chapterLocalSecondsFromFid(frames[cursorIdx] && frames[cursorIdx].fid));
}

function _computeVisibleIndexRange() {
  const frames = currentReviewFrames();
  if (!frameGridEl || !frames.length) return null;
  const metrics = frameGridMetrics(frames.length);
  const range = frameGridVisibleIndexRange(metrics, 0);
  if (!range) return null;
  return { start: range.visibleStart, end: range.visibleEnd };
}

function refreshVisibleRangeFromGrid() {
  if (!state.review || !Array.isArray(state.review.frames) || !state.review.frames.length) {
    state.visibleRange = null;
    return;
  }
  const range = _computeVisibleIndexRange();
  state.visibleRange = range || { start: 0, end: state.review.frames.length - 1 };
  renderActiveSparkline(state.review.frames, state.review.threshold || 0);
  updateTimelineScrubUi();
  prefetchVisibleFrameSheets(null, state.visibleRange);
}

function scheduleVisibleRangeRefresh() {
  if (visibleRangeRefreshPending) return;
  visibleRangeRefreshPending = true;
  window.requestAnimationFrame(() => {
    visibleRangeRefreshPending = false;
    renderFrameGridWindow();
    refreshVisibleRangeFromGrid();
  });
}

function isReviewStepActive() {
  return isReviewMode(stepDef(state.wizardStep).mode);
}

function isBadFrameStepActive() {
  return stepDef(state.wizardStep).mode === 'review';
}

function isGammaStepActive() {
  return stepDef(state.wizardStep).mode === 'gamma';
}

function isPeopleStepActive() {
  return stepDef(state.wizardStep).mode === 'people';
}

function isSubtitlesStepActive() {
  return stepDef(state.wizardStep).mode === 'subtitles';
}

function isSplitStepActive() {
  return stepDef(state.wizardStep).mode === 'split';
}

function isTimelineStepActive() {
  return isPeopleStepActive() || isSubtitlesStepActive() || isSplitStepActive();
}

function updatePeopleStepLayoutSizing() {
  if (!frameGridEl || !sparkPanelEl) return;
  const flipbookFocus = Boolean(page2El && page2El.classList.contains('flipbook-focus'));
  if (!isTimelineStepActive() || flipbookFocus) {
    frameGridEl.style.removeProperty('min-height');
    frameGridEl.style.removeProperty('max-height');
    frameGridEl.style.removeProperty('height');
    sparkPanelEl.style.removeProperty('max-height');
    sparkPanelEl.style.removeProperty('overflow');
    if (subtitlesEditorEl) subtitlesEditorEl.style.removeProperty('max-height');
    if (splitEditorEl) splitEditorEl.style.removeProperty('max-height');
    if (reviewLayoutEl) reviewLayoutEl.style.removeProperty('grid-template-rows');
    return;
  }
  if (reviewLayoutEl) {
    reviewLayoutEl.style.gridTemplateRows = 'auto 1fr 1fr';
  }
  frameGridEl.style.minHeight = `${PEOPLE_STEP_MIN_FRAME_GRID_PX}px`;
  frameGridEl.style.removeProperty('max-height');
  frameGridEl.style.removeProperty('height');
  sparkPanelEl.style.removeProperty('max-height');
  sparkPanelEl.style.overflow = 'auto';
  if (subtitlesEditorEl) subtitlesEditorEl.style.removeProperty('max-height');
  if (splitEditorEl) splitEditorEl.style.removeProperty('max-height');
}

