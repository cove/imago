function normalizeGammaValue(raw, fallback = 1.0) {
  const n = Number(raw);
  const base = Number.isFinite(n) ? n : Number(fallback);
  return _clamp(base, 0.05, 8.0);
}

function chapterFrameSpan() {
  const selectedChapter = chapterByTitle(state.chapter);
  const loadSettingsChapter = String((state.loadSettings && state.loadSettings.chapter) || '').trim();
  const currentChapter = String(state.chapter || '').trim();
  const hasActiveLoadSettings = Boolean(loadSettingsChapter && loadSettingsChapter === currentChapter);
  const startSource = hasActiveLoadSettings
    ? (state.loadSettings && state.loadSettings.start_frame)
    : (selectedChapter && selectedChapter.start_frame);
  const endSource = hasActiveLoadSettings
    ? (state.loadSettings && state.loadSettings.end_frame)
    : (selectedChapter && selectedChapter.end_frame);
  const startRaw = Number(
    startSource
    ?? 0
  );
  const endRaw = Number(
    endSource
    ?? (startRaw + 1)
  );
  const start = Math.max(0, Math.trunc(Number.isFinite(startRaw) ? startRaw : 0));
  const end = Math.max(start + 1, Math.trunc(Number.isFinite(endRaw) ? endRaw : (start + 1)));
  // If we have authoritative chapter metadata (selectedChapter or loadSettings), use it
  // directly. Only fall back to deriving the span from loaded frames' min/max fid when
  // no chapter metadata is available, to avoid inheriting a stale span from a previously
  // loaded chapter when switching between chapters.
  if (selectedChapter || hasActiveLoadSettings) {
    return { start, end };
  }
  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (frames.length) {
    let minFid = Number.POSITIVE_INFINITY;
    let maxFid = Number.NEGATIVE_INFINITY;
    for (const frame of frames) {
      const fid = Math.trunc(Number(frame && frame.fid));
      if (!Number.isFinite(fid)) continue;
      if (fid < minFid) minFid = fid;
      if (fid > maxFid) maxFid = fid;
    }
    if (Number.isFinite(minFid) && Number.isFinite(maxFid) && maxFid >= minFid) {
      return {
        start: Math.max(0, Math.trunc(minFid)),
        end: Math.max(Math.trunc(minFid) + 1, Math.trunc(maxFid) + 1),
      };
    }
  }
  return { start, end };
}

function canonicalizeGammaRanges(rawRanges, options = {}) {
  const span = chapterFrameSpan();
  const clampToChapter = options && options.clampToChapter !== false;
  const rows = [];
  (Array.isArray(rawRanges) ? rawRanges : []).forEach((item, idx) => {
    const start = Number(item && item.start_frame);
    const end = Number(item && item.end_frame);
    const gamma = normalizeGammaValue(item && item.gamma, 1.0);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;
    let a = Math.trunc(start);
    let b = Math.trunc(end);
    if (b <= a) return;
    if (clampToChapter) {
      a = Math.max(span.start, a);
      b = Math.min(span.end, b);
      if (b <= a) return;
    }
    rows.push({ a, b, gamma, idx });
  });
  if (!rows.length) return [];

  const cuts = new Set();
  rows.forEach(r => {
    cuts.add(Number(r.a));
    cuts.add(Number(r.b));
  });
  const ordered = Array.from(cuts).sort((a, b) => a - b);
  const resolved = [];
  for (let i = 0; i < ordered.length - 1; i += 1) {
    const segA = Math.trunc(ordered[i]);
    const segB = Math.trunc(ordered[i + 1]);
    if (segB <= segA) continue;
    let bestIdx = -1;
    let bestGamma = null;
    rows.forEach(r => {
      if (r.a <= segA && segB <= r.b && r.idx >= bestIdx) {
        bestIdx = r.idx;
        bestGamma = r.gamma;
      }
    });
    if (bestGamma === null) continue;
    if (resolved.length) {
      const prev = resolved[resolved.length - 1];
      if (prev.end_frame === segA && Math.abs(Number(prev.gamma) - Number(bestGamma)) < 0.0001) {
        prev.end_frame = segB;
        continue;
      }
    }
    resolved.push({
      start_frame: segA,
      end_frame: segB,
      gamma: Number(bestGamma.toFixed(4)),
    });
  }
  return resolved;
}

function normalizeGammaProfile(rawProfile) {
  const span = chapterFrameSpan();
  const raw = (rawProfile && typeof rawProfile === 'object') ? rawProfile : {};
  const defaultGamma = normalizeGammaValue(raw.default_gamma, 1.0);
  const ranges = canonicalizeGammaRanges(raw.ranges || []);
  let mode = 'whole';
  let level = defaultGamma;
  if (ranges.length === 1 && ranges[0].start_frame <= span.start && ranges[0].end_frame >= span.end) {
    mode = 'whole';
    level = normalizeGammaValue(ranges[0].gamma, defaultGamma);
  } else if (ranges.length > 0) {
    mode = 'regions';
    level = normalizeGammaValue(ranges[0].gamma, defaultGamma);
  }
  return {
    mode,
    defaultGamma,
    level,
    ranges,
    source: String(raw.source || 'default'),
  };
}

function chapterDurationSeconds() {
  const span = chapterFrameSpan();
  return Math.max(0, (Math.max(span.end, span.start + 1) - span.start) * (1001 / 30000));
}

function chapterFrameCount() {
  const span = chapterFrameSpan();
  return Math.max(1, Math.max(span.end, span.start + 1) - span.start);
}

function formatFrameIndex(rawFrame) {
  const n = Number(rawFrame);
  if (!Number.isFinite(n)) return '0';
  return String(Math.max(0, Math.trunc(n)));
}

function parseFrameIndex(rawFrame) {
  let text = String(rawFrame ?? '').trim();
  if (!text) return null;
  text = text.replace(/^f/i, '');
  text = text.replace(/[,\s_]/g, '');
  if (!/^\d+$/.test(text)) return null;
  const n = Number(text);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.trunc(n));
}

function frameIndexFromSeconds(rawSeconds) {
  const step = timelineFrameStepSeconds();
  if (!Number.isFinite(step) || step <= 0) return 0;
  const sec = Math.max(0, Number(rawSeconds || 0));
  return Math.max(0, Math.trunc(Math.round(sec / step)));
}

function secondsFromFrameIndex(rawFrame) {
  const step = timelineFrameStepSeconds();
  if (!Number.isFinite(step) || step <= 0) return 0;
  return Math.max(0, Math.trunc(Number(rawFrame || 0))) * step;
}

function formatTimestampSeconds(rawSeconds) {
  const sec = Math.max(0, Number(rawSeconds || 0));
  const totalMs = Math.round(sec * 1000);
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const seconds = Math.floor((totalMs % 60000) / 1000);
  const millis = Math.max(0, totalMs % 1000);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}

function parseTimestampSeconds(raw) {
  const text = String(raw ?? '').trim();
  if (!text) return null;
  const cleaned = text.replace(',', '.');
  const parts = cleaned.split(':');
  let seconds = NaN;
  if (parts.length === 1) {
    seconds = Number(parts[0]);
  } else if (parts.length === 2) {
    seconds = (Number(parts[0]) * 60) + Number(parts[1]);
  } else if (parts.length === 3) {
    seconds = (Number(parts[0]) * 3600) + (Number(parts[1]) * 60) + Number(parts[2]);
  }
  if (!Number.isFinite(seconds) || Number.isNaN(seconds)) return null;
  return Math.max(0, seconds);
}

function chapterLocalSecondsFromFid(fidRaw) {
  const fid = Number(fidRaw);
  if (!Number.isFinite(fid)) return 0;
  const span = chapterFrameSpan();
  const localFrame = Math.max(0, Math.trunc(fid) - Number(span.start || 0));
  return Number(localFrame) * (TIMELINE_FPS_DEN / TIMELINE_FPS_NUM);
}

function chapterTimecodeFromFid(fidRaw) {
  return formatTimestampSeconds(chapterLocalSecondsFromFid(fidRaw));
}

function chapterLocalFrameFromFid(fidRaw) {
  const fid = Number(fidRaw);
  if (!Number.isFinite(fid)) return 0;
  const span = chapterFrameSpan();
  return Math.max(0, Math.trunc(fid) - Number(span.start || 0));
}

function splitChapterFrameBounds() {
  const span = chapterFrameSpan();
  let start = Math.max(0, Math.trunc(Number(span.start || 0)));
  let endExclusive = Math.max(start + 1, Math.trunc(Number(span.end || (start + 1))));

  const frames = (state.review && Array.isArray(state.review.frames)) ? state.review.frames : [];
  if (frames.length) {
    let minFid = Number.POSITIVE_INFINITY;
    let maxFid = Number.NEGATIVE_INFINITY;
    for (const frame of frames) {
      const fid = Math.trunc(Number(frame && frame.fid));
      if (!Number.isFinite(fid)) continue;
      if (fid < minFid) minFid = fid;
      if (fid > maxFid) maxFid = fid;
    }
    if (Number.isFinite(minFid) && Number.isFinite(maxFid) && maxFid >= minFid) {
      start = Math.max(0, Math.trunc(minFid));
      endExclusive = Math.max(start + 1, Math.trunc(maxFid) + 1);
    }
  }
  const frameCount = Math.max(1, endExclusive - start);
  return {
    start,
    endExclusive,
    endInclusive: endExclusive - 1,
    frameCount,
  };
}

function splitDisplayStartFrame(localStartFrame) {
  const bounds = splitChapterFrameBounds();
  const local = _clamp(Math.trunc(Number(localStartFrame || 0)), 0, bounds.frameCount);
  return bounds.start + local;
}

function splitDisplayEndFrameInclusive(localEndFrameExclusive, localStartFrame = 0) {
  const bounds = splitChapterFrameBounds();
  const localEnd = _clamp(Math.trunc(Number(localEndFrameExclusive || 0)), 0, bounds.frameCount);
  const localStart = _clamp(Math.trunc(Number(localStartFrame || 0)), 0, bounds.frameCount);
  const localInclusive = Math.max(localStart, Math.max(0, localEnd - 1));
  return bounds.start + localInclusive;
}

function parseSplitInputFrameToLocal(rawFrame, options = {}) {
  const parsed = parseFrameIndex(rawFrame);
  if (parsed === null) return null;
  const isEnd = Boolean(options && options.isEnd);
  const bounds = splitChapterFrameBounds();
  if (isEnd && parsed === bounds.endExclusive) {
    return Math.max(0, bounds.frameCount - 1);
  }
  const localMax = isEnd ? Math.max(0, bounds.frameCount - 1) : bounds.frameCount;
  const inLocal = parsed >= 0 && parsed <= localMax;
  const inGlobal = parsed >= bounds.start && parsed <= bounds.endInclusive;
  if (inGlobal && (!inLocal || bounds.start > 0)) {
    return parsed - bounds.start;
  }
  if (inLocal) {
    return parsed;
  }
  if (inGlobal) {
    return parsed - bounds.start;
  }
  return null;
}

function timelineLabelFromFid(fidRaw) {
  if (isSplitStepActive() || isBadFrameStepActive() || isGammaStepActive()) {
    const fid = Number(fidRaw);
    const frame = Number.isFinite(fid) ? Math.max(0, Math.trunc(fid)) : 0;
    return `F${frame}`;
  }
  return chapterTimecodeFromFid(fidRaw);
}

function formatTimelineSecondsLabel(rawSeconds) {
  if (isSplitStepActive()) {
    return `F${splitDisplayStartFrame(frameIndexFromSeconds(rawSeconds))}`;
  }
  if (isBadFrameStepActive() || isGammaStepActive()) {
    const span = chapterFrameSpan();
    return `F${Math.trunc(Number(span.start || 0)) + frameIndexFromSeconds(rawSeconds)}`;
  }
  return formatTimestampSeconds(rawSeconds);
}

function formatFrameCardLabel(frame) {
  if (!frame || frame.fid === undefined || frame.fid === null) return '';
  const tc = timelineLabelFromFid(frame.fid);
  if (!isReviewFrameLoaded(frame)) {
    return tc;
  }
  const score = Number(frame.score);
  const scoreText = Number.isFinite(score) ? `s=${score.toFixed(2)}` : '';
  const sourceText = String(frame.source || '').trim();
  const parts = [tc];
  if (scoreText) parts.push(scoreText);
  if (sourceText) parts.push(sourceText.toUpperCase());
  return parts.join('  ');
}

function visibleRangeTimeText(range, frames) {
  const data = Array.isArray(frames) ? frames : [];
  if (!range || !data.length) return '';
  const startIdx = _clamp(Number(range.start || 0), 0, data.length - 1);
  const endIdx = _clamp(Number(range.end || startIdx), startIdx, data.length - 1);
  const startFrame = data[startIdx] || null;
  const endFrame = data[endIdx] || null;
  if (!startFrame || !endFrame) return '';
  return `${timelineLabelFromFid(startFrame.fid)}-${timelineLabelFromFid(endFrame.fid)}`;
}

function timelineFrameStepSeconds() {
  return TIMELINE_FPS_DEN / TIMELINE_FPS_NUM;
}

function snapTimelineSeconds(rawSeconds) {
  const step = timelineFrameStepSeconds();
  if (!Number.isFinite(step) || step <= 0) return Number(Math.max(0, Number(rawSeconds || 0)).toFixed(3));
  const sec = Math.max(0, Number(rawSeconds || 0));
  return Number((Math.round(sec / step) * step).toFixed(3));
}

