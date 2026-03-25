// audio-sync.js — Audio Sync step logic (Step 3)
// Depends on: tuner-init.js (state, api), tuner-utils.js (api), viewer.js (renderFrameToCanvas, flipbookIndexFromChapterSeconds)

'use strict';

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const audioSyncControlsEl = document.getElementById('audioSyncControls');
const audioSyncOffsetLabelEl = document.getElementById('audioSyncOffsetLabel');
const audioSyncPanelEl = document.getElementById('audioSyncPanel');
const audioSyncWaveEl = document.getElementById('audioSyncWave');
const audioSyncPlayheadEl = document.getElementById('audioSyncPlayhead');
const audioSyncWindowEl = document.getElementById('audioSyncWindow');
const audioSyncFrameEl = document.getElementById('audioSyncFrame');
const audioSyncPlayBtnEl = document.getElementById('audioSyncPlayBtn');
const audioSyncMetaEl = document.getElementById('audioSyncMeta');
const audioSyncAudioEl = document.getElementById('audioSyncAudio');

const audioSyncMinus1Btn = document.getElementById('audioSyncMinus1');
const audioSyncMinus01Btn = document.getElementById('audioSyncMinus01');
const audioSyncMinus001Btn = document.getElementById('audioSyncMinus001');
const audioSyncResetBtn = document.getElementById('audioSyncReset');
const audioSyncPlus001Btn = document.getElementById('audioSyncPlus001');
const audioSyncPlus01Btn = document.getElementById('audioSyncPlus01');
const audioSyncPlus1Btn = document.getElementById('audioSyncPlus1');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let audioSyncPeaks = [];
let audioSyncTotalDurationSec = 0;   // duration of the buffered WAV file
let audioSyncVideoOffsetSec = 0;     // seconds into the WAV where chapter video starts
let audioSyncChapterDurationSec = 0; // duration of the chapter video
let audioSyncPadSeconds = 20.0;      // pad amount on each side (from server)
let audioSyncDrawPending = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function isAudioSyncStepActive() {
  return String((state.currentStep && state.currentStep.mode) || '') === 'audio_sync';
}

function _audioSyncClamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function audioSyncOffsetSeconds() {
  return typeof state.audioSyncOffset === 'number' ? state.audioSyncOffset : 0.0;
}

function _audioSyncFormatOffset(sec) {
  const sign = sec >= 0 ? '+' : '-';
  const abs = Math.abs(sec);
  return `${sign}${abs.toFixed(2)}s`;
}

// ---------------------------------------------------------------------------
// Offset label
// ---------------------------------------------------------------------------
function updateAudioSyncOffsetLabel() {
  if (!audioSyncOffsetLabelEl) return;
  const off = audioSyncOffsetSeconds();
  audioSyncOffsetLabelEl.textContent = `Offset: ${_audioSyncFormatOffset(off)}`;
}

// ---------------------------------------------------------------------------
// Waveform drawing
// ---------------------------------------------------------------------------
function _sampleAudioSyncPeak(secIntoAudio) {
  const peaks = audioSyncPeaks;
  if (!peaks.length || audioSyncTotalDurationSec <= 0) return 0;
  const pos = _audioSyncClamp(secIntoAudio / audioSyncTotalDurationSec, 0, 1) * (peaks.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(peaks.length - 1, lo + 1);
  const t = pos - lo;
  return _audioSyncClamp(Number(peaks[lo] || 0) + (Number(peaks[hi] || 0) - Number(peaks[lo] || 0)) * t, 0, 1);
}

function drawAudioSyncWaveform() {
  audioSyncDrawPending = false;
  if (!audioSyncWaveEl) return;
  const canvas = audioSyncWaveEl;
  const wCss = Math.max(1, canvas.clientWidth || 300);
  const hCss = Math.max(1, canvas.clientHeight || 60);
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.floor(wCss * dpr);
  const h = Math.floor(hCss * dpr);
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, wCss, hCss);

  // Background
  ctx.fillStyle = 'rgba(6, 11, 18, 0.85)';
  ctx.fillRect(0, 0, wCss, hCss);

  // Window shading: the region of the audio that aligns with chapter video at current offset
  if (audioSyncTotalDurationSec > 0) {
    const offset = audioSyncOffsetSeconds();
    const winStart = _audioSyncClamp(audioSyncVideoOffsetSec + offset, 0, audioSyncTotalDurationSec);
    const winEnd = _audioSyncClamp(winStart + audioSyncChapterDurationSec, 0, audioSyncTotalDurationSec);
    const x1 = (winStart / audioSyncTotalDurationSec) * wCss;
    const x2 = (winEnd / audioSyncTotalDurationSec) * wCss;
    ctx.fillStyle = 'rgba(100, 180, 255, 0.12)';
    ctx.fillRect(x1, 0, x2 - x1, hCss);
    // Window border lines
    ctx.strokeStyle = 'rgba(100, 180, 255, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, 0); ctx.lineTo(x1, hCss);
    ctx.moveTo(x2, 0); ctx.lineTo(x2, hCss);
    ctx.stroke();
  }

  // Centre line
  const midY = hCss / 2;
  ctx.strokeStyle = 'rgba(149, 166, 188, 0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, midY); ctx.lineTo(wCss, midY);
  ctx.stroke();

  // Waveform bars
  if (audioSyncPeaks.length) {
    const maxAmp = Math.max(2, Math.floor(hCss * 0.45));
    ctx.strokeStyle = 'rgba(102, 198, 245, 0.85)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < wCss; x += 1) {
      const sec = (x / Math.max(1, wCss - 1)) * audioSyncTotalDurationSec;
      const amp = _sampleAudioSyncPeak(sec);
      const barH = Math.max(1, amp * maxAmp);
      ctx.moveTo(x + 0.5, midY - barH);
      ctx.lineTo(x + 0.5, midY + barH);
    }
    ctx.stroke();
  }
}

function scheduleAudioSyncDraw() {
  if (audioSyncDrawPending) return;
  audioSyncDrawPending = true;
  requestAnimationFrame(() => { drawAudioSyncWaveform(); });
}

// ---------------------------------------------------------------------------
// Playhead position (driven by audio element currentTime)
// ---------------------------------------------------------------------------
function updateAudioSyncPlayhead() {
  if (!audioSyncPlayheadEl || !audioSyncWaveEl) return;
  if (!audioSyncAudioEl || audioSyncTotalDurationSec <= 0) {
    audioSyncPlayheadEl.style.left = '0%';
    return;
  }
  const t = Number(audioSyncAudioEl.currentTime || 0);
  const frac = _audioSyncClamp(t / audioSyncTotalDurationSec, 0, 1);
  audioSyncPlayheadEl.style.left = `${(frac * 100).toFixed(3)}%`;
}

// ---------------------------------------------------------------------------
// Inline video frame display — RAF-driven, contact-sheet backed
// ---------------------------------------------------------------------------
function _audioSyncChapterSecondsFromAudioTime(audioTime) {
  const offset = audioSyncOffsetSeconds();
  return Math.max(0, Number(audioTime) - (audioSyncVideoOffsetSec + offset));
}

// Convert chapter-local seconds → index into currentReviewFrames().
function _audioSyncFrameIdxFromChapterSec(chapterSec) {
  if (typeof chapterFrameSpan !== 'function') return 0;
  const span = chapterFrameSpan();
  if (!span || span.start === undefined) return 0;
  const fps = TIMELINE_FPS_NUM / TIMELINE_FPS_DEN;
  const targetFid = Math.round(span.start + Math.max(0, chapterSec) * fps);
  const clampedFid = Math.max(span.start, Math.min(span.end - 1, targetFid));
  const frames = typeof currentReviewFrames === 'function' ? currentReviewFrames() : [];
  return Math.max(0, Math.min(frames.length - 1, clampedFid - span.start));
}

// Draw the given frame index into the preview canvas via contact-sheet sprites.
function _drawAudioSyncFrameIdx(frameIdx) {
  if (!audioSyncFrameEl) return;
  if (typeof renderFrameToCanvas === 'function') {
    renderFrameToCanvas(frameIdx, audioSyncFrameEl);
  }
}

function updateAudioSyncVideoFrame() {
  if (!audioSyncFrameEl) return;
  const audioTime = Number((audioSyncAudioEl && audioSyncAudioEl.currentTime) || 0);
  const chapterSec = _audioSyncChapterSecondsFromAudioTime(audioTime);
  _drawAudioSyncFrameIdx(_audioSyncFrameIdxFromChapterSec(chapterSec));
}

function showAudioSyncFrameAtChapterStart() {
  _drawAudioSyncFrameIdx(0);
}

// RAF loop — runs while audio is playing to drive smooth frame updates.
let _audioSyncRafId = null;

function _audioSyncRafTick() {
  _audioSyncRafId = null;
  if (!audioSyncAudioEl || audioSyncAudioEl.paused) return;
  updateAudioSyncVideoFrame();
  updateAudioSyncPlayhead();
  _audioSyncRafId = requestAnimationFrame(_audioSyncRafTick);
}

function _startAudioSyncRaf() {
  if (_audioSyncRafId !== null) return;
  _audioSyncRafId = requestAnimationFrame(_audioSyncRafTick);
}

function _stopAudioSyncRaf() {
  if (_audioSyncRafId !== null) {
    cancelAnimationFrame(_audioSyncRafId);
    _audioSyncRafId = null;
  }
}

// ---------------------------------------------------------------------------
// Compute waveform peaks from decoded audio
// ---------------------------------------------------------------------------
function _computeAudioSyncPeaks(samples, bucketCount) {
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
    out.push(_audioSyncClamp(peak, 0, 1));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Load audio from /api/audio_sync_audio and decode waveform peaks
// ---------------------------------------------------------------------------
async function loadAudioSyncAudio() {
  audioSyncPeaks = [];
  audioSyncTotalDurationSec = 0;
  audioSyncVideoOffsetSec = 0;
  audioSyncChapterDurationSec = 0;

  if (audioSyncMetaEl) audioSyncMetaEl.textContent = 'Loading audio…';
  if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.disabled = true;

  // Fetch timing info first
  let info = null;
  try {
    info = await api('/api/audio_sync_info', 'GET', null, 20000);
  } catch (_err) {
    if (audioSyncMetaEl) audioSyncMetaEl.textContent = 'Audio not available for this chapter.';
    scheduleAudioSyncDraw();
    return;
  }
  if (!info || !info.ok) {
    if (audioSyncMetaEl) audioSyncMetaEl.textContent = 'Audio not available.';
    scheduleAudioSyncDraw();
    return;
  }

  audioSyncVideoOffsetSec = Number(info.video_offset_sec || 0);
  audioSyncChapterDurationSec = Number(info.chapter_duration_sec || 0);
  audioSyncPadSeconds = Number(info.pad_seconds || 20);
  // Restore saved offset from server
  if (typeof info.offset_seconds === 'number') {
    state.audioSyncOffset = info.offset_seconds;
    updateAudioSyncOffsetLabel();
  }

  // Set audio src for playback
  if (audioSyncAudioEl) {
    audioSyncAudioEl.src = '/api/audio_sync_audio';
    audioSyncAudioEl.load();
  }

  // Decode audio for waveform peaks
  try {
    const resp = await fetch('/api/audio_sync_audio');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const arrayBuf = await resp.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) throw new Error('AudioContext not supported');
    const ctx = new AudioCtx();
    const decoded = await ctx.decodeAudioData(arrayBuf);
    audioSyncTotalDurationSec = decoded.duration;
    const samples = decoded.getChannelData(0);
    const buckets = Math.max(256, Math.min(2048, Math.floor(audioSyncWaveEl ? audioSyncWaveEl.clientWidth : 800)));
    audioSyncPeaks = _computeAudioSyncPeaks(samples, buckets);
    try { ctx.close(); } catch (_e) {}
  } catch (_err) {
    if (audioSyncMetaEl) audioSyncMetaEl.textContent = 'Could not decode audio waveform.';
  }

  if (audioSyncMetaEl) {
    const totalSec = audioSyncTotalDurationSec;
    const mins = Math.floor(totalSec / 60);
    const secs = (totalSec % 60).toFixed(1);
    audioSyncMetaEl.textContent =
      `Audio: ${mins}:${String(Math.floor(Number(secs))).padStart(2, '0')}.${String(secs).split('.')[1]}  ` +
      `Chapter video starts at +${audioSyncVideoOffsetSec.toFixed(2)}s into clip`;
  }
  if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.disabled = false;
  scheduleAudioSyncDraw();
}

// ---------------------------------------------------------------------------
// Offset adjustment
// ---------------------------------------------------------------------------
function adjustAudioSyncOffset(deltaSec) {
  const maxOffset = audioSyncPadSeconds;
  const current = audioSyncOffsetSeconds();
  const next = _audioSyncClamp(
    Math.round((current + deltaSec) * 1000) / 1000,
    -maxOffset,
    maxOffset,
  );
  state.audioSyncOffset = next;
  updateAudioSyncOffsetLabel();
  scheduleAudioSyncDraw();
  // If audio is playing, update playback start position
  _audioSyncSyncPlayPosition();
}

function resetAudioSyncOffset() {
  state.audioSyncOffset = 0.0;
  updateAudioSyncOffsetLabel();
  scheduleAudioSyncDraw();
  _audioSyncSyncPlayPosition();
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------
function _audioSyncSyncPlayPosition() {
  if (!audioSyncAudioEl || audioSyncTotalDurationSec <= 0) return;
  // Seek to chapter start in the buffered audio adjusted for offset
  const offset = audioSyncOffsetSeconds();
  const startInAudio = _audioSyncClamp(audioSyncVideoOffsetSec + offset, 0, audioSyncTotalDurationSec);
  if (!audioSyncAudioEl.paused) {
    audioSyncAudioEl.currentTime = startInAudio;
  }
}

function toggleAudioSyncPlayback() {
  if (!audioSyncAudioEl) return;
  if (!audioSyncAudioEl.paused) {
    audioSyncAudioEl.pause();
    if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.textContent = '▶';
    return;
  }
  const offset = audioSyncOffsetSeconds();
  const startInAudio = _audioSyncClamp(
    audioSyncVideoOffsetSec + offset,
    0,
    audioSyncTotalDurationSec,
  );
  const endInAudio = _audioSyncClamp(
    startInAudio + audioSyncChapterDurationSec,
    0,
    audioSyncTotalDurationSec,
  );

  audioSyncAudioEl.currentTime = startInAudio;
  audioSyncAudioEl.play().catch(() => {});
  if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.textContent = '||';
  _startAudioSyncRaf();

  // Stop at chapter end
  const stopAt = endInAudio;
  const stopCheck = () => {
    if (!audioSyncAudioEl || audioSyncAudioEl.paused) return;
    if (Number(audioSyncAudioEl.currentTime) >= stopAt) {
      audioSyncAudioEl.pause();
      if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.textContent = '▶';
      return;
    }
    requestAnimationFrame(stopCheck);
  };
  requestAnimationFrame(stopCheck);
}

// ---------------------------------------------------------------------------
// Show / hide step UI
// ---------------------------------------------------------------------------
function showAudioSyncStep() {
  if (audioSyncControlsEl) audioSyncControlsEl.classList.remove('hidden-ui');
  if (audioSyncPanelEl) audioSyncPanelEl.classList.remove('hidden-ui');
}

function hideAudioSyncStep() {
  _stopAudioSyncRaf();
  if (audioSyncControlsEl) audioSyncControlsEl.classList.add('hidden-ui');
  if (audioSyncPanelEl) audioSyncPanelEl.classList.add('hidden-ui');
  if (audioSyncAudioEl && !audioSyncAudioEl.paused) {
    audioSyncAudioEl.pause();
  }
  if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.textContent = '▶';
}

// ---------------------------------------------------------------------------
// Enter step
// ---------------------------------------------------------------------------
async function openAudioSyncStep() {
  showAudioSyncStep();
  updateAudioSyncOffsetLabel();
  scheduleAudioSyncDraw();
  if (typeof ensureAudioSyncFramesReady === 'function') ensureAudioSyncFramesReady();
  showAudioSyncFrameAtChapterStart();
  await loadAudioSyncAudio();
  showAudioSyncFrameAtChapterStart();
  return true;
}

// ---------------------------------------------------------------------------
// Audio element event listeners
// ---------------------------------------------------------------------------
if (audioSyncAudioEl) {
  audioSyncAudioEl.addEventListener('timeupdate', () => {
    updateAudioSyncPlayhead();
    // Frame updates during playback are driven by the RAF loop (_audioSyncRafTick).
    // Update here only when paused (e.g. after a waveform seek).
    if (audioSyncAudioEl.paused) updateAudioSyncVideoFrame();
  });
  audioSyncAudioEl.addEventListener('ended', () => {
    if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.textContent = '▶';
    updateAudioSyncPlayhead();
    showAudioSyncFrameAtChapterStart();
  });
  audioSyncAudioEl.addEventListener('play', () => {
    _startAudioSyncRaf();
  });
  audioSyncAudioEl.addEventListener('pause', () => {
    _stopAudioSyncRaf();
    if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.textContent = '▶';
  });
}

// ---------------------------------------------------------------------------
// Waveform click / drag seeking
// ---------------------------------------------------------------------------
let _audioSyncWaveDragging = false;

function _audioSyncSeekFromClientX(clientX) {
  if (!audioSyncWaveEl || audioSyncTotalDurationSec <= 0) return;
  const rect = audioSyncWaveEl.getBoundingClientRect();
  const frac = _audioSyncClamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
  if (audioSyncAudioEl) {
    audioSyncAudioEl.currentTime = frac * audioSyncTotalDurationSec;
  }
  updateAudioSyncPlayhead();
  updateAudioSyncVideoFrame();
}

if (audioSyncWaveEl) {
  audioSyncWaveEl.addEventListener('mousedown', (e) => {
    _audioSyncWaveDragging = true;
    _audioSyncSeekFromClientX(e.clientX);
    e.preventDefault();
  });
  audioSyncWaveEl.addEventListener('touchstart', (e) => {
    if (e.touches.length !== 1) return;
    _audioSyncWaveDragging = true;
    _audioSyncSeekFromClientX(e.touches[0].clientX);
    e.preventDefault();
  }, { passive: false });
}

window.addEventListener('mousemove', (e) => {
  if (!_audioSyncWaveDragging) return;
  _audioSyncSeekFromClientX(e.clientX);
});
window.addEventListener('mouseup', () => { _audioSyncWaveDragging = false; });
window.addEventListener('touchmove', (e) => {
  if (!_audioSyncWaveDragging || e.touches.length !== 1) return;
  _audioSyncSeekFromClientX(e.touches[0].clientX);
  e.preventDefault();
}, { passive: false });
window.addEventListener('touchend', () => { _audioSyncWaveDragging = false; });

// ---------------------------------------------------------------------------
// Button event listeners
// ---------------------------------------------------------------------------
if (audioSyncMinus1Btn) audioSyncMinus1Btn.addEventListener('click', () => adjustAudioSyncOffset(-1.0));
if (audioSyncMinus01Btn) audioSyncMinus01Btn.addEventListener('click', () => adjustAudioSyncOffset(-0.1));
if (audioSyncMinus001Btn) audioSyncMinus001Btn.addEventListener('click', () => adjustAudioSyncOffset(-0.01));
if (audioSyncResetBtn) audioSyncResetBtn.addEventListener('click', resetAudioSyncOffset);
if (audioSyncPlus001Btn) audioSyncPlus001Btn.addEventListener('click', () => adjustAudioSyncOffset(0.01));
if (audioSyncPlus01Btn) audioSyncPlus01Btn.addEventListener('click', () => adjustAudioSyncOffset(0.1));
if (audioSyncPlus1Btn) audioSyncPlus1Btn.addEventListener('click', () => adjustAudioSyncOffset(1.0));
if (audioSyncPlayBtnEl) audioSyncPlayBtnEl.addEventListener('click', toggleAudioSyncPlayback);

// Redraw waveform on resize
window.addEventListener('resize', () => {
  if (isAudioSyncStepActive()) scheduleAudioSyncDraw();
});
