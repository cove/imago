const FRAME_SHEET_DEFAULT_CHUNK_SIZE = 512;
const FRAME_SHEET_DEFAULT_COLUMNS = 8;
const FRAME_SHEET_DEFAULT_THUMB_WIDTH = 160;
const FRAME_SHEET_DEFAULT_THUMB_HEIGHT = 120;

const state = {
  archives: [],
  archive: '',
  chapters: [],
  chapter: '',
  wizardStep: 1,
  review: null,
  loadSettings: null,
  iqrApplySeq: 0,
  liveIqrPending: false,
  liveIqrInFlight: false,
  visibleRange: null,
  frameImages: new Map(),
  freezeReplacementMap: new Map(),
  frameSheetConfig: {
    rev: '',
    chunkSize: FRAME_SHEET_DEFAULT_CHUNK_SIZE,
    columns: FRAME_SHEET_DEFAULT_COLUMNS,
    thumbWidth: FRAME_SHEET_DEFAULT_THUMB_WIDTH,
    thumbHeight: FRAME_SHEET_DEFAULT_THUMB_HEIGHT,
  },
  simulateFreezeFrame: false,
  simulateFreezeFrameReviewPref: false,
  forceAllFramesGood: false,
  gammaScores: new Map(),
  gammaProfile: {
    mode: 'whole',
    defaultGamma: 1.0,
    level: 1.0,
    ranges: [],
    source: 'default',
  },
  peopleProfile: {
    entries: [],
    source: 'default',
  },
  subtitlesProfile: {
    entries: [],
    source: 'default',
  },
  autoTranscript: false,
  splitProfile: {
    entries: [],
    source: 'default',
  },
  sparklineCache: {
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
  },
  reviewSparklineVersion: 0,
  gammaSparklineVersion: 0,
};

const STEP_DEFS = [
  { num: 1, label: 'Load Chapter', pageId: 'page1', mode: 'load' },
  { num: 2, label: 'Bad Frames', pageId: 'page2', mode: 'review' },
  { num: 3, label: 'Gamma Correction', pageId: 'page2', mode: 'gamma' },
  { num: 4, label: 'People', pageId: 'page2', mode: 'people' },
  { num: 5, label: 'Subtitles', pageId: 'page2', mode: 'subtitles' },
  { num: 6, label: 'Chapter', pageId: 'page2', mode: 'split' },
  { num: 7, label: 'Summary + Save', pageId: 'page3', mode: 'summary' },
];
const STEP_BY_NUM = new Map(STEP_DEFS.map((step) => [step.num, step]));
const STEP_INDEX_BY_NUM = new Map(STEP_DEFS.map((step, index) => [step.num, index]));
const STEP_COUNT = STEP_DEFS.length;
const STEP_FIRST = STEP_DEFS[0];
const STEP_PAGE_IDS = Array.from(new Set(STEP_DEFS.map((step) => step.pageId)));
const STEP_MODE_TO_FIRST = new Map();
STEP_DEFS.forEach((step) => {
  if (!STEP_MODE_TO_FIRST.has(step.mode)) {
    STEP_MODE_TO_FIRST.set(step.mode, step);
  }
});

const FREEZE_BRIDGE_ALWAYS_GAP = 0;
const FREEZE_BRIDGE_SINGLETON_GAP = 0;
const FREEZE_SINGLE_FRAME_SOURCE_SKIP = 0;
const FREEZE_SOURCE_CLEARANCE = 0;
const TIMELINE_FPS_NUM = 30000;
const TIMELINE_FPS_DEN = 1001;
const VHS_FRAME_SECONDS = TIMELINE_FPS_DEN / TIMELINE_FPS_NUM;
const VHS_FRAME_MS = VHS_FRAME_SECONDS * 1000;
const PEOPLE_TIMELINE_DEFAULT_DURATION_SECONDS = 2.0;
const PEOPLE_STEP_MIN_FRAME_GRID_PX = 220;
const PEOPLE_STEP_SPARK_MIN_PX = 220;
const PEOPLE_STEP_SPARK_MAX_PX = 380;
const FRAME_GRID_CARD_MIN_WIDTH_PX = 170;
const FRAME_GRID_CARD_HEIGHT_PX = 149;
const FRAME_GRID_GAP_PX = 8;
const FRAME_GRID_PADDING_PX = 8;
const FRAME_GRID_OVERSCAN_ROWS = 3;
const FRAME_GRID_CONTACT_SHEET_MIN_ROWS = 4;
const FRAME_GRID_CONTACT_SHEET_PREFETCH_AHEAD = 2;
const LOAD_PROGRESS_POLL_MS = 400;
const LOAD_REVIEW_POLL_MIN_MS = 1200;
const LOAD_REVIEW_POLL_MIN_FRAME_DELTA = 48;
const FLIPBOOK_AUDIO_CLOCK_STALL_EPSILON = 0.002;
const FLIPBOOK_AUDIO_CLOCK_STALL_FRAMES = 8;

const statusEl = document.getElementById('status');
const stepPillsEl = document.getElementById('stepPills');
const overlayEl = document.getElementById('overlay');
const overlayMsgEl = document.getElementById('overlayMsg');
const overlayProgressFillEl = document.getElementById('overlayProgressFill');
const overlayProgressTextEl = document.getElementById('overlayProgressText');
const overlayEtaTextEl = document.getElementById('overlayEtaText');
const overlayCancelWrapEl = document.getElementById('overlayCancelWrap');
const overlayCancelBtnEl = document.getElementById('overlayCancelBtn');
const helpBtnEl = document.getElementById('helpBtn');
const helpModalEl = document.getElementById('helpModal');
const helpCloseBtnEl = document.getElementById('helpCloseBtn');

const archiveListEl = document.getElementById('archiveList');
const chapterListEl = document.getElementById('chapterList');

const iqrEl = document.getElementById('iqrK');
const iqrLabelEl = document.getElementById('iqrLabel');
const gammaLevelEl = document.getElementById('gammaLevel');
const gammaLabelEl = document.getElementById('gammaLabel');
const flipbookGammaControlsEl = document.getElementById('flipbookGammaControls');
const flipbookGammaLevelEl = document.getElementById('flipbookGammaLevel');
const flipbookGammaLabelEl = document.getElementById('flipbookGammaLabel');
const gammaModeEl = document.getElementById('gammaMode');
const gammaApplyVisibleBtnEl = document.getElementById('gammaApplyVisible');
const gammaClearBtnEl = document.getElementById('gammaClear');
const peopleControlsEl = document.getElementById('peopleControls');
const subtitlesControlsEl = document.getElementById('subtitlesControls');
const peoplePrefillCastBtnEl = document.getElementById('peoplePrefillCast');
const subtitlesGenerateBtnEl = document.getElementById('subtitlesGenerate');
const subtitlesAutoTranscriptEl = document.getElementById('subtitlesAutoTranscript');
const peopleClearBtnEl = document.getElementById('peopleClear');
const peopleMetaEl = document.getElementById('peopleMeta');
const subtitlesMetaEl = document.getElementById('subtitlesMeta');
const peopleEditorEl = document.getElementById('peopleEditor');
const subtitlesEditorEl = document.getElementById('subtitlesEditor');
const splitEditorEl = document.getElementById('splitEditor');
const reviewStatsEl = document.getElementById('reviewStats');
const frameGridEl = document.getElementById('frameGrid');
const summaryBoxEl = document.getElementById('summaryBox');
const iqrSparkEl = document.getElementById('iqrSpark');
const timelineScrubEl = document.getElementById('timelineScrub');
const timelineAudioEl = document.getElementById('timelineAudio');
const timelineAudioPlayBtnEl = document.getElementById('timelineAudioPlay');
const timelineAudioTrackEl = document.getElementById('timelineAudioTrack');
const timelineAudioWaveEl = document.getElementById('timelineAudioWave');
const timelineAudioPlayheadEl = document.getElementById('timelineAudioPlayhead');
const timelineScrubMetaEl = document.getElementById('timelineScrubMeta');
const timelineScrubWrapEl = timelineScrubEl ? timelineScrubEl.closest('.timeline-scrub') : null;
const peopleTimelineEl = document.getElementById('peopleTimeline');
const sparkMetaEl = document.getElementById('sparkMeta');
const gammaRangeMetaEl = document.getElementById('gammaRangeMeta');
const sparkPlayBtnEl = document.getElementById('sparkPlayBtn');
const page2El = document.getElementById('page2');
const reviewLayoutEl = page2El ? page2El.querySelector('.page-review-layout') : null;
const reviewTopEl = page2El ? page2El.querySelector('.review-top') : null;
const sparkPanelEl = page2El ? page2El.querySelector('.spark-panel') : null;
const fullscreenBtnEl = document.getElementById('toggleFullscreen');
const reviewControlsEl = document.getElementById('reviewControls');
const gammaControlsEl = document.getElementById('gammaControls');
const simulateFreezeFrameEl = document.getElementById('simulateFreezeFrame');
const forceAllFramesGoodEl = document.getElementById('forceAllFramesGood');
const flipbookSimFreezeFrameEl = document.getElementById('flipbookSimFreezeFrame');
const flipbookSimFreezeWrapEl = flipbookSimFreezeFrameEl ? flipbookSimFreezeFrameEl.closest('label.sim-freeze') : null;
const previewRenderBtnEl = document.getElementById('previewRender');
const flipbookPreviewEl = document.getElementById('flipbookPreview');
const flipbookMetaEl = document.getElementById('flipbookMeta');
const flipbookFrameEl = document.getElementById('flipbookFrame');
const flipbookImageEl = document.getElementById('flipbookImage');
const flipbookSubtitlesEl = document.getElementById('flipbookSubtitles');
const flipbookSubtitleRailEl = document.getElementById('flipbookSubtitleRail');
const flipbookAudioEl = document.getElementById('flipbookAudio');
const flipbookVolumeEl = document.getElementById('flipbookVolume');
const flipbookRevBtnEl = document.getElementById('flipbookRevBtn');
const flipbookPlayBtnEl = document.getElementById('flipbookPlayBtn');
const flipbookFwdBtnEl = document.getElementById('flipbookFwdBtn');
const flipbookCloseBtnEl = document.getElementById('flipbookCloseBtn');
const saveProgressDraftBtnEl = document.getElementById('saveProgressDraft');
const loadChapterBtnEl = document.getElementById('loadChapterBtn');
const backToChaptersBtnEl = document.getElementById('backToChaptersBtn');
const loadSpinnerEl = document.getElementById('loadSpinner');

let autoIqrTimer = null;
let visibleRangeRefreshPending = false;
let loadProgressPollTimer = null;
let loadProgressPollInFlight = false;
let previewProgressPollTimer = null;
let subtitlesProgressPollTimer = null;
let loadSampleTimingStartAt = 0;
let loadSampleTimingStartDone = 0;
let overlayProgressStartedAt = 0;
let progressiveReviewFrameCount = 0;
let progressiveLoadOverlayDismissed = false;
let imageFetchedRanges = []; // sorted [{start, end}] frame index ranges fetched for display
let imageFetchVersion = 0;   // incremented on each change — used to bust sparkline cache
const chapterSparkSvgCache = new Map(); // key → svg string, keyed by chapter data hash
const chapterButtonCache = new Map();   // title → button element
let loadProgressMessage = '';
let loadProgressDone = 0;
let loadProgressTotal = 0;
let lastLoadReviewPollAt = 0;
let lastLoadReviewSampleDone = -1;
let isChapterLoadInFlight = false;
let isPreviewRenderInFlight = false;
let isSubtitlesGenerateInFlight = false;
let activeCancelableTask = '';
let sparkDragPointerId = null;
let sparkDragRaf = null;
let sparkDragClientX = 0;
let sparkPlayTimer = null;
let sparkPlayFrames = [];
let sparkPlayIndex = 0;
let sparkPlayUseAudioClock = false;
let sparkPlayLastTickMs = 0;
let sparkPlayAccumulatorMs = 0;
let sparkPlayAudioClockLastTime = -1;
let sparkPlayAudioClockStallCount = 0;
let flipbookAudioSrcKey = '';
let flipbookGridCursorIndex = -1;
let flipbookSubtitleRailRenderKey = '';
let flipbookSubtitleRailActiveIndex = -1;
let flipbookSubtitleRailManualUntilMs = 0;
let flipbookSubtitleRailProgrammaticScroll = false;
let flipbookToggleInFlight = false;
let peopleTimelineRenderState = null;
let peopleTimelineDraft = null;
let subtitleTimelineDraft = null;
let peopleTimelineDrag = null;
let peopleTimelineZoom = 1.0;
let timelineAudioCtx = null;
let timelineAudioWavePeaks = [];
let timelineAudioWaveDurationSec = 0;
let timelineAudioWaveKey = '';
let timelineAudioWaveLoading = false;
let timelineAudioScrubPointerId = null;
let timelineAudioScrubActive = false;
let timelineAudioScrubWasPlaying = false;
let timelineAudioLastScrubSeconds = 0;
let timelineAudioLastScrubAt = 0;
let frameGridSizerEl = null;
let frameGridItemsEl = null;
let frameGridRenderedStart = -1;
let frameGridRenderedEnd = -1;
let frameGridRenderedLayoutKey = '';
let frameSheetPrefetchDone = new Set();
let frameSheetPrefetchPending = new Map();
let frameSheetImageObjects = new Map(); // contact sheet URL -> loaded Image (kept alive for canvas drawImage)
const pendingToggleRequests = new Set();
const navActionButtons = new Map();
const lockExtraButtons = [
  loadChapterBtnEl,
  previewRenderBtnEl,
  gammaApplyVisibleBtnEl,
  gammaClearBtnEl,
  peoplePrefillCastBtnEl,
  subtitlesGenerateBtnEl,
  peopleClearBtnEl,
  saveProgressDraftBtnEl,
  forceAllFramesGoodEl,
].filter((el) => Boolean(el));
let shiftRangeAnchorFid = null;
