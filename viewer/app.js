"use strict";

const CONFIG_PATH = "./config.json";
const GALLERY_PATH = "./gallery.json";
const IMAGE_CACHE_LIMIT = 200;
const PRELOAD_RADIUS = 12;
const TURN_DURATION_MS = 1248;
const TURN_MAX_ANGLE = 164;

const state = {
  albums: [],
  album: null,
  mode: "grid",
  bookIndex: 0,
  turning: false,
  imageCache: new Map(),
  turnRafId: 0,
};

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function renderStatus(text, isError = false) {
  const status = document.getElementById("status");
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function mediaUrlForItem(item) {
  if (typeof item.path === "string" && item.path.trim()) {
    const id = String(item.id || "").trim();
    if (!id) {
      return "";
    }
    return `./media?id=${encodeURIComponent(id)}`;
  }
  return typeof item.url === "string" ? item.url : "";
}

function mediaElementForItem(item, options = {}) {
  const url = mediaUrlForItem(item);
  if (!url) {
    return null;
  }
  let media;
  if (item.type === "video") {
    media = document.createElement("video");
    media.controls = true;
    media.preload = "metadata";
    media.src = url;
  } else {
    media = document.createElement("img");
    media.loading = options.eager ? "eager" : "lazy";
    if (options.eager) {
      media.fetchPriority = "high";
    }
    media.decoding = "async";
    media.src = url;
    media.alt = item.title || "Image";
  }
  media.className = "media";
  media.referrerPolicy = "no-referrer";
  return media;
}

function buildCard(item, template) {
  const node = template.content.firstElementChild.cloneNode(true);
  const mediaWrap = node.querySelector(".media-wrap");
  const title = node.querySelector(".item-title");
  const caption = node.querySelector(".item-caption");
  const sourceLink = node.querySelector(".item-link");
  const embedLink = node.querySelector(".embed-link");

  title.textContent = item.title || "Untitled";
  caption.textContent = item.caption || `${item.source || "unknown"} ${item.type || "media"}`;

  const mediaUrl = mediaUrlForItem(item);
  sourceLink.href = item.shareUrl || mediaUrl || "#";
  sourceLink.textContent = item.path ? "Open file" : "Open source";

  if (item.embedUrl) {
    embedLink.href = item.embedUrl;
  } else {
    embedLink.remove();
  }

  const media = mediaElementForItem(item);
  if (!media) {
    const fallback = document.createElement("p");
    fallback.className = "fallback";
    fallback.textContent = "Missing item URL/path.";
    mediaWrap.appendChild(fallback);
    return node;
  }

  media.addEventListener("error", () => {
    clearNode(mediaWrap);
    mediaWrap.classList.add("media-error");
    const fallback = document.createElement("p");
    fallback.className = "fallback";
    fallback.textContent = "Media failed to load.";
    mediaWrap.appendChild(fallback);
  });
  mediaWrap.appendChild(media);
  return node;
}

function renderAlbums(albums) {
  const select = document.getElementById("albumSelect");
  clearNode(select);
  albums.forEach((album, index) => {
    const option = document.createElement("option");
    option.value = album.id || `album-${index + 1}`;
    option.textContent = album.title || `Album ${index + 1}`;
    select.appendChild(option);
  });
}

function getItems(album) {
  if (!album || !Array.isArray(album.items)) {
    return [];
  }
  return album.items;
}

function getPhotobookItems(album) {
  return getItems(album).filter((item) => item.type === "image");
}

function clampSpreadIndex(items, index) {
  if (items.length === 0) {
    return 0;
  }
  const maxStart = Math.max(0, items.length - (items.length % 2 === 0 ? 2 : 1));
  const evenIndex = Math.max(0, index - (index % 2));
  return Math.min(evenIndex, maxStart);
}

function setMode(mode) {
  state.mode = mode === "photobook" ? "photobook" : "grid";
  const grid = document.getElementById("grid");
  const book = document.getElementById("photobook");
  const bookControls = document.getElementById("bookControls");
  grid.classList.toggle("hidden", state.mode !== "grid");
  book.classList.toggle("hidden", state.mode !== "photobook");
  bookControls.classList.toggle("hidden", state.mode !== "photobook");
  if (state.mode !== "photobook") {
    resetTurnLayer();
  }
}

function isFullscreenActive() {
  return Boolean(document.fullscreenElement);
}

async function togglePhotobookFullscreen() {
  const book = document.getElementById("photobook");
  const button = document.getElementById("fullscreenBtn");
  try {
    if (!isFullscreenActive()) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (_error) {
    // Fallback for environments that block Fullscreen API.
    book.classList.toggle("fullscreen");
  }
  if (button) {
    button.textContent = isFullscreenActive() || book.classList.contains("fullscreen") ? "Exit Fullscreen" : "Fullscreen";
  }
}

function renderGrid(album) {
  const grid = document.getElementById("grid");
  clearNode(grid);
  const template = document.getElementById("cardTemplate");
  const items = getItems(album);
  if (items.length === 0) {
    renderStatus("This album has no media items yet.");
    return;
  }
  renderStatus(`${items.length} item(s) in "${album.title}"`);
  items.forEach((item) => {
    grid.appendChild(buildCard(item, template));
  });
}

function renderBookPage(item, wrapId, titleId, captionId) {
  const wrap = document.getElementById(wrapId);
  clearNode(wrap);
  if (!item) {
    const blank = document.createElement("p");
    blank.className = "fallback";
    blank.textContent = "Blank page";
    wrap.appendChild(blank);
  } else {
    const media = mediaElementForItem(item, { eager: true });
    if (!media) {
      const fallback = document.createElement("p");
      fallback.className = "fallback";
      fallback.textContent = "Missing image path/url.";
      wrap.appendChild(fallback);
    } else {
      wrap.appendChild(media);
    }
  }
  if (titleId) {
    setText(titleId, item ? item.title || "Untitled" : "");
  }
  if (captionId) {
    setText(captionId, item ? item.caption || "" : "");
  }
}

function primeImage(item) {
  if (!item || item.type !== "image") {
    return;
  }
  const url = mediaUrlForItem(item);
  if (!url) {
    return;
  }
  if (state.imageCache.has(url)) {
    const existing = state.imageCache.get(url);
    state.imageCache.delete(url);
    state.imageCache.set(url, existing);
    return;
  }
  const img = new Image();
  img.decoding = "async";
  img.loading = "eager";
  img.addEventListener(
    "load",
    () => {
      if (typeof img.decode === "function") {
        img.decode().catch(() => {});
      }
    },
    { once: true }
  );
  img.src = url;
  state.imageCache.set(url, img);

  while (state.imageCache.size > IMAGE_CACHE_LIMIT) {
    const oldestKey = state.imageCache.keys().next().value;
    if (!oldestKey) {
      break;
    }
    state.imageCache.delete(oldestKey);
  }
}

function preloadSpreadWindow(items, centerIndex, radius = 6) {
  if (!Array.isArray(items) || items.length === 0) {
    return;
  }
  const low = Math.max(0, centerIndex - radius);
  const high = Math.min(items.length - 1, centerIndex + radius + 1);
  for (let i = low; i <= high; i += 1) {
    primeImage(items[i]);
  }
}

function updateBookControls(items) {
  const prev = document.getElementById("prevPageBtn");
  const next = document.getElementById("nextPageBtn");
  const label = document.getElementById("pageLabel");
  const spreadCount = Math.ceil(items.length / 2);
  const currentSpread = spreadCount > 0 ? Math.floor(state.bookIndex / 2) + 1 : 0;
  label.textContent = `Spread ${currentSpread} / ${spreadCount}`;
  prev.disabled = state.turning || state.bookIndex <= 0;
  next.disabled = state.turning || state.bookIndex + 2 >= items.length;
}

function renderSpread(items) {
  const left = items[state.bookIndex] || null;
  const right = items[state.bookIndex + 1] || null;
  renderBookPage(left, "leftMediaWrap", "leftTitle", "leftCaption");
  renderBookPage(right, "rightMediaWrap", "rightTitle", "rightCaption");
}

function renderPhotobook(album) {
  const items = getPhotobookItems(album);
  if (items.length === 0) {
    renderStatus(`"${album.title}" has no image pages for photobook mode.`);
    state.bookIndex = 0;
    renderSpread([]);
    updateBookControls(items);
    return;
  }
  state.bookIndex = clampSpreadIndex(items, state.bookIndex);
  renderStatus(`Photobook: "${album.title}" (${items.length} page(s), ${Math.ceil(items.length / 2)} spread(s))`);
  renderSpread(items);
  preloadSpreadWindow(items, state.bookIndex, PRELOAD_RADIUS);
  updateBookControls(items);
}

function easeInOutCubic(t) {
  if (t < 0.5) {
    return 4 * t * t * t;
  }
  return 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function setTurnMediaFrame(frontMedia, backMedia, direction, progress) {
  const dir = direction === "forward" ? 1 : -1;
  const skew = dir * progress * 3.4;
  const frontScale = 1 - progress * 0.26;
  const backScale = 0.76 + progress * 0.24;
  const frontBrightness = 1 - progress * 0.36;
  const backBrightness = 0.74 + progress * 0.3;

  if (frontMedia) {
    frontMedia.style.transform = `scaleX(${frontScale.toFixed(3)}) skewY(${skew.toFixed(2)}deg)`;
    frontMedia.style.filter = `brightness(${frontBrightness.toFixed(3)}) contrast(1.04)`;
  }
  if (backMedia) {
    backMedia.style.transform = `scaleX(${backScale.toFixed(3)}) skewY(${(-skew * 0.52).toFixed(2)}deg)`;
    backMedia.style.filter = `brightness(${backBrightness.toFixed(3)}) contrast(1.03)`;
  }
}

function resetTurnLayer() {
  if (state.turnRafId) {
    window.cancelAnimationFrame(state.turnRafId);
    state.turnRafId = 0;
  }
  const turnPage = document.getElementById("turnPage");
  const turnLeaf = document.getElementById("turnLeaf");
  const frontWrap = document.getElementById("turnFrontMediaWrap");
  const backWrap = document.getElementById("turnBackMediaWrap");
  const frontMedia = frontWrap.querySelector(".media");
  const backMedia = backWrap.querySelector(".media");

  if (frontMedia) {
    frontMedia.style.transform = "";
    frontMedia.style.filter = "";
  }
  if (backMedia) {
    backMedia.style.transform = "";
    backMedia.style.filter = "";
  }
  clearNode(frontWrap);
  clearNode(backWrap);
  setText("turnFrontTitle", "");
  setText("turnFrontCaption", "");
  setText("turnBackTitle", "");
  setText("turnBackCaption", "");
  turnPage.classList.remove("active", "side-left", "side-right");
  turnPage.style.removeProperty("--fold-alpha");
  turnPage.style.removeProperty("--fold-shift");
  turnLeaf.style.transform = "rotateY(0deg)";
}

function getTurnPair(items, direction, currentIndex, nextIndex) {
  if (direction === "forward") {
    return {
      sideClass: "side-right",
      frontItem: items[currentIndex + 1] || items[currentIndex] || null,
      backItem: items[nextIndex] || null,
      sign: -1,
    };
  }
  return {
    sideClass: "side-left",
    frontItem: items[currentIndex] || null,
    backItem: items[nextIndex + 1] || items[nextIndex] || null,
    sign: 1,
  };
}

function turnPage(direction) {
  if (state.turning || !state.album) {
    return;
  }
  const items = getPhotobookItems(state.album);
  if (items.length === 0) {
    return;
  }
  const delta = direction === "backward" ? -2 : 2;
  const nextIndex = clampSpreadIndex(items, state.bookIndex + delta);
  if (nextIndex === state.bookIndex) {
    return;
  }

  state.turning = true;
  updateBookControls(items);

  const currentIndex = state.bookIndex;
  const turnNode = document.getElementById("turnPage");
  const turnLeaf = document.getElementById("turnLeaf");
  const turnPair = getTurnPair(items, direction, currentIndex, nextIndex);

  renderBookPage(turnPair.frontItem, "turnFrontMediaWrap", "turnFrontTitle", "turnFrontCaption");
  renderBookPage(turnPair.backItem, "turnBackMediaWrap", "turnBackTitle", "turnBackCaption");
  const frontMedia = document.querySelector("#turnFrontMediaWrap .media");
  const backMedia = document.querySelector("#turnBackMediaWrap .media");

  turnNode.classList.remove("side-left", "side-right", "active");
  turnNode.classList.add(turnPair.sideClass, "active");
  turnNode.style.setProperty("--fold-alpha", "0.22");
  turnNode.style.setProperty("--fold-shift", "0%");

  preloadSpreadWindow(items, nextIndex, PRELOAD_RADIUS + 8);

  let didSwapSpread = false;
  const swapToDestinationSpread = () => {
    if (didSwapSpread) {
      return;
    }
    state.bookIndex = nextIndex;
    renderPhotobook(state.album);
    didSwapSpread = true;
  };

  const start = performance.now();
  const step = (now) => {
    const elapsed = now - start;
    const t = Math.min(1, elapsed / TURN_DURATION_MS);
    const eased = easeInOutCubic(t);
    const angle = turnPair.sign * TURN_MAX_ANGLE * eased;
    const foldStrength = Math.sin(eased * Math.PI);
    const foldAlpha = 0.14 + foldStrength * 0.58;
    const foldShift = (turnPair.sign * eased * 28).toFixed(2);

    turnLeaf.style.transform = `rotateY(${angle.toFixed(2)}deg)`;
    turnNode.style.setProperty("--fold-alpha", foldAlpha.toFixed(3));
    turnNode.style.setProperty("--fold-shift", `${foldShift}%`);
    setTurnMediaFrame(frontMedia, backMedia, direction, eased);

    if (!didSwapSpread && eased >= 0.52) {
      swapToDestinationSpread();
    }

    if (t < 1) {
      state.turnRafId = window.requestAnimationFrame(step);
      return;
    }

    if (!didSwapSpread) {
      swapToDestinationSpread();
    }
    resetTurnLayer();
    state.turning = false;
    updateBookControls(items);
  };

  state.turnRafId = window.requestAnimationFrame(step);
}

function renderAlbum(album) {
  if (state.turning) {
    state.turning = false;
    resetTurnLayer();
  }
  state.album = album;
  if (!album) {
    renderStatus("No album selected.", true);
    return;
  }
  if (state.mode === "photobook") {
    renderPhotobook(album);
  } else {
    renderGrid(album);
  }
}

function applyBrand(config) {
  setText("brand", config.brandName || "viewer");
  setText("title", config.siteTitle || "Media Viewer");
  setText("intro", config.intro || "Local viewer");
  if (typeof config.pageTitle === "string" && config.pageTitle.trim()) {
    document.title = config.pageTitle;
  }
  if (typeof config.defaultMode === "string") {
    setMode(config.defaultMode);
  }
}

function bindUi() {
  const modeSelect = document.getElementById("modeSelect");
  const albumSelect = document.getElementById("albumSelect");
  const prevBtn = document.getElementById("prevPageBtn");
  const nextBtn = document.getElementById("nextPageBtn");
  const fullscreenBtn = document.getElementById("fullscreenBtn");

  modeSelect.addEventListener("change", () => {
    if (state.turning) {
      state.turning = false;
      resetTurnLayer();
    }
    setMode(modeSelect.value);
    state.bookIndex = 0;
    renderAlbum(state.album);
  });

  albumSelect.addEventListener("change", () => {
    if (state.turning) {
      state.turning = false;
      resetTurnLayer();
    }
    const album = state.albums.find((a) => (a.id || "") === albumSelect.value) || state.albums[0];
    state.bookIndex = 0;
    renderAlbum(album);
  });

  prevBtn.addEventListener("click", () => turnPage("backward"));
  nextBtn.addEventListener("click", () => turnPage("forward"));
  fullscreenBtn.addEventListener("click", () => {
    if (state.mode === "photobook") {
      void togglePhotobookFullscreen();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (state.mode !== "photobook") {
      return;
    }
    if (event.key === "ArrowLeft") {
      turnPage("backward");
    } else if (event.key === "ArrowRight") {
      turnPage("forward");
    } else if (event.key.toLowerCase() === "f") {
      event.preventDefault();
      void togglePhotobookFullscreen();
    }
  });

  document.addEventListener("fullscreenchange", () => {
    const button = document.getElementById("fullscreenBtn");
    const book = document.getElementById("photobook");
    if (button) {
      button.textContent = isFullscreenActive() || book.classList.contains("fullscreen") ? "Exit Fullscreen" : "Fullscreen";
    }
  });
}

async function bootstrap() {
  try {
    const [config, gallery] = await Promise.all([fetchJson(CONFIG_PATH), fetchJson(GALLERY_PATH)]);
    applyBrand(config);
    bindUi();

    const modeSelect = document.getElementById("modeSelect");
    modeSelect.value = state.mode;

    state.albums = Array.isArray(gallery.albums) ? gallery.albums : [];
    if (state.albums.length === 0) {
      renderStatus("No albums found in gallery.json", true);
      return;
    }
    renderAlbums(state.albums);
    renderAlbum(state.albums[0]);
  } catch (error) {
    renderStatus(`Viewer failed to load: ${error.message}`, true);
  }
}

bootstrap();
