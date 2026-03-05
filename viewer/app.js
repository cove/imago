"use strict";

const CONFIG_PATH = "./config.json";
const GALLERY_PATH = "./gallery.json";

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

function mediaElementForItem(item) {
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
    media.loading = "lazy";
    media.decoding = "async";
    media.src = url;
    media.alt = item.title || "Image";
  }

  media.referrerPolicy = "no-referrer";
  media.className = "media";
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
  const captionBits = [];
  if (item.caption) {
    captionBits.push(item.caption);
  }
  if (item.path) {
    captionBits.push(`path: ${item.path}`);
  } else {
    captionBits.push(`${item.source || "unknown"} ${item.type || "media"}`);
  }
  caption.textContent = captionBits.join(" | ");

  const mediaUrl = mediaUrlForItem(item);
  sourceLink.href = item.shareUrl || mediaUrl || "#";
  sourceLink.textContent = item.path ? "Open file" : "Open source";

  if (item.embedUrl) {
    embedLink.href = item.embedUrl;
    embedLink.textContent = "Open embed";
  } else {
    embedLink.remove();
  }

  const media = mediaElementForItem(item);
  if (!media) {
    mediaWrap.classList.add("media-error");
    const fallback = document.createElement("p");
    fallback.className = "fallback";
    fallback.textContent = "Missing item URL/path. Set url or path in gallery.json.";
    mediaWrap.appendChild(fallback);
    return node;
  }

  media.addEventListener("error", () => {
    mediaWrap.classList.add("media-error");
    media.remove();
    const fallback = document.createElement("p");
    fallback.className = "fallback";
    fallback.textContent = "Direct media load failed. Use source or embed link.";
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

function renderGallery(album) {
  const grid = document.getElementById("grid");
  clearNode(grid);
  const template = document.getElementById("cardTemplate");

  const items = Array.isArray(album.items) ? album.items : [];
  if (items.length === 0) {
    renderStatus("This album has no media items yet.");
    return;
  }

  renderStatus(`${items.length} item(s) in "${album.title}"`);
  items.forEach((item) => {
    const card = buildCard(item, template);
    grid.appendChild(card);
  });
}

function applyBrand(config) {
  setText("brand", config.brandName || "viewer");
  setText("title", config.siteTitle || "Media Viewer");
  setText("intro", config.intro || "Static viewer for public cloud media links.");
  if (typeof config.pageTitle === "string" && config.pageTitle.trim()) {
    document.title = config.pageTitle;
  }
}

async function bootstrap() {
  try {
    const [config, gallery] = await Promise.all([
      fetchJson(CONFIG_PATH),
      fetchJson(GALLERY_PATH),
    ]);

    applyBrand(config);

    const albums = Array.isArray(gallery.albums) ? gallery.albums : [];
    if (albums.length === 0) {
      renderStatus("No albums found in gallery.json", true);
      return;
    }

    renderAlbums(albums);
    const select = document.getElementById("albumSelect");
    renderGallery(albums[0]);
    select.addEventListener("change", () => {
      const album = albums.find((a) => (a.id || "") === select.value);
      renderGallery(album || albums[0]);
    });
  } catch (error) {
    renderStatus(`Viewer failed to load: ${error.message}`, true);
  }
}

bootstrap();
