"""HTTP server for map-based XMP location correction."""

import json
import os
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from .lib.xmp_sidecar import read_ai_sidecar_state, write_xmp_sidecar, read_person_in_image, read_locations_shown
from .lib.ai_geocode import NominatimGeocoder
from .lib.ai_location import _xmp_gps_to_decimal

# We will store paths here globally for the simple HTTP server to access
_XMP_PATHS: list[Path] = []
_GEOCODER = NominatimGeocoder()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Imago Photo Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body, html { height: 100%; font-family: system-ui, sans-serif; background: #111; color: #ddd; overflow: hidden; }
  #app { display: flex; height: 100%; width: 100%; }
  #map { flex: 1; height: 100%; background: #000; }
  
  #sidebar {
    width: 380px; background: #1a1a1a; border-left: 1px solid #333;
    display: flex; flex-direction: column; z-index: 1000; box-shadow: -4px 0 15px rgba(0,0,0,0.5);
  }
  .sidebar-header { padding: 16px; background: #222; border-bottom: 1px solid #333; }
  .sidebar-header h1 { font-size: 16px; font-weight: 600; color: #eee; margin-bottom: 4px; }
  .sidebar-header p { font-size: 12px; color: #888; }
  .sidebar-content { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }
  
  .details-card { background: #111; border: 1px solid #333; border-radius: 8px; overflow: hidden; }
  .details-card img { width: 100%; max-height: 250px; object-fit: contain; background: #000; display: block; }
  .details-card .info { padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .details-card h2 { font-size: 14px; color: #ccc; word-break: break-all; }
  
  .picker-nav { display: flex; align-items: center; justify-content: space-between; padding: 10px; background: #1d2940; border-bottom: 1px solid #333; }
  .picker-nav button { background: #273754; color: #fff; border: 1px solid #4a7fc1; border-radius: 4px; padding: 4px 12px; cursor: pointer; font-weight: bold; }
  .picker-nav button:hover { background: #32466a; }
  .picker-nav span { font-size: 12px; color: #dbe7ff; font-weight: bold; }

  .search-box { display: flex; gap: 8px; margin-top: 12px; }
  .search-box input { flex: 1; padding: 8px; border-radius: 4px; border: 1px solid #444; background: #222; color: #fff; font-size: 13px; }
  .search-box button { padding: 8px 12px; border-radius: 4px; border: 1px solid #4a7fc1; background: #1d2940; color: #dbe7ff; font-weight: 600; cursor: pointer; }
  .search-box button:hover { background: #243046; }
  .search-box button:disabled { opacity: 0.5; cursor: not-allowed; }

  .details-card .tag { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #666; font-weight: 600; }
  .details-card p { font-size: 13px; color: #aaa; }
  .details-card .empty { font-size: 12px; color: #555; font-style: italic; text-align: center; padding: 20px; }
  
  .toast-container { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 2000; display: flex; flex-direction: column; gap: 10px; }
  .toast { background: rgba(20, 20, 20, 0.9); backdrop-filter: blur(8px); border: 1px solid #444; border-radius: 8px; padding: 12px 20px; color: #fff; font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); animation: slideUp 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
  .toast.success { border-bottom: 3px solid #3fb950; }
  .toast.error { border-bottom: 3px solid #f85149; }
  @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

  .leaflet-popup-content { font-family: system-ui, sans-serif; color: #eee; }
  .leaflet-popup-content-wrapper { background: #222; border: 1px solid #444; border-radius: 8px; }
  .leaflet-popup-tip { background: #222; border: 1px solid #444; border-top: none; border-left: none; }
  .popup-internal { display: flex; flex-direction: column; gap: 6px; }
  .popup-internal img { width: 150px; object-fit: contain; background: #000; border-radius: 4px; }
  .leaflet-container a.leaflet-popup-close-button { color: #aaa; }

  .undo-btn { 
    position: absolute; bottom: 20px; left: 20px; z-index: 1000;
    background: #1d2940; border: 1px solid #4a7fc1; color: #dbe7ff; 
    padding: 8px 16px; border-radius: 8px; font-weight: 600; cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none;
  }
  .undo-btn:hover { background: #243046; }

  /* Custom Dot Marker */
  .dot-marker {
    background-color: #4a8cf8;
    border: 2px solid #fff;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
    cursor: grab;
    transition: transform 0.1s;
  }
  .dot-marker:hover {
    transform: scale(1.2);
    background-color: #74a7ff;
  }
  .dot-marker:active {
    cursor: grabbing;
  }

  /* Pulse animation for saving state */
  .saving { animation: pulse 1s infinite alternate; opacity: 0.5; pointer-events: none; }
  @keyframes pulse { from { opacity: 0.5; } to { opacity: 1; } }

  /* Lightbox */
  .lightbox {
    position: absolute; top: 20px; left: 20px; right: 400px; bottom: 20px;
    background: rgba(10, 10, 10, 0.95); z-index: 3000; display: flex; flex-direction: column;
    border-radius: 12px; border: 1px solid #444; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.8);
  }
  .lightbox-nav {
    display: flex; align-items: center; justify-content: center; gap: 15px; padding: 15px; background: #1a1a1a;
  }
  .lightbox-nav button {
    background: #273754; color: #fff; border: 1px solid #4a7fc1; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: bold;
  }
  .lightbox-nav button:hover { background: #32466a; }
  .lightbox-nav span { color: #dbe7ff; font-weight: bold; font-size: 14px; }
  .lightbox-content {
    flex: 1; overflow: hidden; display: flex; align-items: center; justify-content: center; position: relative;
    background: #050505;
  }
  .lightbox-content img {
    max-width: 100%; max-height: 100%; cursor: zoom-in; transition: transform 0.05s linear; transform-origin: top left;
  }

  @media (max-width: 768px) { #app { flex-direction: column; } #sidebar { width: 100%; height: 40%; border-left: none; border-top: 1px solid #333; } .lightbox { right: 20px; bottom: calc(40% + 20px); } }
</style>
</head>
<body>
<div id="app">
  <div id="map"></div>
  <div id="sidebar">
    <div class="sidebar-header">
      <h1>Imago Photo Map</h1>
      <p>Drag photos to correct exact GPS tags</p>
    </div>
    <div class="sidebar-content" id="sidebar-content">
      <div class="details-card">
        <div class="empty">Select a marker on the map to view XMP details.</div>
      </div>
    </div>
  </div>
</div>

<div id="lightbox" class="lightbox" style="display: none;">
  <div class="lightbox-nav">
    <button onclick="prevCycleItem()">&lt; Prev Photo</button>
    <span id="lightbox-counter">1 of 1</span>
    <button onclick="nextCycleItem()">Next Photo &gt;</button>
    <button onclick="closeLightbox()" style="margin-left: auto; background: #444; border-color: #666;">Close [X]</button>
  </div>
  <div class="lightbox-content" id="lightbox-content">
    <img id="lightbox-img" src="" alt="Full Screen Preview" draggable="false" />
  </div>
</div>

<button class="undo-btn" id="undo-btn" onclick="undoLastMove()">Undo Last Move</button>
<div class="toast-container" id="toast-container"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
  const worldBounds = [
    [-90, -180],
    [90, 180]
  ];
  
  const map = L.map('map', {
    maxBounds: worldBounds,
    maxBoundsViscosity: 1.0,
    zoomSnap: 0.5,
    zoomDelta: 0.5
  });
  
  function enforceMinZoom() {
    const w = document.getElementById('map').clientWidth;
    const h = document.getElementById('map').clientHeight;
    // Map width/height at zoom z is 256 * 2^z
    const minZoomW = Math.log2(w / 256);
    const minZoomH = Math.log2(h / 256);
    // Ceiling to nearest 0.5 step to ensure it covers bounds smoothly without excessive zooming
    const baseZoom = Math.max(minZoomW, minZoomH);
    const snapZoom = Math.ceil(baseZoom * 2) / 2;
    map.setMinZoom(snapZoom);
  }
  
  window.addEventListener('resize', enforceMinZoom);
  enforceMinZoom();
  map.setView([0, 0], map.getMinZoom() || 2);
  
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    subdomains: 'abcd',
    maxZoom: 20,
    noWrap: true,
    bounds: worldBounds
  }).addTo(map);

  const markersMap = new Map();
  const allItemsData = new Map();
  let selectedMarkerKey = null;
  const undoStack = [];
  
  let currentLocationItems = [];
  let currentLocationIndex = 0;

  function updateUndoButton() {
    const btn = document.getElementById('undo-btn');
    if (undoStack.length > 0) {
      btn.style.display = 'block';
      btn.textContent = `Undo Last Move (${undoStack.length})`;
    } else {
      btn.style.display = 'none';
    }
  }

  async function undoLastMove() {
    if (undoStack.length === 0) return;
    const last = undoStack.pop();
    updateUndoButton();
    const marker = markersMap.get(last.path);
    if (!marker) return;
    const el = marker.getElement();
    if (el) el.classList.add('saving');
    
    try {
      const resp = await fetch('/api/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: last.path,
          lat: last.lat,
          lon: last.lon,
          city: last.city,
          state: last.state,
          country: last.country,
          undoing: true
        })
      });
      const result = await resp.json();
      if (result.ok) {
        marker.setLatLng([last.lat, last.lon]);
        showToast('Successfully reverted location.');
        if (selectedMarkerKey === last.path) {
          const item = Object.assign({}, last);
          item.city = result.city;
          item.state = result.state;
          item.country = result.country;
          renderSidebar(item);
        }
      } else {
        showToast(result.error || 'Undo failed', 'error');
        undoStack.push(last); // restore to stack
        updateUndoButton();
      }
    } catch (err) {
      showToast(err.message, 'error');
      undoStack.push(last);
      updateUndoButton();
    } finally {
      if (el) el.classList.remove('saving');
    }
  }

  function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function getImageUrl(path) {
    if (!path) return '';
    return `/api/image?path=${encodeURIComponent(path)}`;
  }

  function renderSidebar(data) {
    const content = document.getElementById('sidebar-content');
    if (!data) {
      content.innerHTML = `<div class="details-card"><div class="empty">Select a marker on the map to view XMP details.</div></div>`;
      return;
    }
    const safe = (val) => val ? String(val).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') : '';
    
    // Attempt to render image, plus relevant OCR or description details
    const locText = [data.city, data.state, data.country].filter(Boolean).join(', ');

    let navHtml = '';
    if (currentLocationItems.length > 1) {
      navHtml = `
        <div class="picker-nav">
          <button onclick="prevCycleItem()">&lt; Prev</button>
          <span>${currentLocationIndex + 1} of ${currentLocationItems.length} photos here</span>
          <button onclick="nextCycleItem()">Next &gt;</button>
        </div>
      `;
    }

    content.innerHTML = `
      <div class="details-card">
        ${navHtml}
        <img src="${getImageUrl(data.path)}" alt="Preview" onerror="this.style.display='none'">
        <div class="info">
          <h2>${safe(data.path.split(/[\\/]/).pop())}</h2>
          <div>
            <div class="tag">Address</div>
            <p>${safe(locText) || '(Unknown)'}</p>
            <div class="search-box">
              <input type="text" id="location-search" placeholder="Type new location..." onkeydown="if(event.key==='Enter') searchLocation()">
              <button id="search-btn" onclick="searchLocation()">Move Here</button>
            </div>
          </div>
          <div>
            <div class="tag">OCR Text</div>
            <p>${safe(data.ocr_text) || '(No OCR text)'}</p>
          </div>
          <div>
            <div class="tag">Description</div>
            <p>${safe(data.description) || '(No description)'}</p>
          </div>
        </div>
      </div>
    `;
  }

  function bringMarkerToFront(path) {
    markersMap.forEach((m, key) => {
      if (key === path) m.setZIndexOffset(1000);
      else m.setZIndexOffset(0);
    });
  }

  let lightboxZoom = 1;
  let lbX = 0, lbY = 0;
  let isDraggingLb = false;
  let startX = 0, startY = 0, dragDist = 0;

  function closeLightbox() { 
      document.getElementById('lightbox').style.display='none'; 
      resetLightboxTransform();
  }

  function openLightbox() {
      document.getElementById('lightbox').style.display='flex';
      resetLightboxTransform();
      updateLightbox();
  }

  function resetLightboxTransform() {
      lightboxZoom = 1; lbX = 0; lbY = 0;
      const img = document.getElementById('lightbox-img');
      if (img) {
          img.style.transform = `translate(0px, 0px) scale(1)`;
          img.style.cursor = 'zoom-in';
      }
  }

  function updateLightbox() {
      if(currentLocationItems.length === 0) return;
      const item = currentLocationItems[currentLocationIndex];
      document.getElementById('lightbox-img').src = getImageUrl(item.path);
      document.getElementById('lightbox-counter').textContent = `${currentLocationIndex + 1} of ${currentLocationItems.length}`;
  }

  function updateLightboxZoom(img) {
      img.style.transform = `translate(${lbX}px, ${lbY}px) scale(${lightboxZoom})`;
      img.style.cursor = lightboxZoom > 1 ? (isDraggingLb ? 'grabbing' : 'grab') : 'zoom-in';
  }

  const lbContent = document.getElementById('lightbox-content');
  const lbImg = document.getElementById('lightbox-img');

  lbContent.addEventListener('wheel', (e) => {
      e.preventDefault();
      const delta = e.deltaY * -0.002;
      const newScale = Math.min(Math.max(1, lightboxZoom + delta), 20);
      if (newScale === lightboxZoom) return;

      const rect = lbImg.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const scaleRatio = newScale / lightboxZoom;
      lbX -= mouseX * (scaleRatio - 1);
      lbY -= mouseY * (scaleRatio - 1);
      lightboxZoom = newScale;
      
      if (lightboxZoom === 1) { lbX = 0; lbY = 0; }
      updateLightboxZoom(lbImg);
  });

  lbContent.addEventListener('mousedown', (e) => {
     if (lightboxZoom <= 1 && e.target !== lbImg) return;
     e.preventDefault(); // Prevent native browser dragging of the <img> tag
     isDraggingLb = true;
     dragDist = 0;
     startX = e.clientX - lbX;
     startY = e.clientY - lbY;
     if (lightboxZoom > 1) lbImg.style.cursor = 'grabbing';
     lbImg.style.transition = 'none';
  });

  window.addEventListener('mousemove', (e) => {
     if (!isDraggingLb) return;
     e.preventDefault();
     dragDist += Math.abs(e.movementX) + Math.abs(e.movementY);
     if (lightboxZoom > 1) {
         lbX = e.clientX - startX;
         lbY = e.clientY - startY;
         lbImg.style.transform = `translate(${lbX}px, ${lbY}px) scale(${lightboxZoom})`;
     }
  });

  window.addEventListener('mouseup', () => {
     if (!isDraggingLb) return;
     isDraggingLb = false;
     updateLightboxZoom(lbImg);
     lbImg.style.transition = 'transform 0.05s linear';
  });

  lbImg.addEventListener('click', (e) => {
      e.stopPropagation();
      if (dragDist > 5) return;
      if (lightboxZoom > 1) {
          resetLightboxTransform();
          return;
      }
      
      const rect = lbImg.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      
      const newScale = 2.5;
      const scaleRatio = newScale / lightboxZoom;
      lbX -= mouseX * (scaleRatio - 1);
      lbY -= mouseY * (scaleRatio - 1);
      lightboxZoom = newScale;
      updateLightboxZoom(lbImg);
  });

  function prevCycleItem() {
    if (currentLocationItems.length <= 1) return;
    currentLocationIndex = (currentLocationIndex - 1 + currentLocationItems.length) % currentLocationItems.length;
    const item = currentLocationItems[currentLocationIndex];
    selectedMarkerKey = item.path;
    bringMarkerToFront(item.path);
    renderSidebar(item);
    if (document.getElementById('lightbox').style.display !== 'none') {
        resetLightboxTransform();
        updateLightbox();
    }
  }

  function nextCycleItem() {
    if (currentLocationItems.length <= 1) return;
    currentLocationIndex = (currentLocationIndex + 1) % currentLocationItems.length;
    const item = currentLocationItems[currentLocationIndex];
    selectedMarkerKey = item.path;
    bringMarkerToFront(item.path);
    renderSidebar(item);
    if (document.getElementById('lightbox').style.display !== 'none') {
        resetLightboxTransform();
        updateLightbox();
    }
  }

  async function searchLocation() {
    const input = document.getElementById('location-search');
    const btn = document.getElementById('search-btn');
    if (!input || !input.value.trim() || !selectedMarkerKey) return;
    
    input.disabled = true;
    btn.disabled = true;
    btn.textContent = 'Searching...';

    const marker = markersMap.get(selectedMarkerKey);
    const itemData = { path: selectedMarkerKey, lat: marker.getLatLng().lat, lon: marker.getLatLng().lng };

    if (marker) {
      const el = marker.getElement();
      if (el) el.classList.add('saving');
    }

    try {
      const resp = await fetch('/api/geocode_and_update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: selectedMarkerKey,
          query: input.value.trim()
        })
      });
      const result = await resp.json();
      if (result.ok) {
        undoStack.push({
           path: itemData.path,
           lat: itemData.lat,
           lon: itemData.lon,
           city: '', // Don't have prev city easily, lat/lon is used for undo
           state: '',
           country: ''
        });
        updateUndoButton();

        marker.setLatLng([result.lat, result.lon]);
        map.panTo([result.lat, result.lon]);
        showToast(`Moved to ${result.city || result.country || 'new location'}`);
        
        const originalItem = allItemsData.get(selectedMarkerKey);
        if (originalItem) {
           originalItem.lat = result.lat;
           originalItem.lon = result.lon;
           originalItem.city = result.city;
           originalItem.state = result.state;
           originalItem.country = result.country;
           renderSidebar(originalItem);
        }
      } else {
        showToast(result.error || 'Geocode failed', 'error');
      }
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      if (marker) {
        const el = marker.getElement();
        if (el) el.classList.remove('saving');
      }
      if (input) input.disabled = false;
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Move Here';
      }
    }
  }

  async function loadMarkers() {
    try {
      const res = await fetch('/api/markers');
      const data = await res.json();
      
      const bounds = L.latLngBounds();
      let hasValidCoords = false;

      const customIcon = L.divIcon({
        className: 'dot-marker',
        iconSize: [12, 12],
        iconAnchor: [6, 6],
        popupAnchor: [0, -8]
      });

      data.forEach(item => {
        allItemsData.set(item.path, item);
        const marker = L.marker([item.lat, item.lon], { draggable: true, icon: customIcon }).addTo(map);
        
        marker.on('click', () => {
          selectedMarkerKey = item.path;
          
          // Gather ALL overlapping pins for cycling
          currentLocationItems = [];
          allItemsData.forEach(existing => {
             if (Math.abs(existing.lat - item.lat) < 0.00001 && Math.abs(existing.lon - item.lon) < 0.00001) {
                 currentLocationItems.push(existing);
             }
          });
          currentLocationIndex = currentLocationItems.findIndex(existing => existing.path === item.path);
          if (currentLocationIndex === -1) currentLocationIndex = 0;
          
          bringMarkerToFront(item.path);
          renderSidebar(item);
          openLightbox();
        });

        marker.on('dragstart', () => {
          map.closePopup();
        });

        marker.on('dragend', async (e) => {
          const newPos = marker.getLatLng();
          const el = marker.getElement();
          
          undoStack.push({
             path: item.path,
             lat: item.lat,
             lon: item.lon,
             city: item.city,
             state: item.state,
             country: item.country
          });
          updateUndoButton();

          if (el) el.classList.add('saving');
          
          try {
            const resp = await fetch('/api/update', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                path: item.path,
                lat: newPos.lat,
                lon: newPos.lng
              })
            });
            const result = await resp.json();
            if (result.ok) {
              item.lat = newPos.lat;
              item.lon = newPos.lng;
              item.city = result.city;
              item.state = result.state;
              item.country = result.country;
              showToast(`Updated location to ${result.city || result.country || 'new coordinates'}`);
              if (selectedMarkerKey === item.path) renderSidebar(item);
            } else {
              showToast(result.error || 'Update failed', 'error');
              marker.setLatLng([item.lat, item.lon]); // Revert
              undoStack.pop();
              updateUndoButton();
            }
          } catch (err) {
            showToast(err.message, 'error');
            marker.setLatLng([item.lat, item.lon]); // Revert
            undoStack.pop();
            updateUndoButton();
          } finally {
            if (el) el.classList.remove('saving');
          }
        });

        markersMap.set(item.path, marker);
        if (item.lat !== 0 || item.lon !== 0) {
          bounds.extend([item.lat, item.lon]);
          hasValidCoords = true;
        }
      });

      if (hasValidCoords) {
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });
      }
    } catch (err) {
      showToast('Failed to load markers: ' + err.message, 'error');
    }
  }

  loadMarkers();
</script>
</body>
</html>
"""


class MapHandler(BaseHTTPRequestHandler):
    protocol_version: ClassVar[str] = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send_html(HTML)
            return

        if path == "/api/markers":
            self._send_markers()
            return

        if path == "/api/image":
            self._serve_image(query)
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/update":
            self._handle_update()
            return
        if parsed.path == "/api/geocode_and_update":
            self._handle_geocode_and_update()
            return
        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def _send_markers(self) -> None:
        results = []
        for p in _XMP_PATHS:
            if not p.is_file():
                continue
            state = read_ai_sidecar_state(p)
            if not state:
                continue

            lat_str = str(state.get("gps_latitude") or "").strip()
            lon_str = str(state.get("gps_longitude") or "").strip()

            if lat_str and lon_str:
                try:
                    lat = _xmp_gps_to_decimal(lat_str, axis="lat")
                    lon = _xmp_gps_to_decimal(lon_str, axis="lon")
                except Exception:
                    lat, lon = 0.0, 0.0
            else:
                lat, lon = 0.0, 0.0

            city = str(state.get("location_city", "") or "")

            results.append(
                {
                    "path": str(p),
                    "lat": float(lat),
                    "lon": float(lon),
                    "city": city,
                    "state": str(state.get("location_state", "") or ""),
                    "country": str(state.get("location_country", "") or ""),
                    "ocr_text": str(state.get("ocr_text", "") or ""),
                    "description": str(state.get("description", "") or ""),
                    "locations_shown": read_locations_shown(p),
                }
            )
        self._send_json(results)

    def _serve_image(self, query: dict[str, list[str]]) -> None:
        path_values = query.get("path") or []
        image_path = str(path_values[0] if path_values else "").strip()
        if not image_path:
            self._send_json({"error": "Missing path parameter"}, status=HTTPStatus.BAD_REQUEST)
            return

        xmp_p = Path(image_path)
        p = None

        # User requested: "remove the stuffix .xmp and find the first image file that ends in .tif, .jpg or .png"
        candidates = [
            xmp_p.with_suffix(".tif"),
            xmp_p.with_suffix(".jpg"),
            xmp_p.with_suffix(".png"),
            xmp_p.with_suffix(".jpeg"),
            xmp_p.with_suffix(".tiff"),
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                p = candidate
                break

        if not p:
            self._send_json({"error": f"Image view file not found for {xmp_p.name}"}, status=HTTPStatus.NOT_FOUND)
            return

        ext = p.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            ct = "image/jpeg"
        elif ext == ".png":
            ct = "image/png"
        elif ext in (".tif", ".tiff"):
            # Browsers cannot display TIFF files natively. Convert to JPEG in memory for the web UI.
            import io
            from PIL import Image

            try:
                with Image.open(p) as img:
                    # Convert to RGB if necessary (e.g. CMYK or palettes)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    data = buf.getvalue()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                self._send_json({"error": f"Failed to process TIFF: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
        else:
            self._send_json({"error": "Unsupported media type"}, status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return

        try:
            data = p.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self._send_json({"error": "Read error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_update(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        target_path_str = payload.get("path")
        lat = payload.get("lat")
        lon = payload.get("lon")
        idx = payload.get("loc_idx", -1)
        action = payload.get("action", "update")

        if not target_path_str:
            self._send_json({"error": "Missing path"}, status=HTTPStatus.BAD_REQUEST)
            return

        target_path = Path(target_path_str)
        if not target_path.is_file():
            self._send_json({"error": "XMP file not found"}, status=HTTPStatus.NOT_FOUND)
            return

        existing = read_ai_sidecar_state(target_path)
        if not existing:
            self._send_json({"error": "Could not read XMP"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        locations_shown = read_locations_shown(target_path)

        if action == "delete_shown_location":
            if 0 <= idx < len(locations_shown):
                locations_shown.pop(idx)
            city, state, country = "", "", ""
            # Inherit unchanged Exif variables
            exif_lat = str(existing.get("gps_latitude", ""))
            exif_lon = str(existing.get("gps_longitude", ""))
            exif_city = str(existing.get("location_city", ""))
            exif_state = str(existing.get("location_state", ""))
            exif_country = str(existing.get("location_country", ""))
        else:
            if lat is None or lon is None:
                self._send_json({"error": "Missing lat or lon"}, status=HTTPStatus.BAD_REQUEST)
                return

            city, state, country = "", "", ""
            if payload.get("undoing") and "city" in payload:
                city = payload.get("city") or ""
                state = payload.get("state") or ""
                country = payload.get("country") or ""
            else:
                try:
                    geo_res = _GEOCODER.reverse_geocode(lat, lon)
                    if geo_res:
                        city = geo_res.city
                        state = geo_res.state
                        country = geo_res.country
                except Exception as e:
                    self._send_json({"error": f"Nominatim error: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return

            if idx >= 0:
                # Target specific Location Shown
                if idx >= len(locations_shown):
                    locations_shown.append({})
                locations_shown[idx]["city"] = city
                locations_shown[idx]["province_or_state"] = state
                locations_shown[idx]["country_name"] = country
                locations_shown[idx]["name"] = city
                locations_shown[idx]["gps_latitude"] = str(lat)
                locations_shown[idx]["gps_longitude"] = str(lon)

                # Exif remains unmodified
                exif_lat = str(existing.get("gps_latitude", ""))
                exif_lon = str(existing.get("gps_longitude", ""))
                exif_city = str(existing.get("location_city", ""))
                exif_state = str(existing.get("location_state", ""))
                exif_country = str(existing.get("location_country", ""))
            else:
                # Target Base Exif
                exif_lat = str(lat)
                exif_lon = str(lon)
                exif_city = city
                exif_state = state
                exif_country = country

        write_xmp_sidecar(
            target_path,
            person_names=read_person_in_image(target_path),
            subjects=[],
            description=str(existing.get("description", "")),
            ocr_text=str(existing.get("ocr_text", "")),
            gps_latitude=exif_lat,
            gps_longitude=exif_lon,
            location_city=exif_city,
            location_state=exif_state,
            location_country=exif_country,
            locations_shown=locations_shown,
            title=str(existing.get("title", "")),
            album_title=str(existing.get("album_title", "")),
            ocr_lang=str(existing.get("ocr_lang", "")),
            author_text=str(existing.get("author_text", "")),
            scene_text=str(existing.get("scene_text", "")),
            title_source=str(existing.get("title_source", "")),
            ocr_authority_source=str(existing.get("ocr_authority_source", "")),
            create_date=str(existing.get("create_date", "")),
            dc_date=str(existing.get("dc_date", "")),
            date_time_original=str(existing.get("date_time_original", "")),
            stitch_key=str(existing.get("stitch_key", "")),
            detections_payload=existing.get("detections"),
            ocr_ran=bool(existing.get("ocr_ran")),
            people_detected=bool(existing.get("people_detected")),
            people_identified=bool(existing.get("people_identified")),
            source_text=str(existing.get("source_text", "")),
        )

        self._send_json({"ok": True, "city": city, "state": state, "country": country})

    def _handle_geocode_and_update(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        target_path_str = payload.get("path")
        query = payload.get("query")
        idx = payload.get("loc_idx", -1)

        if not target_path_str or not query:
            self._send_json({"error": "Missing path or query"}, status=HTTPStatus.BAD_REQUEST)
            return

        target_path = Path(target_path_str)
        if not target_path.is_file():
            self._send_json({"error": "XMP file not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            geo_res = _GEOCODER.geocode(query)
            if not geo_res:
                self._send_json({"error": f"Address not found for: {query}"}, status=HTTPStatus.BAD_REQUEST)
                return
            lat = geo_res.latitude
            lon = geo_res.longitude
            city = geo_res.city
            state = geo_res.state
            country = geo_res.country
        except Exception as e:
            self._send_json({"error": f"Nominatim error: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        existing = read_ai_sidecar_state(target_path)
        if not existing:
            self._send_json({"error": "Could not read XMP"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        locations_shown = read_locations_shown(target_path)

        if idx >= 0:
            if idx >= len(locations_shown):
                locations_shown.append({})
            locations_shown[idx]["city"] = city
            locations_shown[idx]["province_or_state"] = state
            locations_shown[idx]["country_name"] = country
            locations_shown[idx]["name"] = city
            locations_shown[idx]["gps_latitude"] = str(lat)
            locations_shown[idx]["gps_longitude"] = str(lon)

            exif_lat = str(existing.get("gps_latitude", ""))
            exif_lon = str(existing.get("gps_longitude", ""))
            exif_city = str(existing.get("location_city", ""))
            exif_state = str(existing.get("location_state", ""))
            exif_country = str(existing.get("location_country", ""))
        else:
            exif_lat = str(lat)
            exif_lon = str(lon)
            exif_city = city
            exif_state = state
            exif_country = country

        write_xmp_sidecar(
            target_path,
            person_names=read_person_in_image(target_path),
            subjects=[],
            description=str(existing.get("description", "")),
            ocr_text=str(existing.get("ocr_text", "")),
            gps_latitude=exif_lat,
            gps_longitude=exif_lon,
            location_city=exif_city,
            location_state=exif_state,
            location_country=exif_country,
            locations_shown=locations_shown,
            title=str(existing.get("title", "")),
            album_title=str(existing.get("album_title", "")),
            ocr_lang=str(existing.get("ocr_lang", "")),
            author_text=str(existing.get("author_text", "")),
            scene_text=str(existing.get("scene_text", "")),
            title_source=str(existing.get("title_source", "")),
            ocr_authority_source=str(existing.get("ocr_authority_source", "")),
            create_date=str(existing.get("create_date", "")),
            dc_date=str(existing.get("dc_date", "")),
            date_time_original=str(existing.get("date_time_original", "")),
            stitch_key=str(existing.get("stitch_key", "")),
            detections_payload=existing.get("detections"),
            ocr_ran=bool(existing.get("ocr_ran")),
            people_detected=bool(existing.get("people_detected")),
            people_identified=bool(existing.get("people_identified")),
            source_text=str(existing.get("source_text", "")),
        )

        self._send_json(
            {"ok": True, "lat": lat, "lon": lon, "city": city, "state": state, "country": country, "loc_idx": idx}
        )

    def _send_html(self, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        pass


def run_server(paths: list[str], port: int = 8095):
    import glob

    for p_str in paths:
        p = Path(p_str)
        if p.is_file() and p.suffix.lower() == ".xmp":
            _XMP_PATHS.append(p.resolve())
        elif p.is_dir():
            for f in p.rglob("*.xmp"):
                _XMP_PATHS.append(f.resolve())
        else:
            for f_str in glob.glob(str(p_str)):
                f_path = Path(f_str)
                if f_path.is_file() and f_path.suffix.lower() == ".xmp":
                    _XMP_PATHS.append(f_path.resolve())

    server = ThreadingHTTPServer(("0.0.0.0", port), MapHandler)
    print(f"Map server running at http://localhost:{port}")
    print(f"Loaded {len(_XMP_PATHS)} XMP files.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
