"""Generate a self-contained HTML tool for manually placing photo regions on a page.

For pages whose crops cannot be located automatically (e.g. the page was re-rendered
after cropping), this produces a standalone .html file per page with the page image and
each crop's thumbnail embedded as data URIs. You draw a box around each photo; the tool
emits JSON (page-pixel coords) to paste back for writing the RegionList.

Usage:
    uv run python -m photoalbums.scripts.make_region_annotator --album England_1983_B01 --page 51
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from photoalbums.common import PHOTO_ALBUMS_DIR
from photoalbums.scripts.restore_orphan_regions import _base_crops_for_page, _has_region_list


def _data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _dims(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    h, w = img.shape[:2]
    return w, h


_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Place regions: {page}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}}
 #left{{flex:1;overflow:auto;background:#222;padding:10px}}
 #right{{width:280px;overflow:auto;border-left:1px solid #ccc;padding:10px;box-sizing:border-box}}
 #wrap{{position:relative;display:inline-block}}
 canvas{{cursor:crosshair;display:block}}
 .crop{{border:2px solid transparent;border-radius:6px;padding:6px;margin-bottom:8px;cursor:pointer}}
 .crop.active{{border-color:#e22;background:#fee}}
 .crop.done{{opacity:.55}}
 .crop img{{width:100%;display:block;border:1px solid #999}}
 .crop b{{display:block;margin-bottom:4px}}
 button{{font-size:14px;padding:6px 10px;margin:4px 0}}
 textarea{{width:100%;height:140px;font-family:monospace;font-size:11px}}
 #status{{font-size:12px;color:#555}}
</style></head><body>
<div id="left"><div id="wrap"><canvas id="cv"></canvas></div></div>
<div id="right">
 <p id="status"></p>
 <p>Click a photo below, then drag a box around it on the page. Redraw to fix.</p>
 <div id="crops"></div>
 <button onclick="exportJSON()">Export JSON</button>
 <button onclick="copyJSON()">Copy</button>
 <textarea id="out" readonly></textarea>
</div>
<script>
const PAGE="{page}", PW={pw}, PH={ph};
const CROPS={crops_json};
const pageImg=new Image(); pageImg.src="{page_uri}";
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
let scale=1, boxes={{}}, active=null, drag=null;
pageImg.onload=()=>{{
  const maxW=Math.min(1100,window.innerWidth-340);
  scale=Math.min(1,maxW/PW);
  cv.width=Math.round(PW*scale); cv.height=Math.round(PH*scale);
  buildCrops(); redraw();
}};
function buildCrops(){{
  const c=document.getElementById("crops");
  CROPS.forEach(cr=>{{
    const d=document.createElement("div"); d.className="crop"; d.id="crop_"+cr.label;
    d.innerHTML="<b>"+cr.label+" ("+cr.w+"x"+cr.h+")</b><img src='"+cr.uri+"'>";
    d.onclick=()=>{{active=cr.label; updateCrops();}};
    c.appendChild(d);
  }});
  active=CROPS[0].label; updateCrops();
}}
function updateCrops(){{
  CROPS.forEach(cr=>{{const e=document.getElementById("crop_"+cr.label);
    e.classList.toggle("active",cr.label===active);
    e.classList.toggle("done",!!boxes[cr.label]);}});
  const n=Object.keys(boxes).length;
  document.getElementById("status").textContent=n+" / "+CROPS.length+" placed. Active: "+active;
}}
function redraw(){{
  ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(pageImg,0,0,cv.width,cv.height);
  ctx.lineWidth=2; ctx.font="16px sans-serif";
  for(const lab in boxes){{const b=boxes[lab];
    ctx.strokeStyle="#1e90ff"; ctx.fillStyle="#1e90ff";
    ctx.strokeRect(b.x*scale,b.y*scale,b.w*scale,b.h*scale);
    ctx.fillText(lab,b.x*scale+4,b.y*scale+18);}}
  if(drag){{ctx.strokeStyle="#e22";
    ctx.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0);}}
}}
function pos(e){{const r=cv.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}}
cv.onmousedown=e=>{{const[x,y]=pos(e); drag={{x0:x,y0:y,x1:x,y1:y}};}};
cv.onmousemove=e=>{{if(!drag)return; const[x,y]=pos(e); drag.x1=x; drag.y1=y; redraw();}};
cv.onmouseup=e=>{{if(!drag)return; const x=Math.min(drag.x0,drag.x1),y=Math.min(drag.y0,drag.y1),
  w=Math.abs(drag.x1-drag.x0),h=Math.abs(drag.y1-drag.y0); drag=null;
  if(w>5&&h>5&&active){{boxes[active]={{x:Math.round(x/scale),y:Math.round(y/scale),
    w:Math.round(w/scale),h:Math.round(h/scale)}};
    const i=CROPS.findIndex(c=>c.label===active); if(i>=0&&i+1<CROPS.length)active=CROPS[i+1].label;}}
  updateCrops(); redraw();}};
function exportJSON(){{
  const regions=CROPS.filter(c=>boxes[c.label]).map(c=>({{label:c.label,...boxes[c.label]}}));
  const payload={{page:PAGE,page_width:PW,page_height:PH,regions}};
  document.getElementById("out").value=JSON.stringify(payload,null,2);
}}
function copyJSON(){{exportJSON();const t=document.getElementById("out");t.select();
  document.execCommand("copy");}}
</script></body></html>"""


def _make_for_page(view_jpg: Path, out_dir: Path) -> Path:
    photos_dir = view_jpg.parent.parent / (view_jpg.parent.name[: -len("_Pages")] + "_Photos")
    page_stem = view_jpg.stem[:-2] if view_jpg.stem.endswith("_V") else view_jpg.stem
    pw, ph = _dims(view_jpg)
    base_crops = _base_crops_for_page(photos_dir, page_stem)
    crops = []
    for base, crop_path in base_crops.items():
        cw, ch = _dims(crop_path)
        crops.append({"label": f"D{base:02d}", "w": cw, "h": ch, "uri": _data_uri(crop_path)})
    html = _HTML.format(
        page=page_stem,
        pw=pw,
        ph=ph,
        page_uri=_data_uri(view_jpg),
        crops_json=json.dumps(crops),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{page_stem}.place-regions.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-root", default=str(PHOTO_ALBUMS_DIR))
    parser.add_argument("--album", default="", help="Substring filter on album/_Pages dir name.")
    parser.add_argument("--page", default="", help="Page filter, e.g. 'P51' or '51'.")
    parser.add_argument("--out-dir", default="", help="Output dir (default: <photos-root>/.region_restore_manual).")
    args = parser.parse_args(argv)

    root = Path(args.photos_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / ".region_restore_manual"
    album = args.album.casefold()
    page = args.page.casefold().lstrip("p")

    made = 0
    for view_jpg in sorted(root.rglob("*_V.jpg")):
        if not view_jpg.parent.name.endswith("_Pages"):
            continue
        if album and album not in view_jpg.parent.name.casefold():
            continue
        if page and f"_p{page}_v" not in view_jpg.name.casefold():
            continue
        if _has_region_list(view_jpg.with_suffix(".xmp")):
            continue
        out = _make_for_page(view_jpg, out_dir)
        print(f"made {out}")
        made += 1
    print(f"\n{made} annotator file(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
