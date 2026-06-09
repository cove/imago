import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_GEDCOM = Path("familysearch/data/cove-schneider-familysearch-ancestors.ged")
DEFAULT_OUTPUT = Path("familysearch/data/cove-schneider-ancestors-6gen.svg")
DEFAULT_DOT = Path("familysearch/data/cove-schneider-ancestors-6gen.dot")
DEFAULT_ROOT = "@I1@"
DEFAULT_DOT_BIN = Path("/opt/homebrew/Cellar/graphviz/15.0.0/bin/dot")
DEFAULT_CAST_ROOT = Path("cast/data")


def apply_individual_line(row, level, tag, value, last_event):
    if level == "1" and tag == "NAME":
        row["names"].append(value.replace("/", "").strip())
        return None
    if level == "1" and tag in {"BIRT", "DEAT"}:
        return tag
    if level == "2" and tag == "DATE" and last_event == "BIRT":
        row["birth"] = value.strip()
        return last_event
    if level == "2" and tag == "DATE" and last_event == "DEAT":
        row["death"] = value.strip()
        return last_event
    if level == "1" and tag == "FAMC":
        row["famc"].append(value.strip())
        return None
    if level == "1":
        return None
    return last_event


def apply_family_line(row, level, tag, value):
    if level == "1" and tag == "HUSB":
        row["husb"] = value.strip()
    elif level == "1" and tag == "WIFE":
        row["wife"] = value.strip()
    elif level == "1" and tag == "CHIL":
        row["chil"].append(value.strip())


def parse_gedcom(path):
    people = {}
    families = {}
    current = None
    kind = None
    last_event = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        match = re.match(r"0 (@[^@]+@) (INDI|FAM)", line)
        if match:
            current, kind = match.group(1), match.group(2)
            if kind == "INDI":
                people[current] = {"names": [], "birth": "", "death": "", "famc": []}
            else:
                families[current] = {"husb": "", "wife": "", "chil": []}
            last_event = None
            continue

        if current is None:
            continue

        parts = line.split(" ", 2)
        if len(parts) < 2:
            continue
        level, tag = parts[0], parts[1]
        value = parts[2] if len(parts) > 2 else ""

        if kind == "INDI":
            last_event = apply_individual_line(people[current], level, tag, value, last_event)
        elif kind == "FAM":
            apply_family_line(families[current], level, tag, value)

    return people, families


def ancestor_edges(people, families, root_id, generations):
    selected = {root_id}
    edges = []
    frontier = {root_id}

    for _ in range(generations):
        next_frontier = set()
        for child in sorted(frontier):
            for family_id in people.get(child, {}).get("famc", []):
                family = families.get(family_id, {})
                for parent in (family.get("husb"), family.get("wife")):
                    if parent and parent in people:
                        selected.add(parent)
                        edges.append((parent, child))
                        next_frontier.add(parent)
        frontier = next_frontier
        if not frontier:
            break

    return selected, sorted(set(edges))


def person_sort_key(person_id):
    return int(re.sub(r"\D", "", person_id) or 0)


def person_label(people, person_id):
    row = people[person_id]
    name = row["names"][0] if row["names"] else person_id.strip("@")
    dates = []
    if row["birth"]:
        dates.append(f"b. {row['birth']}")
    if row["death"]:
        dates.append(f"d. {row['death']}")
    return "\n".join([name, *dates])


def normalized_tokens(value):
    text = str(value or "").casefold()
    text = text.replace("née", "nee")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [token for token in text.split() if token and token != "nee"]


def load_face_crops(cast_root):
    people_path = cast_root / "people.json"
    faces_path = cast_root / "faces.jsonl"
    if not people_path.is_file() or not faces_path.is_file():
        return {}, {}

    people_payload = json.loads(people_path.read_text(encoding="utf-8"))
    cast_people = {
        str(row.get("person_id") or ""): str(row.get("display_name") or "").strip()
        for row in list(people_payload.get("people") or [])
        if str(row.get("person_id") or "").strip()
    }
    best_faces = {}
    for raw in faces_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        person_id = str(row.get("person_id") or "").strip()
        if not person_id or person_id not in cast_people:
            continue
        if str(row.get("review_status") or "").strip().casefold() in {"ignored", "rejected"}:
            continue
        crop_path = str(row.get("crop_path") or "").strip()
        if not crop_path:
            continue
        crop = cast_root / crop_path
        if not crop.is_file():
            continue
        quality = float(row.get("quality") or 0.0)
        current = best_faces.get(person_id)
        if current is None or quality > current["quality"]:
            best_faces[person_id] = {"crop": crop.resolve(), "quality": quality}
    return cast_people, best_faces


def match_face_crops(people, selected, cast_people, best_faces):
    cast_rows = []
    for person_id, display_name in cast_people.items():
        if person_id in best_faces:
            cast_rows.append((person_id, display_name, set(normalized_tokens(display_name))))

    crops = {}
    for gedcom_id in selected:
        name = people[gedcom_id]["names"][0] if people[gedcom_id]["names"] else ""
        tokens = normalized_tokens(name)
        if len(tokens) < 2:
            continue
        first = tokens[0]
        surname = tokens[-1]
        matches = [
            person_id
            for person_id, _display_name, cast_tokens in cast_rows
            if first in cast_tokens and surname in cast_tokens
        ]
        if len(matches) == 1:
            crops[gedcom_id] = best_faces[matches[0]]["crop"]
    return crops


def html_label(people, person_id, crop_path):
    lines = [html.escape(part) for part in person_label(people, person_id).splitlines()]
    text_rows = "".join(f'<TR><TD><FONT POINT-SIZE="10">{line}</FONT></TD></TR>' for line in lines)
    image_row = ""
    if crop_path:
        image_row = (
            f'<TR><TD><IMG SRC="{html.escape(str(crop_path))}" '
            'SCALE="TRUE" FIXEDSIZE="TRUE" WIDTH="72" HEIGHT="72"/></TD></TR>'
        )
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">{image_row}{text_rows}</TABLE>>'


def write_dot(path, people, selected, edges, root_id, face_crops):
    lines = [
        "digraph ancestors {",
        '  graph [rankdir=TB, bgcolor="white", pad=0.35, nodesep=0.35, ranksep=0.65, splines=ortho];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica", fontsize=10, margin=0.08];',
        '  edge [color="#94a3b8", arrowsize=0.6];',
    ]
    for person_id in sorted(selected, key=person_sort_key):
        attrs = ""
        if person_id == root_id:
            attrs = ', fillcolor="#e0f2fe", color="#0284c7", penwidth=2'
        crop_path = face_crops.get(person_id)
        if crop_path:
            lines.append(f"  {person_id.strip('@')} [label={html_label(people, person_id, crop_path)}{attrs}];")
        else:
            lines.append(f"  {person_id.strip('@')} [label={json.dumps(person_label(people, person_id))}{attrs}];")
    for parent, child in edges:
        lines.append(f"  {parent.strip('@')} -> {child.strip('@')};")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Render a bounded FamilySearch GEDCOM ancestor tree to SVG.")
    parser.add_argument("--gedcom", type=Path, default=DEFAULT_GEDCOM)
    parser.add_argument("--root-id", default=DEFAULT_ROOT)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--dot", type=Path, default=DEFAULT_DOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dot-bin", type=Path, default=DEFAULT_DOT_BIN)
    parser.add_argument("--with-face-crops", action="store_true")
    parser.add_argument("--cast-root", type=Path, default=DEFAULT_CAST_ROOT)
    args = parser.parse_args()

    root_id = args.root_id if args.root_id.startswith("@") else f"@{args.root_id}@"
    people, families = parse_gedcom(args.gedcom)
    selected, edges = ancestor_edges(people, families, root_id, args.generations)
    face_crops = {}
    if args.with_face_crops:
        cast_people, best_faces = load_face_crops(args.cast_root)
        face_crops = match_face_crops(people, selected, cast_people, best_faces)

    args.dot.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_dot(args.dot, people, selected, edges, root_id, face_crops)
    subprocess.run([str(args.dot_bin), "-Tsvg", str(args.dot), "-o", str(args.output)], check=True)
    sys.stdout.write(
        f"wrote {args.output} with {len(selected)} people, "
        f"{len(edges)} parent-child edges, and {len(face_crops)} face crops\n"
    )


if __name__ == "__main__":
    main()
