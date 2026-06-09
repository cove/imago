import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_GEDCOM = Path("familysearch/data/cove-schneider-familysearch-ancestors.ged")
DEFAULT_OUTPUT = Path("familysearch/data/cove-schneider-relationship-graph-4gen.svg")
DEFAULT_DOT = Path("familysearch/data/cove-schneider-relationship-graph-4gen.dot")
DEFAULT_ROOT = "@I1@"
DEFAULT_DOT_BIN = Path("/opt/homebrew/Cellar/graphviz/15.0.0/bin/dot")


def new_person():
    return {"name": "", "birth": "", "death": "", "famc": [], "fams": []}


def new_family():
    return {"husb": "", "wife": "", "chil": []}


def parse_gedcom(path):
    people = {}
    families = {}
    current = None
    kind = None
    last_event = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"0 (@[^@]+@) (INDI|FAM)", raw)
        if match:
            current, kind = match.group(1), match.group(2)
            if kind == "INDI":
                people[current] = new_person()
            else:
                families[current] = new_family()
            last_event = None
            continue

        if current is None:
            continue
        parts = raw.split(" ", 2)
        if len(parts) < 2:
            continue
        level, tag = parts[0], parts[1]
        value = parts[2] if len(parts) > 2 else ""
        if kind == "INDI":
            last_event = apply_person_line(people[current], level, tag, value, last_event)
        elif kind == "FAM":
            apply_family_line(families[current], level, tag, value)

    return people, families


def apply_person_line(row, level, tag, value, last_event):
    if level == "1" and tag == "NAME" and not row["name"]:
        row["name"] = value.replace("/", "").strip()
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
    if level == "1" and tag == "FAMS":
        row["fams"].append(value.strip())
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


def select_ancestor_graph(people, families, root_id, generations):
    selected_people = {root_id}
    selected_families = set()
    frontier = {root_id}

    for _ in range(generations):
        next_frontier = set()
        for child in sorted(frontier):
            for family_id in people.get(child, {}).get("famc", []):
                family = families.get(family_id)
                if not family:
                    continue
                selected_families.add(family_id)
                for parent in (family.get("husb"), family.get("wife")):
                    if parent and parent in people:
                        selected_people.add(parent)
                        next_frontier.add(parent)
        frontier = next_frontier
        if not frontier:
            break

    return selected_people, selected_families


def add_spouse_family_context(people, families, selected_people, selected_families):
    for person_id in set(selected_people):
        for family_id in people[person_id]["fams"]:
            family = families.get(family_id)
            if not family:
                continue
            selected_families.add(family_id)
            for member in [family.get("husb"), family.get("wife"), *family.get("chil", [])]:
                if member and member in people:
                    selected_people.add(member)

    return selected_people, selected_families


def select_graph(people, families, root_id, generations):
    selected_people, selected_families = select_ancestor_graph(people, families, root_id, generations)
    return add_spouse_family_context(people, families, selected_people, selected_families)


def graph_id(ref):
    return re.sub(r"[^A-Za-z0-9_]", "_", ref.strip("@"))


def sort_key(ref):
    return int(re.sub(r"\D", "", ref) or 0)


def person_label(people, person_id):
    row = people[person_id]
    parts = [row["name"] or person_id.strip("@")]
    if row["birth"]:
        parts.append(f"b. {row['birth']}")
    if row["death"]:
        parts.append(f"d. {row['death']}")
    return "\n".join(parts)


def visible_family_groups(families, selected_families, selected_people):
    groups = {}
    for family_id in sorted(selected_families, key=sort_key):
        family = families[family_id]
        partners = tuple(
            ref
            for ref in (family.get("husb"), family.get("wife"))
            if ref and ref in selected_people
        )
        children = tuple(ref for ref in family.get("chil", []) if ref in selected_people)
        if not partners and not children:
            continue
        key = tuple(sorted(partners)) if partners else ("children", *sorted(children))
        group = groups.setdefault(key, {"ids": [], "partners": partners, "children": set()})
        group["ids"].append(family_id)
        group["children"].update(children)
    return sorted(groups.values(), key=lambda group: sort_key(group["ids"][0]))


def person_attrs(person_id, root_id):
    if person_id == root_id:
        return 'shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#0284c7", penwidth=2'
    return 'shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b"'


def write_dot(path, people, families, selected_people, selected_families, root_id):
    family_groups = visible_family_groups(families, selected_families, selected_people)
    lines = [
        "digraph family_relationship_graph {",
        '  graph [rankdir=TB, bgcolor="white", pad=0.35, nodesep=0.45, ranksep=0.75, splines=ortho];',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8, color="#475569", arrowsize=0.55];',
        "  subgraph cluster_legend {",
        '    label="Legend"; color="#cbd5e1"; style=rounded; fontsize=11;',
        '    legend_person [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", label="Person"];',
        '    legend_spouse [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", label="Spouse / partner"];',
        '    legend_child [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", label="Child"];',
        '    legend_junction [shape=point, width=0.08, color="#475569", label=""];',
        "    legend_person -> legend_spouse [dir=none, style=dashed, constraint=false];",
        "    legend_person -> legend_junction [dir=none];",
        "    legend_junction -> legend_child;",
        "  }",
    ]
    lines.extend(
        (
            f"  {graph_id(person_id)} [{person_attrs(person_id, root_id)}, "
            f"label={json.dumps(person_label(people, person_id))}];"
        )
        for person_id in sorted(selected_people, key=sort_key)
    )
    for family_group in family_groups:
        if family_group["children"]:
            family_node = "family_" + graph_id(family_group["ids"][0])
            lines.append(
                f'  {family_node} [shape=point, width=0.08, color="#475569", '
                f'label="", tooltip="{", ".join(family_group["ids"])}"];'
            )
            if family_group["partners"]:
                same_rank_nodes = [
                    *(graph_id(partner) for partner in family_group["partners"]),
                    family_node,
                ]
                lines.append("  { rank=same; " + "; ".join(same_rank_nodes) + "; }")
            lines.extend(
                f"  {graph_id(parent)} -> {family_node} [dir=none, weight=20];"
                for parent in family_group["partners"]
            )
            lines.extend(
                f"  {family_node} -> {graph_id(child)} [weight=12];"
                for child in sorted(family_group["children"], key=sort_key)
            )
        elif len(family_group["partners"]) == 2:
            lines.append(
                "  { rank=same; "
                + "; ".join(graph_id(partner) for partner in family_group["partners"])
                + "; }"
            )
            lines.append(
                f"  {graph_id(family_group['partners'][0])} -> "
                f"{graph_id(family_group['partners'][1])} "
                f'[dir=none, style=dashed, weight=20, tooltip="{", ".join(family_group["ids"])}"];'
            )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Render a FamilySearch GEDCOM relationship graph.")
    parser.add_argument("--gedcom", type=Path, default=DEFAULT_GEDCOM)
    parser.add_argument("--root-id", default=DEFAULT_ROOT)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--dot", type=Path, default=DEFAULT_DOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dot-bin", type=Path, default=DEFAULT_DOT_BIN)
    args = parser.parse_args()

    root_id = args.root_id if args.root_id.startswith("@") else f"@{args.root_id}@"
    people, families = parse_gedcom(args.gedcom)
    selected_people, selected_families = select_graph(people, families, root_id, args.generations)
    args.dot.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_dot(args.dot, people, families, selected_people, selected_families, root_id)
    subprocess.run([str(args.dot_bin), "-Tsvg", str(args.dot), "-o", str(args.output)], check=True)
    rendered_family_count = len(visible_family_groups(families, selected_families, selected_people))
    sys.stdout.write(
        f"wrote {args.output} with {len(selected_people)} people "
        f"and {rendered_family_count} family groups\n"
    )


if __name__ == "__main__":
    main()
