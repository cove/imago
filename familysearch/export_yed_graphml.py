import argparse
import html
import sys
from pathlib import Path

from render_relationship_graph import (
    DEFAULT_GEDCOM,
    DEFAULT_ROOT,
    graph_id,
    parse_gedcom,
    person_label,
    select_graph,
    sort_key,
    visible_family_groups,
)

DEFAULT_OUTPUT = Path("familysearch/data/cove-schneider-yed-familytree-4gen.graphml")


def xml(value):
    return html.escape(str(value), quote=True)


def node_graphics(node_id, label, x, y, width, height, fill, border, shape):
    return f"""    <node id="{xml(node_id)}">
      <data key="d0">
        <y:ShapeNode>
          <y:Geometry x="{x}" y="{y}" width="{width}" height="{height}"/>
          <y:Fill color="{fill}" transparent="false"/>
          <y:BorderStyle color="{border}" type="line" width="1.0"/>
          <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Helvetica" fontSize="10" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" textColor="#111827" visible="true">{xml(label)}</y:NodeLabel>
          <y:Shape type="{shape}"/>
        </y:ShapeNode>
      </data>
    </node>"""


def edge_graphics(edge_id, source, target):
    return f"""    <edge id="{xml(edge_id)}" source="{xml(source)}" target="{xml(target)}">
      <data key="d1">
        <y:PolyLineEdge>
          <y:LineStyle color="#64748b" type="line" width="1.0"/>
          <y:Arrows source="none" target="standard"/>
          <y:BendStyle smoothed="false"/>
        </y:PolyLineEdge>
      </data>
    </edge>"""


def write_graphml(path, people, families, selected_people, selected_families, root_id):
    family_groups = visible_family_groups(families, selected_families, selected_people)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:y="http://www.yworks.com/xml/graphml"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">',
        '  <key id="d0" for="node" yfiles.type="nodegraphics"/>',
        '  <key id="d1" for="edge" yfiles.type="edgegraphics"/>',
        '  <graph id="G" edgedefault="directed">',
    ]

    for index, person_id in enumerate(sorted(selected_people, key=sort_key)):
        fill = "#e0f2fe" if person_id == root_id else "#f8fafc"
        border = "#0284c7" if person_id == root_id else "#64748b"
        lines.append(
            node_graphics(
                "p_" + graph_id(person_id),
                person_label(people, person_id),
                (index % 8) * 170,
                (index // 8) * 110,
                135,
                48,
                fill,
                border,
                "roundrectangle",
            )
        )

    for index, family_group in enumerate(family_groups):
        lines.append(
            node_graphics(
                "f_" + graph_id(family_group["ids"][0]),
                "",
                (index % 8) * 170 + 70,
                (index // 8) * 110 + 70,
                8,
                8,
                "#475569",
                "#475569",
                "ellipse",
            )
        )

    edge_index = 0
    for family_group in family_groups:
        family_node = "f_" + graph_id(family_group["ids"][0])
        for parent in family_group["partners"]:
            lines.append(edge_graphics(f"e{edge_index}", "p_" + graph_id(parent), family_node))
            edge_index += 1
        for child in sorted(family_group["children"], key=sort_key):
            lines.append(edge_graphics(f"e{edge_index}", family_node, "p_" + graph_id(child)))
            edge_index += 1

    lines.extend(["  </graph>", "</graphml>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(family_groups)


def main():
    parser = argparse.ArgumentParser(description="Export a yEd GraphML family tree experiment.")
    parser.add_argument("--gedcom", type=Path, default=DEFAULT_GEDCOM)
    parser.add_argument("--root-id", default=DEFAULT_ROOT)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root_id = args.root_id if args.root_id.startswith("@") else f"@{args.root_id}@"
    people, families = parse_gedcom(args.gedcom)
    selected_people, selected_families = select_graph(people, families, root_id, args.generations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    family_count = write_graphml(
        args.output,
        people,
        families,
        selected_people,
        selected_families,
        root_id,
    )
    sys.stdout.write(
        f"wrote {args.output} with {len(selected_people)} people and {family_count} family nodes\n"
    )


if __name__ == "__main__":
    main()
