from vhs_pipeline.cli import build_parser


def test_cli_has_subtitles_command_with_filters() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "subtitles",
            "--archive",
            "callahan",
            "--archive",
            "demo",
            "--title",
            "Chapter",
            "--title-exact",
        ]
    )
    assert args.group == "subtitles"
    assert args.archive == ["callahan", "demo"]
    assert args.title == ["Chapter"]
    assert args.title_exact is True
