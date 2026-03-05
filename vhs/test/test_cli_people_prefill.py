from vhs_pipeline.cli import build_parser


def test_cli_has_people_prefill_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "people",
            "prefill",
            "--archive",
            "callahan_01_archive",
            "--chapter",
            "Example Chapter",
            "--apply",
        ]
    )
    assert args.group == "people"
    assert args.people_kind == "prefill"
    assert args.archive == "callahan_01_archive"
    assert args.chapter == "Example Chapter"
    assert args.apply is True
