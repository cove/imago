import argparse
import os
import sys
from pathlib import Path

from getmyancestors.getmyancestors import main as getmyancestors_main

DEFAULT_PERSON_ID = "P631-4WH"
DEFAULT_USERNAME = "CoveSchneider"
DEFAULT_OUTPUT = Path("familysearch/data/cove-schneider-familysearch-ancestors.ged")
DEFAULT_LOG = Path("familysearch/data/cove-schneider-familysearch-ancestors.log")


def main():
    parser = argparse.ArgumentParser(description="Download the Cove Schneider FamilySearch ancestor tree.")
    parser.add_argument("--person-id", default=os.environ.get("FAMILYSEARCH_PERSON_ID", DEFAULT_PERSON_ID))
    parser.add_argument("--username", default=os.environ.get("FAMILYSEARCH_USERNAME", DEFAULT_USERNAME))
    parser.add_argument(
        "--generations", type=int, default=int(os.environ.get("FAMILYSEARCH_ASCEND_GENERATIONS", "200"))
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("FAMILYSEARCH_TIMEOUT", "120")))
    parser.add_argument("--rate-limit", type=int, default=int(os.environ.get("FAMILYSEARCH_RATE_LIMIT", "2")))
    args = parser.parse_args()

    password = os.environ.get("FAMILYSEARCH_PASSWORD")
    if not password:
        sys.exit("Missing FAMILYSEARCH_PASSWORD")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    original_argv = sys.argv
    sys.argv = [
        "getmyancestors",
        "--username",
        args.username,
        "--password",
        password,
        "--individuals",
        args.person_id,
        "--ascend",
        str(args.generations),
        "--marriage",
        "--outfile",
        str(args.output),
        "--logfile",
        str(args.log),
        "--timeout",
        str(args.timeout),
        "--rate-limit",
        str(args.rate_limit),
        "--verbose",
    ]
    try:
        getmyancestors_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
