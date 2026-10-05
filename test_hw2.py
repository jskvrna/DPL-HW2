"""Local checks for DPL HW2, shared with the upload evaluator.

    python test_hw2.py                 # all scoring units
    python test_hw2.py 1               # NumPy only
    python test_hw2.py 1.3 2.3          # selected assignments
    python test_hw2.py --dir DIR       # an extracted submission
    python test_hw2.py --json report.json
    python test_hw2.py 2.4 --verbose    # every failed check, including independent training diagnostics

BRUTE adds seeded inputs to the same checks; it does not add requirements.
"""

import argparse
import json
from pathlib import Path
import sys

# The shared test helpers and the public fixtures live in tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from hw2_runner import evaluate  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", help="Task IDs, parts, or scoring units")
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing hw2_numpy.py and hw2_torch.py")
    parser.add_argument("--json", type=Path, help="Save all check details as JSON")
    parser.add_argument("--timeout", type=float, default=30., help="Seconds allowed per scoring unit")
    parser.add_argument("--verbose", action="store_true", help="Show every failed check and its diagnostic")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.dir, mode="public", task_selectors=args.tasks, timeout=args.timeout)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.verbose:
        for task in report["tasks"]:
            for case in task["cases"]:
                if case["status"] != "PASSED":
                    print(f"\n[{task['tid']}] {case['name']} — {case['status']}")
                    print(case["message"])
                    if case.get("detail"):
                        print(case["detail"])
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Detailed results: {args.json}")
    print("Upload evaluation uses the same checks with additional seeded inputs.")
    return 0 if report["earned"] == report["possible"] else 1


if __name__ == "__main__":
    sys.exit(main())
