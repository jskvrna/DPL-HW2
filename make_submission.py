"""
Creates hw2_submission.zip for upload to BRUTE.

    python make_submission.py               # check, run the local tests, zip
    python make_submission.py --skip-tests  # only check and zip

The zip contains exactly the files that are graded (at the top level of the
archive). The script also warns about common problems: syntax errors, TODO
blocks that are still not implemented, and forbidden imports in Part 1.
"""

import argparse
import py_compile
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))

from hw2_policy import check_source, ForbiddenImport  # noqa: E402
FILES = ["hw2_numpy.py", "hw2_torch.py"]
ZIP_NAME = "hw2_submission.zip"
TODO_LEFT = re.compile(r'raise NotImplementedError\("(TODO[^"]*)"\)')


def main():
    parser = argparse.ArgumentParser(description="Zip the HW2 files for upload.")
    parser.add_argument("--skip-tests", action="store_true", help="do not run the local tests")
    args = parser.parse_args()

    problems = []
    for name in FILES:
        path = HERE / name
        if not path.exists():
            print(f"ERROR: {name} not found next to make_submission.py")
            return 1
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            print(f"ERROR: {name} has a syntax error - it would get 0 points:\n{e.msg}")
            return 1
        source = path.read_text(encoding="utf-8")
        todos = TODO_LEFT.findall(source)
        if todos:
            problems.append(f"{name}: not implemented yet: {', '.join(todos)}")
        if name == "hw2_numpy.py":
            try:
                check_source(path)
            except ForbiddenImport as error:
                problems.append(str(error))

    if not args.skip_tests:
        print("Running the local tests (python test_hw2.py) ...\n")
        result = subprocess.run([sys.executable, "test_hw2.py"], cwd=HERE,
                                capture_output=True, text=True)
        lines = (result.stdout + result.stderr).strip().splitlines()
        not_passed = [l for l in lines if l.startswith("[") and "PASSED" not in l]
        print("\n".join(not_passed + [l for l in lines if l.startswith("Local tests:")]) or "\n".join(lines[-5:]))
        print()

    zip_path = HERE / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.write(HERE / name, arcname=name)

    for p in problems:
        print(f"WARNING: {p}")
    print(f"\nCreated {zip_path.name} with: {', '.join(FILES)}")
    print("Upload this file to BRUTE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
