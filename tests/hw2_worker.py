"""Run one scoring unit in a fresh process; used locally and by BRUTE."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# -I removes the script directory from sys.path. Only trust our own checks first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hw2_checks import run_task, Suite
from hw2_policy import check_source, ImportGuard, ForbiddenImport


def main():
    submission, key, case_file, result_file = sys.argv[1:]
    started = time.monotonic()
    suite, guard = Suite(), None
    environment = {"python": sys.version.split()[0]}
    try:
        # Bound verbose student output on Unix; the parent also enforces wall time.
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 ** 2, 16 * 1024 ** 2))
        except (ImportError, ValueError, OSError):
            pass
        import numpy as np
        environment["numpy"] = np.__version__
        filename = "hw2_numpy.py" if key.startswith("1.") else "hw2_torch.py"
        path = Path(submission) / filename
        if key.startswith("1."):
            check_source(path)
            guard = ImportGuard().install()
        else:
            import torch
            torch.set_num_threads(1)
            torch.manual_seed(0)
            environment["torch"] = torch.__version__
        cases = json.loads(Path(case_file).read_text(encoding="utf-8"))
        sys.path.insert(1, str(Path(submission)))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        suite = run_task(key, module, cases)
    except BaseException as error:
        # Imports, SystemExit and setup failures must still produce a result.
        def fail():
            raise error
        suite.run("Load submission and prepare checks", fail)
    if guard is not None and guard.attempts:
        suite.cases.append({"name": "NumPy-only policy", "status": "POLICY ERROR",
                            "message": "Forbidden import attempted: " + ", ".join(sorted(set(guard.attempts))) +
                            ". Imports remain forbidden even if their exceptions are caught. Part 1 receives 0 points."})
    result = {"key": key, "status": suite.status(), "cases": suite.cases,
              "elapsed": time.monotonic() - started, "environment": environment}
    Path(result_file).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
