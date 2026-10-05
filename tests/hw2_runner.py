"""Shared isolated runner. Private BRUTE configuration supplies extra fixtures."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from hw2_checks import select_tasks
from hw2_policy import check_source, ForbiddenImport


def _failed_task(task, status, message, elapsed=0.):
    return {**asdict(task), "earned": 0., "status": status, "elapsed": elapsed,
            "cases": [{"name": "Submission checks", "status": status, "message": message}]}


def _preflight(submission):
    errors = {}
    for part, filename in (("1", "hw2_numpy.py"), ("2", "hw2_torch.py")):
        path = submission / filename
        if not path.is_file():
            errors[part] = ("ERROR", f"Missing {filename}. Put both Python files at the top level of hw2_submission.zip.")
        else:
            try:
                if part == "1":
                    check_source(path)
                else:
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except ForbiddenImport as error:
                errors[part] = ("POLICY ERROR", str(error))
            except SyntaxError as error:
                errors[part] = ("ERROR", f"{filename}:{error.lineno}: {error.msg}. Fix the syntax error and upload again.")
            except (OSError, UnicodeError) as error:
                errors[part] = ("ERROR", f"Cannot read {filename}: {error}.")
    return errors


def _worker(task, submission, case_file, scratch, timeout):
    result_file, log_file = scratch / f"{task.key}.json", scratch / f"{task.key}.log"
    command = [sys.executable, "-I", str(Path(__file__).with_name("hw2_worker.py")),
               str(submission), task.key, str(case_file), str(result_file)]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    result = None
    with log_file.open("wb") as log:
        process = subprocess.Popen(command, cwd=scratch, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=(os.name == "posix"))
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            result = _failed_task(task, "TIMEOUT", f"This task exceeded {timeout:g} seconds. "
                                  "Check for an infinite loop or unnecessary repeated work. Other tasks continue.",
                                  time.monotonic() - started)
    if result is None:
        try:
            worker = json.loads(result_file.read_text(encoding="utf-8"))
            if worker["key"] != task.key or worker["status"] not in {
                    "PASSED", "FAILED", "ERROR", "NOT IMPLEMENTED", "POLICY ERROR"}:
                raise ValueError("Invalid task result")
            result = {**asdict(task), **worker, "earned": task.points if worker["status"] == "PASSED" else 0.}
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = _failed_task(task, "ERROR", f"The submitted program stopped before this task finished "
                                  f"(exit code {process.returncode}). Check the captured output.", time.monotonic() - started)
    with log_file.open("rb") as log:
        output = log.read(12001)
    if output:
        result["log"] = output[:12000].decode("utf-8", errors="replace")
        if len(output) > 12000:
            result["log"] += "\n[Output truncated after 12,000 bytes.]"
    return result


def _print_task(result):
    label = f"[{result['tid']}] {result['title']} ".ljust(39, ".")
    print(f"{label} {result['status']}   {result['earned']:g} / {result['points']:g} pt", flush=True)
    failures = [case for case in result["cases"] if case["status"] != "PASSED"]
    messages = list(dict.fromkeys(str(case["message"]) for case in failures))
    for message in messages[:2]:
        print("   " + message.replace("\n", "\n   "), flush=True)
    if len(failures) > 2:
        print(f"   {len(failures)} checks failed; save the JSON report for individual case details.", flush=True)


def _dependency_feedback(tasks):
    activations = ("1.3.sigmoid", "1.3.tanh", "1.3.relu")
    dependencies = {
        "1.6.forward": ("1.4.linear",) + activations,
        "1.6.backward": ("1.4.linear",) + activations,
        "1.7.train": ("1.4.linear", "1.4.bce", "1.6.forward", "1.6.backward", "1.7.sgd") + activations,
        "2.3.bridge": ("2.2.model",),
    }
    results = {task["key"]: task for task in tasks}
    for task in tasks:
        if task["status"] not in {"FAILED", "ERROR", "NOT IMPLEMENTED"}:
            continue
        failed = [results[key] for key in dependencies.get(task["key"], ())
                  if key in results and results[key]["status"] != "PASSED"]
        if failed:
            task["cases"].append({"name": "Components used by this assignment", "status": "BLOCKED",
                                  "message": "This assignment uses components that also failed: " +
                                  ", ".join(f"{dependency['tid']} {dependency['title']}" for dependency in failed) +
                                  ". Fix those components first, then rerun this task."})


def evaluate(submission_dir, *, mode="public", task_selectors=None, output_dir=None,
             timeout=30., cases=None, report_writer=None, progress=True):
    submission = Path(submission_dir).resolve()
    tasks = select_tasks(task_selectors)
    if not tasks:
        raise ValueError("No tasks matched. Use 1, 2, task IDs such as 1.3, or a unit such as 2.1.dataset.")
    if timeout <= 0:
        raise ValueError("The per-task timeout must be positive.")
    started = time.monotonic()
    report = {"title": "DPL HW2 evaluation", "mode": mode, "tasks": [], "earned": 0.,
              "possible": sum(task.points for task in tasks), "elapsed": 0.,
              "environment": {"python": sys.version.split()[0]}, "notices": []}
    output = Path(output_dir).resolve() if output_dir is not None else None
    def save():
        report["elapsed"] = time.monotonic() - started
        if output is not None and report_writer is not None:
            report_writer(output, report)
    report["tasks"] = [_failed_task(task, "BLOCKED", "This task has not run yet.") for task in tasks]
    save()  # A usable zero-score report exists even if evaluation is interrupted.
    errors = _preflight(submission)
    if cases is None:
        cases = json.loads(Path(__file__).with_name("hw2_cases.json").read_text(encoding="utf-8"))
    policy_detected = errors.get("1", (None,))[0] == "POLICY ERROR"
    with tempfile.TemporaryDirectory(prefix="hw2-evaluation-") as temporary:
        scratch = Path(temporary)
        case_file = scratch / "cases.json"
        case_file.write_text(json.dumps(cases), encoding="utf-8")
        for index, task in enumerate(tasks):
            if task.key[0] in errors:
                result = _failed_task(task, *errors[task.key[0]])
            elif task.key.startswith("1.") and policy_detected:
                result = _failed_task(task, "POLICY ERROR", "A forbidden import was attempted in hw2_numpy.py. Part 1 receives 0 points.")
            else:
                result = _worker(task, submission, case_file, scratch, timeout)
            report["environment"].update(result.pop("environment", {}))
            report["tasks"][index] = result
            if task.key.startswith("1.") and result["status"] == "POLICY ERROR":
                policy_detected = True
                # The policy applies to the whole file, including earlier successful units.
                for previous in report["tasks"]:
                    if previous["key"].startswith("1.") and previous["status"] != "POLICY ERROR":
                        previous.update(status="POLICY ERROR", earned=0.)
                        previous["cases"].append({"name": "NumPy-only policy", "status": "POLICY ERROR",
                                                   "message": "A forbidden import was attempted elsewhere in hw2_numpy.py. Part 1 receives 0 points."})
            report["earned"] = sum(unit["earned"] for unit in report["tasks"])
            save()
            if progress:
                _print_task(result)
    _dependency_feedback(report["tasks"])
    if policy_detected:
        report["notices"].append("Part 1 uses a forbidden library. All NumPy points are zero; Part 2 is graded independently.")
    report["notices"].append("Points are awarded per scoring unit: all its required checks must pass.")
    report["notices"].append("Local and upload evaluation use the same checks. Upload evaluation adds seeded inputs, not new requirements.")
    save()
    if progress:
        print(f"\n{'Local tests' if mode == 'public' else 'Upload evaluation'}: "
              f"{report['earned']:g} / {report['possible']:g} points", flush=True)
        if policy_detected:
            print("Part 1 total is zero because a forbidden import was detected.", flush=True)
    return report
