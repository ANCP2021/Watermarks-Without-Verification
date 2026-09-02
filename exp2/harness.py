from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .settings import EXEC_FSIZE_MB, EXEC_MEM_MB, EXEC_TIMEOUT_S

RUNNER = Path(__file__).with_name("_runner.py")
STDERR_TAIL = 600

STATUS_BY_EXIT = {0: "passed", 1: "failed", 2: "syntax_error", 3: "harness_error"}
LIMITS_MARKER = "EXP2_LIMITS "
HARNESS_ERROR = "harness_error"


def harness_description() -> dict:
    return {
        "isolation": "fresh subprocess per candidate, own session/process group, "
                     "temp cwd, scrubbed env (PYTHONHASHSEED=0), stdin closed",
        "limits": "RLIMIT_AS + RLIMIT_DATA = mem_mb, RLIMIT_FSIZE = fsize_mb, "
                  "RLIMIT_CORE = 0 (best effort: the record says what was enforced)",
        "network": "socket module disabled in-process; --exec-prefix for a kernel "
                   "network namespace",
        "timeout": "SIGKILL to the process group, status 'timeout'",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "runner": str(RUNNER),
    }


def _kill_group(proc: subprocess.Popen) -> None:

    if proc.poll() is not None:
        return
    for fn in (lambda: os.killpg(proc.pid, signal.SIGKILL), proc.kill):
        try:
            fn()
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _parse_limits(stderr: str) -> dict | None:
    """The runner's limits line, or None when the runner never got that far."""
    for line in stderr.splitlines():
        if line.startswith(LIMITS_MARKER):
            return dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)
    return None


def preflight(timeout_s: float = EXEC_TIMEOUT_S, mem_mb: int = EXEC_MEM_MB,
              fsize_mb: int = EXEC_FSIZE_MB, exec_prefix: str | None = None) -> dict:
    """A trivial program must pass, or every candidate would be misjudged."""
    result = run_program("x = 1\nassert x == 1\n", timeout_s, mem_mb, fsize_mb, exec_prefix)
    if result["status"] != "passed":
        raise SystemExit(
            f"execution harness preflight failed: a trivial program came back "
            f"{result['status']!r} (exit {result['exit_code']}): {result['stderr_tail'][-300:]!r}\n"
            f"Check --exec-prefix ({exec_prefix!r}) and the interpreter {sys.executable}.")
    return result


def run_program(program: str, timeout_s: float = EXEC_TIMEOUT_S,
                mem_mb: int = EXEC_MEM_MB, fsize_mb: int = EXEC_FSIZE_MB,
                exec_prefix: str | None = None) -> dict:
    tmp = tempfile.mkdtemp(prefix="exp2-exec-")
    started = time.time()
    try:
        candidate = os.path.join(tmp, "candidate.py")
        with open(candidate, "w", encoding="utf-8") as f:
            f.write(program)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp,
            "TMPDIR": tmp,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "OMP_NUM_THREADS": "1",
        }
        cmd = (shlex.split(exec_prefix) if exec_prefix else []) + [
            sys.executable, "-s", "-B", str(RUNNER), candidate, str(int(mem_mb)),
            str(int(fsize_mb))]
        proc = subprocess.Popen(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                start_new_session=True)
        timed_out = False
        try:
            _, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            try:
                _, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                err = b""
        finally:
            _kill_group(proc)
        stderr = (err or b"").decode("utf-8", errors="replace")
        code = proc.returncode
        limits = _parse_limits(stderr)
        if timed_out:
            status = "timeout"
        elif limits is None:
            status = HARNESS_ERROR 
        elif code in STATUS_BY_EXIT:
            status = STATUS_BY_EXIT[code]
        elif code is not None and code < 0:
            status = "killed" 
        else:
            status = "error"
        tail = "\n".join(l for l in stderr.splitlines() if not l.startswith(LIMITS_MARKER))
        return {
            "status": status,
            "exit_code": code,
            "elapsed_s": round(time.time() - started, 3),
            "stderr_tail": tail[-STDERR_TAIL:],
            "limits": limits or {},
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
