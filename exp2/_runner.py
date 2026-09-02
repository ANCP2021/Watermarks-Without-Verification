import os
import sys

_exit = os._exit 


def apply_limits(mem_mb, fsize_mb):
    import resource

    enforced = {}
    targets = {
        "RLIMIT_AS": mem_mb * 1024 * 1024,
        "RLIMIT_DATA": mem_mb * 1024 * 1024,
        "RLIMIT_FSIZE": fsize_mb * 1024 * 1024,
        "RLIMIT_CORE": 0,
    }
    for name, value in targets.items():
        lim = getattr(resource, name, None)
        if lim is None:
            enforced[name] = "unavailable"
            continue
        try:
            _, hard = resource.getrlimit(lim)
            if hard != resource.RLIM_INFINITY and value > hard:
                value = hard 
            resource.setrlimit(lim, (value, value))
            enforced[name] = "set"
        except (ValueError, OSError):
            enforced[name] = "refused"
    return enforced


def guard():
    import builtins
    import faulthandler
    import shutil
    import socket
    import subprocess

    faulthandler.disable()

    def blocked(*args, **kwargs):
        raise OSError("disabled inside the execution sandbox")
    
    class BlockedSocket:
        def __init__(self, *args, **kwargs):
            raise OSError("network access is disabled inside the execution sandbox")

    socket.socket = BlockedSocket
    socket.SocketType = BlockedSocket
    for name in ("create_connection", "create_server", "socketpair", "fromfd",
                 "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
                 "getnameinfo"):
        if hasattr(socket, name):
            setattr(socket, name, blocked)
    try:
        import _socket
        _socket.socket = BlockedSocket
    except ImportError:
        pass

    builtins.exit = None
    builtins.quit = None
    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
                 "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
                 "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
                 "lchflags", "lchmod", "lchown", "chdir", "_exit", "execv", "execve",
                 "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnl",
                 "spawnle", "spawnv", "spawnve", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter", "multiprocessing",
                "ctypes", "_ctypes", "signal"):
        sys.modules[mod] = None


def main():
    if len(sys.argv) != 4:
        print("usage: _runner.py <candidate.py> <mem_mb> <fsize_mb>", file=sys.stderr)
        _exit(3)
    candidate, mem_mb, fsize_mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]

    try:
        with open(candidate, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as exc:
        print(f"harness: cannot read candidate: {exc}", file=sys.stderr)
        _exit(3)

    enforced = apply_limits(mem_mb, fsize_mb)
    print("EXP2_LIMITS " + " ".join(f"{k}={v}" for k, v in enforced.items()),
          file=sys.stderr, flush=True)

    try:
        code = compile(source, "candidate.py", "exec")
    except (SyntaxError, ValueError) as exc: 
        print(f"syntax error: {exc}", file=sys.stderr)
        sys.stderr.flush()
        _exit(2)

    guard()

    namespace = {"__name__": "__main__", "__builtins__": __builtins__}
    try:
        exec(code, namespace)
    except BaseException as exc: 
        import traceback
        tb = traceback.format_exception_only(type(exc), exc)
        print("".join(tb)[-2000:], file=sys.stderr)
        try:
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass
        _exit(1)
    try:
        sys.stderr.flush()
        sys.stdout.flush()
    except Exception:
        pass
    _exit(0)


if __name__ == "__main__":
    main()
