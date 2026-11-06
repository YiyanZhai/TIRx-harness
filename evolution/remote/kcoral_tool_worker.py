"""Remote-only tool runners, uploaded as source; this is not a local CLI.

A separate file keeps the uploaded code easy to lint and test.
"""

import contextlib
import fcntl
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

GPU_LOCK = "/sources/.thor-gpu-timing.lock"
GPU_LOCK_WAIT_S = "240"  # inside the tools' default 300 s request timeout


def _below_this_process(pid):
    try:
        while pid > 1:
            with open(f"/proc/{pid}/stat") as stat:
                pid = int(stat.read().rsplit(")", 1)[1].split()[1])
            if pid == os.getpid():
                return True
    except OSError:  # it exited
        pass
    return False


def _queued_below_this_process(inode):
    """Whether one of this process's descendants waits for the lock on file ``inode``."""
    with open("/proc/locks") as locks:
        for line in locks:
            fields = line.split()
            if fields[1] == "->" and fields[6].endswith(f":{inode}"):
                if _below_this_process(int(fields[5])):
                    return True
    return False


@contextlib.contextmanager
def gpu_lock(env):
    """Hold the machine's shared GPU timing lock, if it exists, around one tool subprocess.

    Waits at most GROOT_GPU_LOCK_WAIT_S. A blocking flock in a helper thread queues like the other
    jobs' waiters. Scripts and benchmark suites take this lock themselves, some with a hard-coded
    path, and would wait on us forever: once a descendant queues on it, it gets the lock instead.
    """

    path = env.get("GROOT_GPU_LOCK", GPU_LOCK)
    if not os.path.exists(path):
        yield
        return
    limit = float(env.get("GROOT_GPU_LOCK_WAIT_S", GPU_LOCK_WAIT_S))
    file = open(path)
    acquired, abandoned, mutex = threading.Event(), threading.Event(), threading.Lock()

    def take():
        fcntl.flock(file, fcntl.LOCK_EX)
        with mutex:
            if abandoned.is_set():
                file.close()
            else:
                acquired.set()

    start = time.monotonic()
    threading.Thread(target=take, daemon=True).start()
    if not acquired.wait(limit):
        with mutex:
            if not acquired.is_set():
                abandoned.set()
                raise RuntimeError(
                    f"GPU timing lock {path} busy for {limit:.0f} s (another job is timing); "
                    "retry later, or raise --timeout and -e GROOT_GPU_LOCK_WAIT_S together"
                )
    print(
        f"kcoral: GPU lock {path} held (waited {time.monotonic() - start:.1f} s)",
        file=sys.stderr,
        flush=True,
    )
    finished = threading.Event()

    def hand_over():
        inode = os.fstat(file.fileno()).st_ino
        while not finished.wait(0.5):
            if _queued_below_this_process(inode):
                fcntl.flock(file, fcntl.LOCK_UN)
                return

    watcher = threading.Thread(target=hand_over, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join()
        file.close()


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"expected a relative file path: {name!r}")
    return path


def unpack_inputs(archive, workdir):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            path = relative_path(member.name)
            if not member.isfile():
                raise ValueError(f"input is not a regular file: {member.name!r}")
            destination = workdir.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            destination.chmod(0o700 if member.mode & 0o111 else 0o600)


def environment(workdir, overrides):
    return {**os.environ, **overrides, "KCORAL_DIR": str(workdir)}


def _worker_python(application):
    """Keep bare Python commands in the benchmark worker's environment."""
    if application and application[0] in ("python", "python3"):
        return [sys.executable, *application[1:]]
    return application


def collect_files(workdir, paths):
    files, missing = {}, []
    for name in paths:
        path = workdir.joinpath(*relative_path(name).parts)
        if path.is_symlink() or not path.resolve().is_relative_to(workdir.resolve()):
            raise ValueError(f"artifact escapes input directory or is a symlink: {path}")
        entries = sorted(path.rglob("*")) if path.is_dir() else [path]
        found = False
        for entry in entries:
            if entry.is_symlink() or not entry.resolve().is_relative_to(workdir.resolve()):
                raise ValueError(f"artifact escapes input directory or is a symlink: {entry}")
            if entry.is_dir():
                continue
            if not entry.is_file():
                continue
            files[entry.relative_to(workdir).as_posix()] = entry.read_bytes()
            found = True
        if not found:
            missing.append(name)
    return files, missing


def _resolve_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} is not installed in the remote server environment")
    return os.path.abspath(executable)


def _run_command(archive, command, fetch, overrides):
    executable = _resolve_executable(command[0])
    with tempfile.TemporaryDirectory(prefix="kcoral_tool_") as root:
        workdir = Path(root)
        unpack_inputs(archive, workdir)
        env = environment(workdir, overrides)
        with gpu_lock(env):
            completed = subprocess.run(
                [executable, *command[1:]],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        files, missing = collect_files(workdir, fetch)
        return {"returncode": completed.returncode, "files": files, "missing": missing}


def python(archive, arguments, overrides):
    return _run_command(archive, [sys.executable, *arguments], [], overrides)


def compute_sanitizer(archive, arguments, fetch, overrides):
    return _run_command(archive, ["compute-sanitizer", *arguments], fetch, overrides)


def iket(archive, arguments, overrides):
    executable = _resolve_executable("run-iket")
    boundary = arguments.index("--")
    arguments = [*arguments[: boundary + 1], *_worker_python(arguments[boundary + 1 :])]
    with tempfile.TemporaryDirectory(prefix="kcoral_iket_") as root:
        workdir, reports = Path(root) / "inputs", Path(root) / "reports"
        workdir.mkdir()
        unpack_inputs(archive, workdir)
        child_env = environment(workdir, overrides)
        child_env.setdefault("TVM_IKET_OFFICIAL_PROFILE", "cutlass-4.6.0")
        with gpu_lock(child_env):
            completed = subprocess.run(
                [executable, "--output-dir", str(reports), *arguments],
                cwd=workdir,
                env=child_env,
                stdin=subprocess.DEVNULL,
            )
        paths = [p.name for p in sorted(reports.iterdir())] if reports.is_dir() else []
        files, _ = collect_files(reports, paths)
        return {
            "returncode": completed.returncode,
            "files": files,
            "missing": [] if files else ["IKET output"],
        }


def ncu(archive, ncu_args, application, overrides):
    executable = _resolve_executable("ncu")
    application = _worker_python(application)
    with tempfile.TemporaryDirectory(prefix="kcoral_ncu_") as root:
        workdir = Path(root) / "inputs"
        workdir.mkdir()
        unpack_inputs(archive, workdir)
        report = Path(root) / "capture.ncu-rep"
        env = environment(workdir, overrides)
        export = ["--config-file", "0", "--export", str(report)]
        with gpu_lock(env):
            completed = subprocess.run(
                [executable, *export, *ncu_args, *application],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        return {
            "returncode": completed.returncode,
            "report": report.read_bytes() if report.is_file() else None,
        }
