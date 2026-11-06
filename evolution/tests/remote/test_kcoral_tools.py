"""Core contracts for the remote tool clients and their shared transport."""

import fcntl
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolution.remote import kcoral_compute_sanitizer as sanitizer
from evolution.remote import kcoral_iket as iket
from evolution.remote import kcoral_ncu as ncu
from evolution.remote import kcoral_python as python
from evolution.remote import kcoral_tool as tool
from evolution.remote import kcoral_tool_worker as worker


@pytest.fixture(autouse=True)
def no_gpu_lock(monkeypatch):
    """The worker's default lock path exists on the Thor; tests must not queue on the real lock."""
    monkeypatch.setenv("GROOT_GPU_LOCK", "missing-lock")


@pytest.fixture
def fake_client(monkeypatch):
    def install(result, *, timeout=300):
        class Client:
            def __init__(self, url):
                assert url == "http://server"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, program, **kwargs):
                assert kwargs == {"timeout_seconds": timeout, "output_limit_bytes": 16 * 1024**2}
                return result

        monkeypatch.setitem(sys.modules, "kcoral", SimpleNamespace(Client=Client))

    return install


def test_tool_arguments_and_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_VALUE", "forwarded")
    common = [
        "--remote",
        "http://server",
        "-e",
        "LOCAL_VALUE",
        "-e",
        "VALUE=first",
        "-e",
        "VALUE=a=b; $HOME",
        "-e",
        "EMPTY=",
    ]
    for cli, options, forwarded in [
        (sanitizer, [], ["--tool", "memcheck", "python", "script.py", "-e", "app-value"]),
        (
            iket,
            ["--output-dir", "local"],
            ["profile", "--postprocess", "json", "--", "python", "script.py", "-o", "app-file"],
        ),
        (python, [], ["-u", "-c", "print('; $HOME')", "-e", "app-value"]),
    ]:
        args, actual = cli.parse_args([*common, *options, "--", *forwarded])
        assert actual == forwarded
        assert args.env == {"LOCAL_VALUE": "forwarded", "VALUE": "a=b; $HOME", "EMPTY": ""}
    monkeypatch.delenv("UNSET_TEST", raising=False)
    for value in [
        "UNSET_TEST",
        "=x",
        "BAD-NAME=x",
        "X=\x00",
        "CUDA_VISIBLE_DEVICES=0",
        "KCORAL_DIR=x",
    ]:
        with pytest.raises(SystemExit):
            python.parse_args(["--remote", "url", "-e", value, "--", "script.py"])


def test_invalid_tool_contracts():
    for cli, arguments in [
        (sanitizer, ["--fetch", "log", "--", "python", "s.py"]),
        (iket, ["--output-dir", "local", "--", "postprocess", "old"]),
        (iket, ["--output-dir", "local", "--", "-oremote", "profile", "--", "python", "s.py"]),
        (python, ["--", "-ui", "s.py"]),
        (python, ["--", "-"]),
        (python, ["--timeout", "0", "--", "s.py"]),
    ]:
        with pytest.raises(SystemExit):
            cli.parse_args(["--remote", "url", *arguments])


def test_archive_layout_and_validation(tmp_path):
    (tmp_path / "pkg").mkdir()
    script = tmp_path / "pkg/run.py"
    script.write_text("pass")
    script.chmod(0o755)
    archive = tool.pack_inputs([tmp_path])
    assert archive == tool.pack_inputs([tmp_path])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert tar.getnames() == ["pkg/run.py"]
        assert tar.getmember("pkg/run.py").mode == 0o700
    with pytest.raises(ValueError, match="conflicting"):
        tool.pack_inputs([tmp_path, tmp_path])
    (tmp_path / "link").symlink_to(script)
    with pytest.raises(ValueError, match="symlink"):
        tool.pack_inputs([tmp_path])
    for name in ["../escape", "/absolute", "a\\b", "."]:
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            tar.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
        with pytest.raises(ValueError):
            worker.unpack_inputs(data.getvalue(), tmp_path)
        with pytest.raises(ValueError):
            tool.relative_path(name)
    with pytest.raises(ValueError, match="symlink"):
        worker.collect_files(tmp_path, ["link"])


@pytest.mark.parametrize("mode", ["script", "-c", "-m"])
def test_python_subprocess_inputs_environment_and_status(tmp_path, capfd, monkeypatch, mode):
    source = (
        "import os,sys\nfrom pathlib import Path\n"
        "assert Path('input').read_text() == 'payload'\n"
        "assert os.environ['VALUE'] == 'literal; $HOME'\n"
        "assert os.environ['EMPTY'] == ''\n"
        "assert sys.stdin.read() == ''\n"
        "print(os.environ['KCORAL_DIR'], flush=True)\n"
        "print(sys.argv[1], file=sys.stderr)\n"
        "raise SystemExit(7)\n"
    )
    (tmp_path / "capture.py").write_text(source)
    (tmp_path / "input").write_text("payload")
    monkeypatch.setenv("VALUE", "server")
    arguments = {"script": ["capture.py"], "-c": ["-c", source], "-m": ["-m", "capture"]}[mode]
    result = worker.python(
        tool.pack_inputs([tmp_path]),
        [*arguments, "argument"],
        {"VALUE": "literal; $HOME", "EMPTY": ""},
    )
    assert result == {"returncode": 7, "files": {}, "missing": []}
    output = capfd.readouterr()
    assert output.err == "argument\n"
    assert not Path(output.out.strip()).exists()
    assert worker.os.environ["VALUE"] == "server"


@pytest.mark.parametrize("failure", [False, True])
def test_iket_managed_outputs_and_cleanup(tmp_path, monkeypatch, failure):
    workdirs = []
    monkeypatch.delenv("TVM_IKET_OFFICIAL_PROFILE", raising=False)
    monkeypatch.setattr(worker.shutil, "which", lambda _: "/opt/run-iket")

    def run(argv, *, cwd, env, stdin):
        workdirs.append(cwd)
        assert env["MODE"] == "test"
        assert env["KCORAL_DIR"] == str(cwd)
        assert argv[:2] == ["/opt/run-iket", "--output-dir"]
        reports = Path(argv[2])
        assert not reports.exists() and not reports.is_relative_to(cwd)
        assert argv[3:] == ["profile", "--", sys.executable, "capture.py"]
        assert env["TVM_IKET_OFFICIAL_PROFILE"] == "cutlass-4.6.0"
        if failure:
            raise OSError("launch failed")
        reports.mkdir()
        (reports / "trace.json").write_bytes(b"trace")
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(worker.subprocess, "run", run)
    if failure:
        with pytest.raises(OSError, match="launch failed"):
            worker.iket(
                tool.pack_inputs(), ["profile", "--", "python", "capture.py"], {"MODE": "test"}
            )
    else:
        assert worker.iket(
            tool.pack_inputs(), ["profile", "--", "python", "capture.py"], {"MODE": "test"}
        ) == {"returncode": 9, "files": {"trace.json": b"trace"}, "missing": []}
    assert workdirs and not workdirs[0].parent.exists()


def test_sanitizer_artifacts_on_nonzero_exit(tmp_path, monkeypatch):
    def executable(name):
        assert name == "compute-sanitizer"
        return sys.executable

    monkeypatch.setattr(worker.shutil, "which", executable)
    result = worker.compute_sanitizer(
        tool.pack_inputs(),
        ["-c", "from pathlib import Path; Path('log').write_bytes(b'error'); raise SystemExit(9)"],
        ["log", "missing"],
        {},
    )
    assert result == {"returncode": 9, "files": {"log": b"error"}, "missing": ["missing"]}
    tool.save_artifacts(tmp_path, result["files"], overwrite=False)
    assert tool.artifact_exit_code(result) == 9
    assert (tmp_path / "log").read_bytes() == b"error"
    with pytest.raises(FileExistsError):
        tool.save_file(tmp_path / "log", b"replacement")
    tool.save_file(tmp_path / "log", b"replacement", True)
    assert (tmp_path / "log").read_bytes() == b"replacement"
    assert list(tmp_path.iterdir()) == [tmp_path / "log"]
    assert tool.artifact_exit_code({"returncode": 0, "files": {}, "missing": ["report"]}) == 1


def test_failed_request_preserves_streams(fake_client, capsys):
    fake_client(
        SimpleNamespace(
            stdout="stdout",
            stderr="stderr",
            completed=False,
            stdout_truncated=True,
            stderr_truncated=False,
            status="FAILED",
            error={"message": "missing tool", "traceback": "remote stack"},
        ),
        timeout=42,
    )
    with pytest.raises(RuntimeError, match="missing tool"):
        tool.execute(SimpleNamespace(remote="http://server", timeout=42), None)
    output = capsys.readouterr()
    assert output.out == "stdout"
    assert "stderr" in output.err and "truncated" in output.err


def test_python_option_clusters_and_target_arguments():
    for arguments in [
        ["-uc", "print(1)", "-i"],
        ["-P", "-c", "print(1)"],
        ["-uWignore", "-X", "dev", "-m", "package", "-i"],
        ["--check-hash-based-pycs", "always", "script.py", "-i"],
    ]:
        _, forwarded = python.parse_args(["--remote", "url", "--", *arguments])
        assert forwarded == arguments


def test_invalid_output_prevents_remote_work(tmp_path, monkeypatch, capsys):
    output = tmp_path / "file"
    output.write_text("existing")

    def unexpected_request(*args, **kwargs):
        pytest.fail("invalid output must be rejected before a remote request")

    cases = [(ncu, ["-o", str(tmp_path)], ["python", "capture.py"])]
    for destination in [output, output / "child"]:
        cases.extend(
            [
                (
                    iket,
                    ["--output-dir", str(destination)],
                    ["profile", "--", "python", "capture.py"],
                ),
                (
                    sanitizer,
                    ["--output-dir", str(destination), "--fetch", "log"],
                    ["python", "check.py"],
                ),
            ]
        )
    for cli, options, target in cases:
        monkeypatch.setattr(cli, "execute", unexpected_request)
        assert cli.main(["--remote", "url", *options, "--", *target]) == 1
    assert "not a directory" in capsys.readouterr().err
    assert output.read_text() == "existing"


def test_ncu_arguments_and_environment():
    args, options, application = ncu.parse_args(
        [
            "--remote",
            "http://server",
            "-o",
            "local.ncu-rep",
            "-e",
            "MODE=test",
            "--set",
            "full",
            "--kernel-name",
            "regex:a b",
            "--",
            "python",
            "-c",
            "print('a; b')",
            "-o",
            "app-output",
        ]
    )
    assert args.env == {"MODE": "test"}
    assert args.export == Path("local.ncu-rep")
    assert options == ["--set", "full", "--kernel-name", "regex:a b"]
    assert application == ["python", "-c", "print('a; b')", "-o", "app-output"]


@pytest.mark.parametrize(
    "suffix", [[], ["--"], ["--import", "old", "--", "python"], ["--mode=attach", "--", "python"]]
)
def test_ncu_invalid_capture_contract(suffix):
    with pytest.raises(SystemExit):
        ncu.parse_args(["--remote", "http://server", "-o", "report", *suffix])


@pytest.mark.parametrize("failure", [False, True])
def test_ncu_worker_argv_inputs_and_cleanup(tmp_path, monkeypatch, failure):
    source = tmp_path / "capture.py"
    source.write_text("application source")
    source.chmod(0o755)
    workdirs = []
    monkeypatch.setenv("PROFILE_MODE", "server-value")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "worker-gpu")
    monkeypatch.setattr(worker.shutil, "which", lambda _: "/opt/ncu")

    def execute(argv, *, cwd, env, stdin):
        workdirs.append(cwd)
        assert (cwd / "capture.py").read_text() == "application source"
        assert (cwd / "capture.py").stat().st_mode & 0o111
        assert stdin == worker.subprocess.DEVNULL
        assert not Path(argv[4]).is_relative_to(cwd)
        assert env["KCORAL_DIR"] == str(cwd)
        assert env["PROFILE_MODE"] == "override"
        assert env["CUDA_VISIBLE_DEVICES"] == "worker-gpu"
        assert argv[:4] == ["/opt/ncu", "--config-file", "0", "--export"]
        assert argv[5:] == ["--set", "basic", sys.executable, "capture.py", "literal;value"]
        if failure:
            raise OSError("launch failed")
        Path(argv[4]).write_bytes(b"report\x00\xff")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(worker.subprocess, "run", execute)
    if failure:
        with pytest.raises(OSError):
            worker.ncu(
                tool.pack_inputs([source]),
                ["--set", "basic"],
                ["python", "capture.py", "literal;value"],
                {"PROFILE_MODE": "override"},
            )
    else:
        assert worker.ncu(
            tool.pack_inputs([source]),
            ["--set", "basic"],
            ["python", "capture.py", "literal;value"],
            {"PROFILE_MODE": "override"},
        ) == {"returncode": 7, "report": b"report\x00\xff"}
    assert workdirs and not workdirs[0].parent.exists()
    assert worker.os.environ["PROFILE_MODE"] == "server-value"


@pytest.mark.parametrize(
    "code,report,expected", [(0, b"bytes", 0), (9, b"partial", 9), (0, None, 1), (-15, None, 143)]
)
def test_ncu_report_delivery(tmp_path, monkeypatch, fake_client, capsys, code, report, expected):
    output = tmp_path / "report.ncu-rep"
    argv = ["--remote", "http://server", "-o", str(output), "--", "python", "capture.py"]
    result = SimpleNamespace(
        stdout="app stdout\n",
        stderr="app stderr\n",
        completed=True,
        stdout_truncated=False,
        stderr_truncated=False,
        results={"outcome": {"returncode": code, "report": report}},
    )

    fake_client(result)
    monkeypatch.setattr(ncu, "build_program", lambda *args, **kwargs: "program")
    assert ncu.main(argv) == expected
    captured = capsys.readouterr()
    assert captured.out == "app stdout\n"
    assert "app stderr\n" in captured.err
    assert output.exists() == (report is not None)
    if report is not None:
        assert output.read_bytes() == report


@pytest.mark.parametrize("command", ["python", "python3", "/custom/bin/python", "./python", "app"])
def test_profiler_python_uses_worker_unless_path_is_explicit(command):
    args = [command, "-u", "capture.py", "python", "literal;value"]
    resolved = worker._worker_python(args)
    assert resolved == [sys.executable if command in ("python", "python3") else command, *args[1:]]
    assert args[0] == command


@pytest.mark.parametrize(
    "inherited,override,expected",
    [
        (None, None, "cutlass-4.6.0"),
        ("inherited", None, "inherited"),
        ("inherited", "explicit", "explicit"),
        (None, "explicit", "explicit"),
        ("", None, ""),
        ("inherited", "", ""),
        (None, "", ""),
    ],
)
def test_iket_profile_defaults_preserve_explicit_configuration(
    monkeypatch, inherited, override, expected
):
    key = "TVM_IKET_OFFICIAL_PROFILE"
    if inherited is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, inherited)
    monkeypatch.setattr(worker.shutil, "which", lambda _: "/opt/run-iket")

    def run(argv, *, cwd, env, stdin):
        assert env[key] == expected
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", run)
    worker.iket(
        tool.pack_inputs(),
        ["profile", "--", "python3", "capture.py"],
        {} if override is None else {key: override},
    )
    assert worker.os.environ.get(key) == inherited


def _locked(path):
    with open(path) as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


@pytest.mark.parametrize("name", ["python", "sanitizer", "iket", "ncu"])
def test_tools_hold_the_gpu_lock_around_the_subprocess(tmp_path, monkeypatch, name):
    lock = tmp_path / "lock"
    lock.touch()
    overrides = {"GROOT_GPU_LOCK": str(lock)}
    held = []

    def run(argv, *, cwd, env, stdin):
        held.append(_locked(lock))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.shutil, "which", lambda _: "/opt/tool")
    monkeypatch.setattr(worker.subprocess, "run", run)
    archive = tool.pack_inputs()
    if name == "python":
        worker.python(archive, ["-c", "pass"], overrides)
    elif name == "sanitizer":
        worker.compute_sanitizer(archive, ["python"], [], overrides)
    elif name == "iket":
        worker.iket(archive, ["profile", "--", "python", "capture.py"], overrides)
    else:
        worker.ncu(archive, [], ["python", "capture.py"], overrides)
    assert held == [True]
    assert not _locked(lock)


@pytest.mark.parametrize("nested", [False, True])
def test_script_taking_the_gpu_lock_itself_gets_it(tmp_path, nested):
    lock = tmp_path / "lock"
    lock.touch()
    # The alarm turns a self-deadlock into a failed assertion instead of a hung test.
    source = (
        f"import fcntl, signal; signal.alarm(20); f = open({str(lock)!r}); "
        "fcntl.flock(f, fcntl.LOCK_EX)"
    )
    if nested:
        source = (
            f"import subprocess, sys; "
            f"sys.exit(subprocess.run([sys.executable, '-c', {source!r}]).returncode)"
        )
    result = worker.python(tool.pack_inputs(), ["-c", source], {"GROOT_GPU_LOCK": str(lock)})
    assert result["returncode"] == 0
    assert not _locked(lock)


def test_only_our_descendants_get_the_gpu_lock_handed_over(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.touch()
    waiter = f"import fcntl; fcntl.flock(open({str(lock)!r}), fcntl.LOCK_EX)"
    inode = lock.stat().st_ino
    with open(lock) as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        process = subprocess.Popen([sys.executable, "-c", waiter])
        try:
            deadline = time.monotonic() + 20
            while not worker._queued_below_this_process(inode):
                assert time.monotonic() < deadline
                time.sleep(0.05)
            monkeypatch.setattr(worker.os, "getpid", lambda: -1)
            assert not worker._queued_below_this_process(inode)
        finally:
            process.kill()
            process.wait()


def test_gpu_lock_wait_is_bounded(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.touch()
    ran = []
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: ran.append(a))
    overrides = {"GROOT_GPU_LOCK": str(lock), "GROOT_GPU_LOCK_WAIT_S": "0.2"}
    with open(lock) as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="retry later"):
            worker.python(tool.pack_inputs(), ["-c", "pass"], overrides)
    assert not ran
    deadline = time.monotonic() + 20  # the abandoned helper thread takes and drops the lock
    while _locked(lock):
        assert time.monotonic() < deadline
        time.sleep(0.05)
