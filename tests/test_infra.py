import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InfraTests(unittest.TestCase):
    def test_config_is_valid_json(self):
        with (ROOT / "config.json").open(encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.assertEqual(config["entrypoint"].split()[0:3], ["python", "-m", "uvicorn"])
        self.assertIs(config["only_for_instance_admins"], True)

    def test_create_venv_script_is_valid_and_uses_tracked_requirements(self):
        script_path = ROOT / "create_venv.sh"
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((ROOT / "dev_requirements.txt").is_file())
        self.assertIn('requirements_file="dev_requirements.txt"', script_path.read_text())
        self.assertIn("sys.version_info < (3, 11)", script_path.read_text())
        self.assertIn("flock -n 9", script_path.read_text())
        self.assertNotIn('rm -f -- "$lock_file"', script_path.read_text())

    def test_create_venv_rejects_concurrent_update_and_keeps_stable_lockfile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script_path = temp_root / "create_venv.sh"
            shutil.copy(ROOT / "create_venv.sh", script_path)
            (temp_root / "dev_requirements.txt").write_text("", encoding="utf-8")

            fake_bin = temp_root / "fake-bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3.11"
            fake_python.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    if [ "$1" = "-c" ]; then
                        exit 0
                    fi
                    if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
                        mkdir -p "$3/bin"
                        cp "$0" "$3/bin/python"
                        exit 0
                    fi
                    if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
                        : > "$FAKE_PIP_STARTED"
                        while [ ! -e "$FAKE_PIP_RELEASE" ]; do sleep 0.01; done
                    fi
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            started = temp_root / "pip-started"
            release = temp_root / "pip-release"
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            env["FAKE_PIP_STARTED"] = str(started)
            env["FAKE_PIP_RELEASE"] = str(release)
            first = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=temp_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(started.exists(), "first venv update did not reach pip")

                second = subprocess.run(
                    ["bash", str(script_path)],
                    cwd=temp_root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertNotEqual(second.returncode, 0)
                self.assertIn("already running", second.stderr)
            finally:
                release.touch()
                _, first_stderr = first.communicate(timeout=5)

            self.assertEqual(first.returncode, 0, first_stderr)
            self.assertTrue((temp_root / ".venv.lock").is_file())

    def test_create_venv_restores_backup_if_interrupted_after_move(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script_path = temp_root / "create_venv.sh"
            shutil.copy(ROOT / "create_venv.sh", script_path)
            (temp_root / "dev_requirements.txt").write_text("", encoding="utf-8")
            old_venv = temp_root / ".venv"
            old_venv.mkdir()
            sentinel = old_venv / "original-environment"
            sentinel.write_text("keep", encoding="utf-8")

            fake_bin = temp_root / "fake-bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3.11"
            fake_python.write_text(
                '#!/bin/bash\n[ "$1" = "-c" ] && exit 0\nexit 0\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_mv = fake_bin / "mv"
            fake_mv.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    /bin/mv "$@"
                    if [ ! -e "$FAKE_MV_INTERRUPTED" ]; then
                        : > "$FAKE_MV_INTERRUPTED"
                        kill -TERM "$PPID"
                    fi
                    """
                ),
                encoding="utf-8",
            )
            fake_mv.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            env["FAKE_MV_INTERRUPTED"] = str(temp_root / "mv-interrupted")
            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=temp_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(sentinel.is_file())
            self.assertEqual(list(temp_root.glob(".venv.backup.*")), [])

    def test_create_venv_replaces_existing_environment_only_after_install(self):
        for install_fails in (True, False):
            with self.subTest(install_fails=install_fails), tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                script_path = temp_root / "create_venv.sh"
                shutil.copy(ROOT / "create_venv.sh", script_path)
                (temp_root / "dev_requirements.txt").write_text("", encoding="utf-8")

                old_venv = temp_root / ".venv"
                old_venv.mkdir()
                sentinel = old_venv / "original-environment"
                sentinel.write_text("keep", encoding="utf-8")

                fake_bin = temp_root / "fake-bin"
                fake_bin.mkdir()
                fake_python = fake_bin / "python3.11"
                fake_python.write_text(
                    textwrap.dedent(
                        """\
                        #!/bin/bash
                        if [ "$1" = "-c" ]; then
                            exit 0
                        fi
                        if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
                            mkdir -p "$3/bin"
                            cp "$0" "$3/bin/python"
                            printf '%s' "$3" > "$3/created-at"
                            exit 0
                        fi
                        if [ "${FAKE_PIP_FAIL:-0}" = "1" ] && [[ "$*" == *" -r "* ]]; then
                            exit 9
                        fi
                        exit 0
                        """
                    ),
                    encoding="utf-8",
                )
                fake_python.chmod(0o755)

                env = os.environ.copy()
                env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
                env["FAKE_PIP_FAIL"] = "1" if install_fails else "0"
                result = subprocess.run(
                    ["bash", str(script_path)],
                    cwd=temp_root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if install_fails:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(sentinel.is_file())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(sentinel.exists())
                    self.assertTrue((old_venv / "bin" / "python").is_file())
                    self.assertEqual(
                        (old_venv / "created-at").read_text(encoding="utf-8"),
                        ".venv",
                    )

                self.assertEqual(list(temp_root.glob(".venv.tmp.*")), [])
                self.assertEqual(list(temp_root.glob(".venv.backup.*")), [])

    def test_create_venv_is_runnable_at_final_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script_path = temp_root / "create_venv.sh"
            shutil.copy(ROOT / "create_venv.sh", script_path)
            (temp_root / "dev_requirements.txt").write_text("", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=temp_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            venv_dir = temp_root / ".venv"
            pip_result = subprocess.run(
                [str(venv_dir / "bin" / "pip"), "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(pip_result.returncode, 0, pip_result.stderr)

            activate_result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1/bin/activate"; python -c "import sys; print(sys.prefix)"',
                    "bash",
                    str(venv_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(activate_result.returncode, 0, activate_result.stderr)
            self.assertEqual(Path(activate_result.stdout.strip()), venv_dir.resolve())

    def test_create_venv_validates_requirements_before_replacing_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script_path = temp_root / "create_venv.sh"
            shutil.copy(ROOT / "create_venv.sh", script_path)

            old_venv = temp_root / ".venv"
            old_venv.mkdir()
            sentinel = old_venv / "original-environment"
            sentinel.write_text("keep", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=temp_root,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dev_requirements.txt", result.stderr)
            self.assertTrue(sentinel.is_file())

    def test_active_requests_directory_setup_is_non_destructive(self):
        globals_tree = ast.parse((ROOT / "src" / "globals.py").read_text())
        destructive_values = []

        for node in ast.walk(globals_tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "mkdir":
                continue
            for keyword in node.keywords:
                if keyword.arg == "remove_content_if_exists":
                    destructive_values.append(ast.literal_eval(keyword.value))

        self.assertNotIn(True, destructive_values)


if __name__ == "__main__":
    unittest.main()
