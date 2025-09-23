import os
import typer
import subprocess

from typing import Any

from minisweagent.environments.singularity import SingularityEnvironment

class ApptainerEnvironment(SingularityEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Apptainer action execution server...")
        # self.execute("python -m openhands.runtime.action_execution_server.py 8120")
        print("sandbox_dir:", self.sandbox_dir)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        cmd = [self.config.executable, "exec"]

        # Do not inherit directories and env vars from host
        cmd.extend(["--contain", "--cleanenv", "--no-mount", "hostfs"])

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}