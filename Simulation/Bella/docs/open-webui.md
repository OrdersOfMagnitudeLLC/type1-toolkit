# Connecting Bella to Open WebUI (Optional — requires Docker)

This guide explains how to expose Bella as a tool inside an Open WebUI conversation, so researchers can ask questions like:

> "Run a nitrogen-fixation discovery and email me the PDF report."

## Prerequisites

- A working Bella installation (`bella --help` runs from any directory)
- Docker installed
- 15 minutes

## Step 1: Start Open WebUI

Open WebUI runs as a Docker container. One command:

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Then browse to `http://localhost:3000`, create an admin account, and you are in.

## Step 2: Create the Bella tool file

Inside Open WebUI, go to **Workspace → Tools → Create Tool**. Paste the following definition:

```python
import subprocess
import json
import os

class Tools:
    def __init__(self):
        self.bella_path = os.environ.get("BELLA_PATH", "bella")

    def _run(self, *args):
        try:
            result = subprocess.run(
                [self.bella_path] + list(args),
                capture_output=True,
                text=True,
                timeout=1200,
            )
            return {"output": result.stdout, "errors": result.stderr, "returncode": result.returncode}
        except Exception as e:
            return {"output": "", "errors": str(e), "returncode": 1}

    def discover(self, domain: str = "nitrogen-fixation", sparc_quality: str = "screen") -> dict:
        """Run an automated Bella discovery for the requested domain."""
        return self._run("discover", "--domain", domain, "--generative", "--headless", f"--sparc-quality={sparc_quality}")

    def adsorb(self, formula: str, molecule: str = "N2") -> dict:
        """Compute adsorption energy of a molecule on a material surface."""
        return self._run("adsorb", formula, molecule)

    def report(self, findings_json: str) -> dict:
        """Generate a PDF report from a Bella findings JSON file."""
        return self._run("report", findings_json)
```

Name the tool `bella` and save it.

## Step 3: Configure the tool

In the Open WebUI **Admin Panel → Settings → Tools**, set the environment variable:

```
BELLA_PATH=~/.local/bin/bella
```

Adjust the path to match your system. You can find it with:

```bash
which bella
```

## Step 4: Use Bella in chat

With the tool enabled on a model, prompts like these work:

- `Run a Bella discovery for nitrogen fixation and show the top candidate.`
- `How does Mo2FeN2 adsorb N2?`
- `Generate a PDF report for the latest findings JSON in ~/NS/Bella/findings/`

The model decides whether to call `discover`, `adsorb`, or `report`, then formats Bella's terminal output for the researcher.

## Screenshot placeholders

- **Screenshot 1**: `open_webui_tools_tab.png` — The Open WebUI Workspace → Tools page, with the "Create Tool" button highlighted.
- **Screenshot 2**: `bella_tool_definition.png` — The tool code editor showing the Bella tool definition pasted in.
- **Screenshot 3**: `chat_call_bella.png` — A chat where the assistant invokes `bella discover` and displays the BELLA RESULT block.

Replace these placeholders with real screenshots after your first successful run.

## Troubleshooting

- **"bella not found" inside Open WebUI:** Make sure `BELLA_PATH` is an absolute path and the file is executable. Use `which bella` on the host.
- **Timeouts on discovery:** Increase `timeout=...` in the tool definition to `3600` for confirm-quality runs.
- **PDF not opening:** The container cannot see `~/NS/Bella/findings`. Either mount that folder as a Docker volume or run report on paths the container can access.

## Estimated setup time

15 minutes
