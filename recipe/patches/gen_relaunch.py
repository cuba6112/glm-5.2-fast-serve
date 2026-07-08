#!/usr/bin/env python3
"""Generate relaunch_glm.sh / rollback_glm.sh from the container backup JSON.

Changes vs the original container:
  1. Bind-mounts the patched glm4_moe_tool_parser.py over the site-packages
     copy (fixes streaming drop of zero-argument tool calls).
  2. Adds --reasoning-parser glm45 to the vllm serve command (routes <think>
     reasoning into reasoning_content instead of leaking it into content).

Run on ai-node:  python3 gen_relaunch.py
"""
import json
import shlex

OUT_DIR = "/data/glm-5.2/recipe/patches"
PATCH = f"{OUT_DIR}/glm4_moe_tool_parser.py"
TARGET = "/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/glm4_moe_tool_parser.py"

d = json.load(open("/tmp/glm_container_backup.json"))[0]
cfg, hc = d["Config"], d["HostConfig"]

# ── env file ─────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/glm52.env", "w") as f:
    for e in cfg["Env"]:
        f.write(e + "\n")

# ── modified serve command ───────────────────────────────────────────
cmd = cfg["Cmd"][1]
anchor = "--enable-auto-tool-choice"
assert anchor in cmd, "anchor flag not found in Cmd"
assert "--reasoning-parser" not in cmd, "reasoning parser already set"
new_cmd = cmd.replace(anchor, "--reasoning-parser glm45     " + anchor, 1)

# ── docker run flags from HostConfig ─────────────────────────────────
lines = [
    "docker run -d --name glm-5.2-vllm",
    f"  --network {hc['NetworkMode']}",
    f"  --ipc {hc['IpcMode']}",
    f"  --shm-size {hc['ShmSize']}",
]
if hc.get("Privileged"):
    lines.append("  --privileged")
if hc.get("Runtime") and hc["Runtime"] != "runc":
    lines.append(f"  --runtime {hc['Runtime']}")
for u in hc.get("Ulimits") or []:
    lines.append(f"  --ulimit {u['Name']}={u['Soft']}:{u['Hard']}")
for dr in hc.get("DeviceRequests") or []:
    if dr.get("Count") == -1:
        lines.append("  --gpus all")
lines.append(f"  --restart {hc['RestartPolicy']['Name'] or 'no'}")
lines.append(f"  --env-file {OUT_DIR}/glm52.env")
for b in hc.get("Binds") or []:
    lines.append(f"  -v {b}")
lines.append(f"  -v {PATCH}:{TARGET}:ro")
if cfg.get("WorkingDir"):
    lines.append(f"  -w {cfg['WorkingDir']}")
lines.append(f"  --entrypoint {cfg['Entrypoint'][0]}")
lines.append(f"  {cfg['Image']}")
lines.append(f"  -lc {shlex.quote(new_cmd)}")

run_cmd = " \\\n".join(lines)

with open(f"{OUT_DIR}/relaunch_glm.sh", "w") as f:
    f.write(f"""#!/bin/bash
# Relaunch glm-5.2-vllm with patched tool parser + reasoning parser.
# Generated {__file__}. Original container kept as glm-5.2-vllm-old.
set -euo pipefail
docker stop glm-5.2-vllm
docker rename glm-5.2-vllm glm-5.2-vllm-old
{run_cmd}
echo "New container started. Poll: curl -s localhost:30001/health"
""")

with open(f"{OUT_DIR}/rollback_glm.sh", "w") as f:
    f.write("""#!/bin/bash
# Roll back to the original glm-5.2-vllm container.
set -euo pipefail
docker stop glm-5.2-vllm || true
docker rm glm-5.2-vllm || true
docker rename glm-5.2-vllm-old glm-5.2-vllm
docker start glm-5.2-vllm
echo "Rolled back. Poll: curl -s localhost:30001/health"
""")

print("wrote relaunch_glm.sh / rollback_glm.sh / glm52.env")
print("--- relaunch command preview ---")
print(run_cmd)
