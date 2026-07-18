#!/usr/bin/env python3
"""Fleet dashboard — in-place TUI with TUNNELS + FLEET + node grid + EVENTS.

Called by proxy-ai.cmd:
    proxy-ai check        -> one-shot render
    proxy-ai watch [SECS] -> in-place refresh loop with live countdown

Features:
  - Stable panel (cursor-home redraw, no scroll/paste)
  - Unicode box-drawing chars, color-coded status dots + node bars
  - TUNNELS panel (daily fleet from .serve-state-*.json)
  - FLEET panel (monitor state + inline /hot-swap hint when actionable)
  - NODE GRID: all 8 fat nodes as colored GPU bars (green=free, red=used, X=drained)
  - EVENTS panel (recent ALERT/INFO with time-only timestamps)
  - Live countdown to next refresh (1s UI ticks, cadence-bounded data fetch)
  - Terminal bell on new ALERT

Observer data only — this never triggers anything. Operator uses /hot-swap.

NOTE: Sanitized reference copy. Replace <account>, login host, jobnames, and
node-name parsing with your cluster's conventions before running.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LLM_BASE = "/shared/project/<account>/llm"

TUNNELS = [
    # (label, state_file, hpc_port, jobname, local_tunnel_port, local_proxy_port, local_logger_port)
    # jobname lets us resolve the REAL running job via squeue (ground truth),
    # because the state file can be clobbered/stale (experiment scripts writing
    # to the same file). hpc_port = the remote model port. local_tunnel_port =
    # the laptop-side SSH-forward port (usually == hpc_port). local_proxy_port =
    # the laptop-side translator proxy (claude-code-proxy). local_logger_port =
    # the usage-logger port (wedges between claude and the proxy — writes JSONL);
    # None for models without a logger layer (Inkling/Kimi). Panel can't probe
    # 127.0.0.1 itself (runs on HPC), so local health is passed in via --local.
    #
    # jobname CAN be:
    #   - a string  -> exact squeue match
    #   - a tuple/list -> tries each alt in order, first RUNNING match wins
    #   - None      -> pure state-file mode: node+jobid come from state_file only
    #                  (used for aliases like GLM-old, pinned to a specific job)
    ("GLM-5.2 FP8",   ".serve-state-glm.json",        8103, ("glm52-serve", "glm52-noarfusion"), 8103, 5007, 5021),
    ("GLM-alt NVFP4", ".serve-state-glm-nvfp4.json",  8106, "glm52nvfp4",  8106, 5027, 5023),
    # GLM REAP-504B: 0xSero/GLM-5.2-504B (REAP-pruned 168/256 experts + NVFP4).
    # 4x B200, 3d QoS, port 8109, NATIVE 1M context (KV cache holds 1.45M tokens).
    # Distinct from the alt (full 754B NVFP4, 512K): REAP is the long-ctx-on-half-hw
    # niche. 70.5% TB — NOT a coding-model replacement.
    ("GLM REAP-504B", ".serve-state-glm52-reap.json", 8109, "glm52-reap",  8109, 5017, 5029),
    # GLM-old alias: pinned to whatever job the state file names (usually a
    # pre-swap primary kept alive until wall expiry). jobname=None so we don't
    # collide with the primary GLM row on the diagnostic jobname variant. When
    # the state file's job is dead, the health probe fails -> healer stays quiet,
    # no error spam. To retire this row, just delete the state file.
    ("GLM-5.2 old",   ".serve-state-glm-old.json",    8103, None,          8113, 5033, 5031),
    ("Inkling NVFP4", ".serve-state-ink.json",        8110, "inkling-serve", 8110, 5015, 5025),
    # Kimi: two serve variants share ONE port 8104 + ONE state file
    # + ONE served-model-name (kimi-k2.7). ONE AT A TIME. K2.7-Code is primary
    # (always-thinks, vision via MoonViT via --mm-encoder-tp-mode data); K2.6
    # (serve-kimi-k2.sh) is the fallback that CAN suppress thinking.
    # No usage logger for Kimi (user requested minimal background procs).
    ("Kimi K2.7",     ".serve-state-kimi.json",       8104, ("kimi-k27-serve", "kimi-k2-serve"), 8104, 5008, None),
]

# Local-stack health passed in from proxy-ai.cmd (runs on the laptop, can reach
# 127.0.0.1). Format: "tunnelport=0/1,proxyport=0/1,..." e.g.
# "8103=1,5007=1,8106=1,5027=0,8109=0". Parsed once per render in one-shot mode;
# in watch mode the panel can't refresh it (CMD passes a snapshot), so watch shows
# the snapshot with a staleness note. Empty/missing -> local column shows dim '?'.
_LOCAL_HEALTH = {}

BAD_STATES = ("drain", "down", "maint", "fail", "resv")

ANSI_HOME = "\x1b[H"
ANSI_CLEAR = "\x1b[2J"
ANSI_CLEAR_LINE = "\x1b[K"
ANSI_HIDE = "\x1b[?25l"
ANSI_SHOW = "\x1b[?25h"
ANSI_RESET = "\x1b[0m"
ANSI_DIM = "\x1b[2m"
ANSI_GREEN = "\x1b[32m"
ANSI_RED = "\x1b[31m"
ANSI_YELLOW = "\x1b[33m"
ANSI_CYAN = "\x1b[36m"
ANSI_BOLD = "\x1b[1m"
ANSI_BELL = "\a"

W = 72

# Cache for squeue wall-left probes: {jobid: (probe_time, wall_left_str)}.
# Avoids re-probing squeue on every 1s countdown tick.
_WL_CACHE = {}
_WL_TTL = 30  # seconds before re-probing a jobid's wall-left

# Cache for our-jobs-by-node squeue probe (which of OUR jobs are on each fat node,
# and how many B200s each holds). Refreshed once per watch cycle so the 1s
# countdown ticks don't hammer squeue. One-shot mode fetches on first render.
_OUR_JOBS_CACHE = {"t": 0, "data": {}}
_OUR_JOBS_TTL = 30


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return r.stdout or ""
    except Exception:
        return ""


def box(title, rows):
    lines = []
    pad = W - len(title) - 2
    if pad < 3:
        pad = 3
    lines.append("┌─ " + title + " " + "─" * (pad - 1) + "┐")
    for r in rows:
        inner = W - 2
        visible = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", r)
        if len(visible) > inner:
            out_chars, vis_count, i = [], 0, 0
            while i < len(r) and vis_count < inner:
                if r[i] == "\x1b":
                    m = re.match(r"\x1b\[[0-9;?]*[a-zA-Z]", r[i:])
                    if m:
                        out_chars.append(m.group(0))
                        i += len(m.group(0))
                    else:
                        out_chars.append(r[i]); i += 1
                else:
                    out_chars.append(r[i]); vis_count += 1; i += 1
            r = "".join(out_chars) + ANSI_RESET
            visible = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", r)
        lines.append("│ " + r + " " * (inner - len(visible)) + " │")
    lines.append("└" + "─" * (W - 2) + "┘")
    return lines


def load_state(state_path):
    if not state_path or not os.path.exists(state_path):
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception:
        return None


def probe_health(node, port):
    """Probe node:port/health. Returns True if the model responds healthy.
    Runs on the HPC login node, which can reach compute nodes directly.
    Short timeout so an unreachable node doesn't stall the panel."""
    if not node or node == "-" or node == "N/A":
        return False
    try:
        r = subprocess.run(
            ["curl", "-sf", "--connect-timeout", "3", "--max-time", "5",
             f"http://{node}:{port}/health"],
            capture_output=True, text=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def resolve_job(jobname):
    """Find the real RUNNING job for a jobname via squeue (ground truth).
    Returns (jobid, node) or (None, None) if no RUNNING job matches.
    The state file can be clobbered/stale, so squeue is the source of truth
    for which job is actually running and on which node.

    jobname may be a single string OR a tuple/list of alternatives (e.g. two
    serve variants share one TUNNELS row but have distinct jobnames; only one
    runs at a time). First RUNNING match wins."""
    names = jobname if isinstance(jobname, (tuple, list)) else [jobname]
    out = sh(f"squeue -u $USER -h -o '%i|%j|%N|%T' 2>/dev/null")
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid, jname, node, state = parts[0], parts[1], parts[2], parts[3]
        if jname in names and state == "RUNNING" and node.strip():
            return (jid, node.strip())
    return (None, None)


def tunnel_status(state_file, port, jobname):
    """Determine tunnel status from GROUND TRUTH (squeue + health probe),
    not the state file (which can be clobbered/stale — e.g. experiment scripts
    writing to the same file).

    Logic:
      - resolve the real RUNNING job by jobname via squeue -> real node + jobid
      - probe health at real_node:expected_port
      - healthy -> up (green)
      - job RUNNING but health fails -> booting (yellow, still loading)
      - no RUNNING job -> fall back to state file (maybe loading), else down
    """
    jobid, node = resolve_job(jobname)
    if jobid is None:
        # no running job — fall back to state file for a hint (maybe loading)
        path = os.path.join(LLM_BASE, state_file)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                file_status = d.get("status", "")
                node = d.get("node", "-")
                jobid = str(d.get("job_id", "-"))
                if file_status == "loading" and probe_health(node, port):
                    return ("●", "up", jobid, node)  # actually up, file lagging
                if file_status == "loading":
                    return ("◐", "loading", jobid, node)
                return ("○", "down", jobid, node)
            except Exception:
                return ("○", "bad file", "-", "-")
        return ("○", "not deployed", "-", "-")

    # Real RUNNING job — probe its health on the expected port.
    if probe_health(node, port):
        return ("●", "up", jobid, node)
    return ("◐", "booting", jobid, node)  # RUNNING but not responding yet


def wall_left_for_job(jobid, use_cache=True):
    if not jobid or jobid == "-":
        return "-"
    now = time.time()
    if use_cache and jobid in _WL_CACHE:
        t, v = _WL_CACHE[jobid]
        if now - t < _WL_TTL:
            return v
    out = sh(f"squeue -j {jobid} -h -o '%L' 2>/dev/null")
    s = out.strip()
    if not s or s == "UNLIMITED":
        result = "-"
    else:
        m = re.match(r"(\d+)-(\d+):(\d+):(\d+)", s)
        if m:
            d, h, _, _ = m.groups()
            result = f"{d}d{h}h"
        else:
            m = re.match(r"(\d+):(\d+):(\d+)", s)
            if m:
                h, _, _ = m.groups()
                result = f"{int(h)}h"
            else:
                m = re.match(r"(\d+):(\d+)", s)
                result = f"{m.group(1)}m" if m else s
    _WL_CACHE[jobid] = (now, result)
    return result


# Map our known jobnames to a short label for the node-grid legend.
JOBNAME_TAGS = {
    "glm52-serve":      "GLM",
    "glm52-noarfusion": "GLM-old",
    "glm52nvfp4":       "NVFP4",
    "glm52reap":        "REAP",
    "glm52-reap":       "REAP",
    "glm52nvfp4-700k":  "700K",
    "glm52nvfp4wm":     "WARM",
    "inkling-serve":    "INK",
    "minimax-m3":       "MM3",
    "minimax-m3-nvfp4": "MM3-N",
    "minimax-m3-serve": "MM3-X",
    "gemma4-b200":      "GEM",
    "v4pro":            "V4P",
    "kimi-k27-serve":   "K2.7",
    "kimi-k2-serve":    "K2.6",
    "qwen235b":         "QWN",
}


def our_jobs_by_node(use_cache=True):
    """Our running jobs grouped by node. Returns {node: [{jobname, gpus}]}.

    Used by the node grid to mark which of OUR jobs are on each node (cyan bars)
    vs other-teams' usage (red). Refreshed once per cycle (cached `_OUR_JOBS_TTL`)
    so 1s countdown ticks don't hammer squeue."""
    now = time.time()
    if use_cache and now - _OUR_JOBS_CACHE["t"] < _OUR_JOBS_TTL:
        return _OUR_JOBS_CACHE["data"]
    # %N = nodes, %j jobname, %b GRES, %T state. Filter RUNNING with a node.
    out = sh("squeue -u $USER -h -o '%N|%j|%b|%T' 2>/dev/null")
    by_node = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        node, jobname, gres, state = parts[0], parts[1], parts[2], parts[3]
        if state != "RUNNING" or not node.strip():
            continue
        gpus = 0
        m = re.search(r"b200[=:](\d+)", gres)
        if m:
            gpus = int(m.group(1))
        if gpus == 0:
            continue  # cpu-only job (e.g. mmm-serve) — no bar contribution
        by_node.setdefault(node.strip(), []).append({"jobname": jobname, "gpus": gpus})
    _OUR_JOBS_CACHE["data"] = by_node
    _OUR_JOBS_CACHE["t"] = now
    return by_node


def parse_local_health(spec):
    """Parse the --local arg into {port: state}. Format: '8103=up,5007=down,...'
    or legacy '8103=1,5007=0'. States: up / listening / down. Empty/None -> {}.
    Three states (not bool) so the panel can show a busy tunnel as yellow rather
    than red, and so the non-destructive heal never kills a working listener."""
    out = {}
    if not spec:
        return out
    for pair in spec.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        p, v = pair.split("=", 1)
        p = p.strip()
        v = v.strip().lower()
        try:
            port = int(p)
        except ValueError:
            continue
        if v in ("1", "true", "up", "healthy"):
            out[port] = "up"
        elif v in ("0", "false", "down", "dn"):
            out[port] = "down"
        elif v in ("2", "listening", "busy", "booting", "lst"):
            out[port] = "listening"
        else:
            out[port] = "down"
    return out


def _local_dot(port, lh):
    """Local-stack dot for a port from the parsed local-health map.
    Returns (dot_str, word). '?' (dim) when no data passed in (panel ran on HPC
    without --local, e.g. one-off ssh from a shell).

    Three states (the non-destructive-heal fix):
      "up"        -> green ●   listener exists AND /health responds
      "listening" -> yellow ◐  listener exists, /health timed out (busy/booting)
      "down"/False-> red ○      no listener (process died)
    Legacy bool input still works: True->"up", False->"down"."""
    if port is None:
        return (".", "n/a")
    if port not in lh:
        return (ANSI_DIM + "?" + ANSI_RESET, "?")
    v = lh[port]
    if v == "up" or v is True:
        return (ANSI_GREEN + "●" + ANSI_RESET, "up")
    if v == "listening":
        return (ANSI_YELLOW + "◐" + ANSI_RESET, "lst")
    return (ANSI_RED + "○" + ANSI_RESET, "dn")


def render_tunnels(s):
    rows = []
    lh = _LOCAL_HEALTH
    # When run via fleet_watch_local (laptop), s["_tunnels"] holds the HPC-side
    # resolution (jobid/node/healthy fetched remotely via ssh, since the laptop
    # can't run squeue or reach compute nodes). Prefer that over the local
    # tunnel_status() call, which would resolve_job() via squeue here and fail
    # (no squeue on laptop) -> "not deployed" for healthy models.
    # Key pre by LOCAL tunnel port (each row's identity from the laptop's PoV) —
    # keyed by hpc_port would collide when local!=remote (GLM-old shares hpc_port
    # 8103 with the primary GLM but binds locally on 8113).
    pre = {t["port"]: t for t in s.get("_tunnels", [])} if isinstance(s, dict) else {}
    for row in TUNNELS:
        # Backwards-compat: legacy 6-tuple rows have no logger port; treat as None.
        if len(row) == 7:
            label, sf, hpc_port, jobname, lt_port, lp_port, ll_port = row
        else:
            label, sf, hpc_port, jobname, lt_port, lp_port = row
            ll_port = None
        if lt_port in pre:
            t = pre[lt_port]
            jobid = t.get("jobid") or "-"
            node = t.get("node") or "-"
            if t.get("healthy"):
                dot, word = "●", "up"
            elif jobid != "-":
                dot, word = "◐", "booting"
            else:
                dot, word = "○", "not deployed"
        else:
            dot, word, jobid, node = tunnel_status(sf, hpc_port, jobname)
        # up=green, loading/booting/stale/up?=yellow, down/bad/not=red
        if word == "up":
            dot_c = ANSI_GREEN + dot + ANSI_RESET
        elif word in ("loading", "booting", "stale", "up?"):
            dot_c = ANSI_YELLOW + dot + ANSI_RESET
        else:
            dot_c = ANSI_RED + dot + ANSI_RESET
        # show wall-left only for genuinely-up tunnels (healthy). Use the
        # pre-fetched wall_left from _tunnels when available (laptop path can't
        # run squeue -j for wall-left either).
        if word == "up":
            wl = pre.get(lt_port, {}).get("wall_left") or wall_left_for_job(jobid)
        else:
            wl = ""
        jid_s = str(jobid)[:8].rjust(8) if jobid and jobid != "-" else "       -"
        node_s = (node or "-")[:13].ljust(13)
        wl_s = (wl or "").rjust(6)
        right = f"{jid_s} {node_s} {wl_s}"
        # Local-stack column: tunnel + proxy + logger dots (e.g. "●●●" all up).
        # Logger dot omitted (kept as "." spacer) for models without a logger
        # so the column stays 3-wide and rows align.
        lt_dot, _ = _local_dot(lt_port, lh)
        lp_dot, _ = _local_dot(lp_port, lh)
        ll_dot, _ = _local_dot(ll_port, lh) if ll_port else (ANSI_DIM + "." + ANSI_RESET, "n/a")
        local_str = f"{lt_dot}{lp_dot}{ll_dot}"
        rows.append(f"{dot_c} {label:<16} :{lt_port} {word:<8} {local_str}  {right}")
    rows.append(ANSI_DIM + "  HPC / local: tunnel+proxy+logger  (●up ◐busy ○dn ?unknown)" + ANSI_RESET)
    return box("TUNNELS", rows)


def render_node_grid(s):
    """All fat nodes as 8-cell GPU bars using solid BLOCKS (█), color-coded:
      █ cyan   = ours (our running jobs' B200s)
      █ dim    = other team
      █ green  = free
      █ red    = drained (whole node unusable)
    Each node tagged with our running jobnames. 2 nodes per row.
    Color is the distinguisher (cyan vs dim vs green is readable at a glance);
    blocks give the solid visual scan the operator wants."""
    fn = sorted(s.get("free_nodes") or [], key=lambda n: n.get("node", ""))
    our_jobs = our_jobs_by_node()
    rows = []
    cells = []
    for n in fn:
        node = n.get("node", "?")
        label = node[-3:] if len(node) >= 3 else node
        used = n.get("used_b200", 0)
        free = n.get("free_b200", 0)
        state = n.get("state", "").lower()
        drained = any(x in state for x in BAD_STATES)
        if drained:
            bar = ANSI_RED + "█" * 8 + ANSI_RESET
            free_str = ANSI_RED + "drained" + ANSI_RESET
        else:
            ours = our_jobs.get(node, [])
            our_gpus = sum(o["gpus"] for o in ours)
            other_gpus = max(0, used - our_gpus)
            # Solid blocks, color-coded: cyan ours, dim other, green free
            bar = ANSI_CYAN + "█" * our_gpus + ANSI_RESET
            bar += ANSI_DIM + "█" * other_gpus + ANSI_RESET
            bar += ANSI_GREEN + "█" * free + ANSI_RESET
            missing = 8 - our_gpus - other_gpus - free
            if missing > 0:
                bar += ANSI_DIM + " " * missing + ANSI_RESET
            free_str = f"{free} free"
        # Tag with our jobnames (short labels), e.g. "GLM" or "NVFP4+700K"
        ours = our_jobs.get(node, [])
        if ours:
            tags = []
            for o in ours:
                jn = o["jobname"]
                tag = JOBNAME_TAGS.get(jn, jn[:5].upper())
                tags.append(tag)
            tag_raw = "+".join(tags)
            tag_str = ANSI_CYAN + tag_raw + ANSI_RESET
        else:
            tag_raw = "-"
            tag_str = ANSI_DIM + tag_raw + ANSI_RESET
        # Fixed-width trailing fields so column 2 lands straight regardless of
        # tag length or free count. free_str padded to 8 visible, tag to 10.
        # Strip ANSI for visible-width calc (free_str may be colored "drained").
        free_vis = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", free_str)
        free_pad = free_str + " " * max(0, 8 - len(free_vis))
        tag_vis = tag_raw
        tag_pad = tag_str + " " * max(0, 10 - len(tag_vis))
        cells.append(f"{ANSI_BOLD}{label}{ANSI_RESET} [{bar}] {free_pad}{tag_pad}")
    # 2 cells per row
    for i in range(0, len(cells), 2):
        chunk = cells[i:i+2]
        rows.append("  " + "  ".join(chunk))
    # Legend row — solid blocks match the bars
    rows.append(ANSI_DIM + "  legend: " + ANSI_RESET +
               ANSI_CYAN + "█" + ANSI_RESET + ANSI_DIM + "ours  " + ANSI_RESET +
               ANSI_DIM + "█" + ANSI_RESET + ANSI_DIM + "other  " + ANSI_RESET +
               ANSI_GREEN + "█" + ANSI_RESET + ANSI_DIM + "free  " + ANSI_RESET +
               ANSI_RED + "█" + ANSI_RESET + ANSI_DIM + "drained" + ANSI_RESET)
    return box("NODE GRID", rows)


def hot_swap_hint(s):
    """Return an inline /hot-swap hint string if there's an actionable situation."""
    c = s.get("current") or {}
    fn = s.get("free_nodes") or []
    bg = s.get("budget") or {}
    idle = [n["node"] for n in fn
            if n.get("free_b200") == 8
            and not any(x in n.get("state", "").lower() for x in BAD_STATES)]
    free_budget = bg.get("free", 0)

    if not c.get("jobid"):
        if idle:
            return f"{ANSI_BOLD}{ANSI_CYAN}→ /hot-swap{ANSI_RESET}  gap — relaunch on {idle[0]}"
        return f"{ANSI_BOLD}{ANSI_CYAN}→ /hot-swap{ANSI_RESET}  gap — no idle node yet"
    if idle and free_budget > 0:
        return f"{ANSI_BOLD}{ANSI_CYAN}→ /hot-swap {idle[0]}{ANSI_RESET}  opportunity (8x idle, 7d budget free)"
    wl = c.get("wall_left_h")
    if wl is not None and wl < 48:
        return f"{ANSI_BOLD}{ANSI_CYAN}→ /hot-swap{ANSI_RESET}  swap window open ({wl:.0f}h left)"
    return None


def render_fleet(s):
    dm = s.get("daemon") or {}
    c = s.get("current") or {}
    a = s.get("alt") or {}
    r = s.get("reap") or {}
    old = s.get("old") or []
    bg = s.get("budget") or {}

    rows = []
    d_state = dm.get("state", "?")
    d_wl = dm.get("wall_left_h")
    d_wl_str = f"{d_wl}h left" if d_wl is not None else "?"
    if d_state == "running":
        rows.append(f"{ANSI_GREEN}●{ANSI_RESET} monitor    running    {dm.get('jobid','-')}  ({d_wl_str})")
    else:
        rows.append(f"{ANSI_RED}●{ANSI_RESET} monitor    {d_state}    (check mmm-serve job)")

    if not c.get("jobid"):
        rows.append(f"{ANSI_RED}●{ANSI_RESET} current    — none running  {ANSI_BOLD}GAP{ANSI_RESET}")
    else:
        wl = c.get("wall_left_h")
        wl_str = f"{wl}h left" if wl is not None else "?"
        if wl is not None and wl < 12:
            wl_c = ANSI_RED
        elif wl is not None and wl < 48:
            wl_c = ANSI_YELLOW
        else:
            wl_c = ANSI_GREEN
        rows.append(f"{ANSI_GREEN}●{ANSI_RESET} current    {c.get('jobid')}  {c.get('node')}  {c.get('state')}  {wl_c}{wl_str}{ANSI_RESET}")

    # 'old' primaries: legacy jobs kept alive across a hot-swap (operator chose
    # not to cancel until wall expires). Rendered dim so they don't compete with
    # the current row visually; each carries its wall countdown.
    for o in old:
        wl = o.get("wall_left_h")
        wl_str = f"{wl}h left" if wl is not None else "?"
        wl_c = ANSI_RED if (wl is not None and wl < 12) else ANSI_YELLOW if (wl is not None and wl < 48) else ANSI_DIM
        rows.append(f"{ANSI_DIM}●{ANSI_RESET} old        {o.get('jobid')}  {o.get('node')}  {o.get('state')}  {wl_c}{wl_str}{ANSI_RESET}")

    if a.get("jobid"):
        wl = a.get("wall_left_h")
        wl_str = f"{wl}h left" if wl is not None else ""
        rows.append(f"{ANSI_CYAN}●{ANSI_RESET} alt        {a.get('jobid')}  {a.get('node')}  {wl_str}")

    # REAP-504B: the 4th slot — REAP-pruned NVFP4, 1M native ctx, long-ctx niche.
    # Parallel to alt, NOT a replacement (standalone until proven). 3d QoS.
    if r.get("jobid"):
        wl = r.get("wall_left_h")
        wl_str = f"{wl}h left" if wl is not None else ""
        rows.append(f"{ANSI_CYAN}●{ANSI_RESET} reap       {r.get('jobid')}  {r.get('node')}  {r.get('state','?')}  {wl_str}")
    else:
        rows.append(f"{ANSI_DIM}○ reap       — not deployed{ANSI_RESET}")

    used, free_n, cap = bg.get("used", 0), bg.get("free", 0), bg.get("cap", 16)
    if free_n == 0:
        bg_c = ANSI_RED
    elif free_n < 4:
        bg_c = ANSI_YELLOW
    else:
        bg_c = ANSI_GREEN
    rows.append(f"  7d budget  {bg_c}{used}/{cap} used  ({free_n} slots free){ANSI_RESET}")

    cad = s.get("cadence", "?")
    reason = s.get("cadence_reason", "")
    rows.append(f"  cadence    {cad}s  {ANSI_DIM}({reason}){ANSI_RESET}")

    lh = s.get("last_hit")
    rows.append(f"  last hit   {lh if lh else '— (never)'}")

    # Pending jobs row: our queued jobs (snipes/swap candidates waiting to dispatch)
    pending = s.get("pending") or []
    if pending:
        p_str = ", ".join(f"{p.get('jobname','?')} {p.get('jobid')} [{p.get('qos','?')}] {p.get('reason','')}"
                          for p in pending[:3])
        if len(pending) > 3:
            p_str += f"  +{len(pending)-3} more"
        rows.append(f"{ANSI_YELLOW}◐{ANSI_RESET} pending     {p_str}")
    else:
        rows.append(f"{ANSI_DIM}○ pending     none{ANSI_RESET}")

    hint = hot_swap_hint(s)
    if hint:
        rows.append(hint)

    return box("FLEET", rows)


def render_events(events_path):
    rows = []
    if events_path and os.path.exists(events_path):
        with open(events_path) as f:
            ev = f.read().splitlines()[-8:]
        for line in ev:
            if not line.strip():
                continue
            m = re.match(r"\d{4}-\d{2}-\d{2}T(\d{2}:\d{2}:\d{2}).*?(INFO|ALERT|WARN)\s+(.*)", line)
            if m:
                t, lvl, msg = m.groups()
                if lvl == "ALERT":
                    lvl_c = ANSI_RED + lvl + ANSI_RESET
                elif lvl == "WARN":
                    lvl_c = ANSI_YELLOW + lvl + ANSI_RESET
                else:
                    lvl_c = ANSI_DIM + lvl + ANSI_RESET
                rows.append(f"{ANSI_DIM}{t}{ANSI_RESET}  {lvl_c}  {msg}")
            else:
                rows.append(line)
    if not rows:
        rows.append(ANSI_DIM + "(no events)" + ANSI_RESET)
    return box("EVENTS", rows)


def render_header(countdown=None):
    now = datetime.now().strftime("%H:%M:%S")
    cd = f"  {ANSI_DIM}next refresh {countdown}s{ANSI_RESET}" if countdown is not None else ""
    return f"{ANSI_BOLD}proxy-ai fleet dashboard{ANSI_RESET}  {ANSI_DIM}updated {now}{ANSI_RESET}{cd}  {ANSI_DIM}(q to quit){ANSI_RESET}"


def render_once(state_path, events_path, countdown=None):
    lines = [render_header(countdown), ""]
    s = load_state(state_path)
    if s is None:
        lines += box("FLEET", [f"{ANSI_RED}● no state file{ANSI_RESET}",
                               "mmm-serve not deploying monitor?",
                               "ensure serve.sh launches fleet-monitor.py"])
        return "\n".join(lines)
    lines += render_tunnels(s)
    lines += render_fleet(s)
    lines += render_node_grid(s)
    lines += render_events(events_path)
    return "\n".join(lines)


def latest_alert_sig(state_path):
    """Return a signature of the latest ALERT event, or None. Used for bell dedup."""
    s = load_state(state_path)
    if not s:
        return None
    ev = s.get("events_tail") or []
    for line in reversed(ev):
        if "ALERT" in line:
            return line
    return None


def draw(out):
    sys.stdout.write(ANSI_HOME)
    for line in out.split("\n"):
        sys.stdout.write(line + ANSI_CLEAR_LINE + "\r\n")
    # wipe any stale lines below (previous taller render)
    for _ in range(4):
        sys.stdout.write(ANSI_CLEAR_LINE + " \r\n")
    sys.stdout.write(ANSI_HOME)
    sys.stdout.flush()


def _read_key(timeout=1.0):
    """Non-blocking key read. Returns the key char, or None if no key pressed
    within `timeout` seconds. Uses select on stdin (Linux/HPC side) + cbreak
    mode so keys register immediately without Enter."""
    import select
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = sys.stdin.read(1)
            return ch
        return None
    except Exception:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def render_loop(state_path, events_path, interval):
    global _WL_TTL
    _WL_TTL = interval  # cache squeue probes for the full cadence
    last_alert = latest_alert_sig(state_path)
    sys.stdout.write(ANSI_HIDE + ANSI_CLEAR + ANSI_HOME)
    sys.stdout.flush()
    try:
        while True:
            # Full fetch boundary: probe squeue (caches wall-left for `interval`s)
            # and check for a new ALERT to bell.
            new_alert = latest_alert_sig(state_path)
            if new_alert and new_alert != last_alert:
                sys.stdout.write(ANSI_BELL)
                sys.stdout.flush()
            last_alert = new_alert
            # Prime the wall-left cache by probing tunnel jobs once per cycle.
            for row in TUNNELS:
                label, sf, hpc_port, jobname, lt_port, lp_port = row[:6]
                _, _, jobid, _ = tunnel_status(sf, hpc_port, jobname)
                wall_left_for_job(jobid, use_cache=False)
            # Prime the our-jobs-by-node cache (used by the node grid to mark
            # which of our jobs are on each node). One squeue call per cycle.
            our_jobs_by_node(use_cache=False)
            # Countdown loop: 1s UI ticks re-reading state.json (cheap local read),
            # squeue stays cached. Polls for 'r' (refresh now) / 'q' (quit) keys.
            for cd in range(interval, 0, -1):
                out = render_once(state_path, events_path, countdown=cd)
                # Show key hints in header on first tick
                if cd == interval:
                    out = out.replace("(q to quit)", "(q quit, r refresh)")
                draw(out)
                key = _read_key(1.0)
                if key is None:
                    continue
                k = key.lower()
                if k == 'q':
                    return  # clean exit
                if k == 'r':
                    break  # break countdown -> immediate full refresh
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\r\n" * 30 + ANSI_SHOW)
        sys.stdout.flush()


def main():
    global _LOCAL_HEALTH
    args = sys.argv[1:]
    state_path = args[0] if len(args) > 0 else os.path.join(LLM_BASE, ".fleet/state.json")
    events_path = args[1] if len(args) > 1 else os.path.join(LLM_BASE, ".fleet/events.log")
    watch = "--watch" in args
    # --local "port=0/1,..." : laptop-side stack health, probed by proxy-ai.cmd
    # (which CAN reach 127.0.0.1) and passed in. Panel itself runs on HPC and
    # cannot probe the laptop, so this is the only path for local indicators.
    local_spec = None
    for i, a in enumerate(args):
        if a == "--local" and i + 1 < len(args):
            local_spec = args[i + 1]
        elif a.startswith("--local="):
            local_spec = a.split("=", 1)[1]
    _LOCAL_HEALTH = parse_local_health(local_spec)
    interval = 30
    if watch:
        for a in args:
            if a.isdigit():
                interval = max(5, min(3600, int(a)))
    if watch:
        render_loop(state_path, events_path, interval)
    else:
        print(render_once(state_path, events_path))


if __name__ == "__main__":
    main()
