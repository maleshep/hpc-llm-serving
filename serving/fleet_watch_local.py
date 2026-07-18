#!/usr/bin/env python3
"""Local laptop watch loop for the fleet dashboard.

WHY THIS EXISTS (not ssh -t fleet_panel.py --watch):
  The HPC-side watch loop reads 'r'/'q' keystrokes via select+cbreak on the
  REMOTE stdin. Over `ssh -t` from Windows, keystrokes must travel keyboard ->
  Win console -> ssh stdin -> remote PTY -> remote stdin -> select. Two of
  those hops (Win console input mode, ssh stdin forwarding) are unreliable on
  Windows OpenSSH -> 'r' never arrives until Enter is pressed.

  This local loop collapses that to ONE hop: keyboard -> msvcrt.getch() (instant,
  no Enter, no ssh forwarding). It fetches HPC state via a short ssh call each
  cycle (the monitor already writes .fleet/state.json) and probes local 127.0.0.1
  ports directly for LIVE local-stack indicators (the HPC panel can only show a
  snapshot). Renders the same panels as fleet_panel.py.

Usage (from proxy-ai.cmd :watch):
  python fleet_watch_local.py [SECS]   # default 5s refresh

Keys: r = refresh now, q = quit. Ctrl+C also quits.

NOTE: Sanitized reference copy. Replace <account>, the SSH target, and the
PROXY_DIR/LOGGER_DIR paths with your own before running.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# colorama converts ANSI -> Win32 SetConsoleTextAttribute, works even with VT off.
import colorama
import msvcrt

# Reuse the HPC panel's render logic so the look is identical.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fleet_panel as fp  # noqa

LLM_BASE = "/shared/project/<account>/llm"
STATE = f"{LLM_BASE}/.fleet/state.json"
EVENTS = f"{LLM_BASE}/.fleet/events.log"
SSH = "ssh user@hpc-login.example.com"

# (label, local_tunnel_port, local_proxy_port) — must match fleet_panel.TUNNELS
LOCAL_PORTS = [(8103, 5007), (8113, 5033), (8106, 5027), (8110, 5015), (8109, None), (8104, 5008)]

# Per-model proxy restart config (mirrors proxy-ai.cmd do_glm/do_glm_alt/do_minimax/do_kimi).
# tunnel_port -> proxy env: (proxy_port, model, ctx, extra_env)
# extra_env is a dict of env vars to set on the proxy process.
PROXY_CONFIG = {
    8103: dict(proxy_port=5007, model="glm-5.2", ctx=1000000, extra={}),
    # GLM-old alias: same model + ctx as the primary GLM (points at a legacy
    # job kept alive across a hot-swap). No SUMMARIZE_ENABLED; the ONLY
    # summarizing GLM row is :5027 (glm-alt).
    8113: dict(proxy_port=5033, model="glm-5.2", ctx=1048576, extra={}),
    8106: dict(proxy_port=5027, model="glm-5.2-nvfp4", ctx=524288,
               extra={"SUMMARIZE_ENABLED": "1", "SESSION_MEMORY_ENABLED": "0"}),
    # Inkling NVFP4 (4× B200): SUMMARIZE_ENABLED=1 + SESSION_MEMORY disabled,
    # mirrors proxy-ai.cmd :do_ink / :fleet config (port 8110/5015, 1M ctx).
    8110: dict(proxy_port=5015, model="inkling-nvfp4", ctx=1048576,
               extra={"SUMMARIZE_ENABLED": "1", "SESSION_MEMORY_ENABLED": "0"}),
    # Kimi K2.7-Code: always-thinks (proxy sends reasoning_effort=high always).
    # SUMMARIZE_ENABLED=1: 185K ctx is the tightest in the fleet — tool-heavy
    # convos overflow fast without summarization. THINKING_MODE=native (renders
    # reasoning as a native Anthropic thinking block via reasoning_content
    # extractor; text mode leaked thinking as inline "💭 Internal Reasoning"
    # text).
    8104: dict(proxy_port=5008, model="kimi-k2.7", ctx=185000,
               extra={"THINKING_MODE": "native", "SUMMARIZE_ENABLED": "1"}),
    # 8109 (REAP) has no proxy yet
}
PROXY_DIR = os.path.expanduser("~/repo/claude-code-proxy")
LOG_DIR = os.path.expanduser("~/.proxy-ai/logs")

# --- Usage-logger wedges (:5021, :5023, :5025) — the third local layer between
# claude and the proxy that claude-glm.cmd / claude-glm-alt.cmd / claude-ink.cmd spawn.
# Not monitored by fleet_panel.TUNNELS (the dashboard only shows tunnel+proxy);
# they die as collateral when the proxy heal walks a process tree. Scoped
# to glm + glm-alt + ink only (kimi/minimax/etc. left unmanaged by request —
# user does not want extra background processes).
#
# tunnel_port -> logger config: (logger_port, upstream_proxy_port, route_name, log_filename)
# gate: logger heal ONLY runs when its upstream proxy is up (dead upstream = spin-loop).
LOGGER_CONFIG = {
    8103: dict(logger_port=5021, upstream=5007, route="hpc",     usage_log="hpc.jsonl"),
    # GLM-old alias: logger writes to hpc-old.jsonl (kept separate for A/B).
    8113: dict(logger_port=5031, upstream=5033, route="hpc-old", usage_log="hpc-old.jsonl"),
    8106: dict(logger_port=5023, upstream=5027, route="hpc-alt", usage_log="hpc-alt.jsonl"),
    8110: dict(logger_port=5025, upstream=5015, route="hpc-ink", usage_log="hpc-ink.jsonl"),
}
LOGGER_DIR = os.path.expanduser("~/repo/claude-code-proxy/llm-usage-logger")
USAGE_LOG_DIR = os.path.expanduser("~/.proxy-ai/logs/llm-usage")

# Track the last HPC jobid we saw each tunnel serving. Populated by
# heal_local_stack on every cycle. If the incoming jobid differs from the
# cached one AND the local tunnel is still up, we treat it as a same-node
# hot-swap (job died + resubmitted on the same node) and force a repoint
# just like case (b) — even though live_node == node. Motivating case: a GLM
# job wedged on a node → cancelled → a fresh GLM job healthy on the same node.
# Same node, different job, dead upstream port. Node comparison alone missed
# it; jobid comparison catches it.
_last_tunnel_jobid: dict = {}


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
ANSI_BOLD = "\x1b[1m"
ANSI_BELL = "\a"


def fetch_hpc_state():
    """Pull state.json + events.log + resolved tunnel/our-jobs data from HPC via
    ONE ssh call. Returns (state, events_text).

    The local laptop can't run squeue or reach compute nodes, so we delegate the
    HPC-side probes (squeue for job resolution, node:port health, our-jobs-by-node)
    to a python script run on the HPC login node. Base64-encodes the script to
    avoid ALL shell-quoting issues (parens/quotes/semicolons break remote bash)."""
    import base64
    # The remote script computes everything HPC-side and returns one JSON blob:
    #   s = state.json contents
    #   e = last 8 events.log lines
    #   tunnels = [{label, jobname, port, jobid, node, healthy}] for each TUNNEL
    #   our_jobs = {node: [{jobname, gpus}]}  (for the node grid ours/other split)
    # Normalize jobname to a LIST (some rows use a tuple of alternatives;
    # others are a single string). The remote loop tries each, first RUNNING
    # match wins — squeue -n takes one name at a time.
    # Serialize each tunnel row for the remote resolver. Preserve:
    #   label, jobnames (list or None), hpc_port, state_file, local_port.
    # Most rows have local_port == hpc_port; aliases like GLM-old differ
    # (local 8113 -> remote 8103). Healer uses local_port for probing and
    # ssh -L binding, hpc_port for the remote-side probe + ssh -L target.
    tunnels_py = repr([
        (
            t[0],
            (list(t[3]) if isinstance(t[3], (tuple, list)) else ([t[3]] if t[3] else None)),
            t[2],
            t[1],
            t[4],
        )
        for t in fp.TUNNELS
    ])
    script = (
        "import json,os,subprocess,re\n"
        f"STATE='{STATE}'; EVENTS='{EVENTS}'\n"
        "s=open(STATE).read() if os.path.exists(STATE) else '{}'\n"
        "e=open(EVENTS).read().splitlines()[-8:] if os.path.exists(EVENTS) else []\n"
        f"TUNNELS={tunnels_py}\n"
        "def sh(c):\n"
        "    try: return subprocess.run(c,shell=True,capture_output=True,text=True,timeout=10).stdout or ''\n"
        "    except Exception: return ''\n"
        "def probe(node,port):\n"
        "    if not node or node=='-': return False\n"
        "    try:\n"
        "        r=subprocess.run(['curl','-sf','--connect-timeout','3','--max-time','5',f'http://{node}:{port}/health'],capture_output=True,timeout=8)\n"
        "        return r.returncode==0\n"
        "    except Exception: return False\n"
        f"LLM_BASE='{LLM_BASE}'\n"
        "def resolve_from_state(state_file):\n"
        "    # For jobname=None rows (pinned aliases): read node+jobid from the\n"
        "    # HPC state file. Verify the jobid is still RUNNING via squeue;\n"
        "    # dead jobid -> return (None,None,None) so healer stays quiet.\n"
        "    path=os.path.join(LLM_BASE, state_file)\n"
        "    if not os.path.exists(path): return None,None,None\n"
        "    try: st=json.loads(open(path).read())\n"
        "    except Exception: return None,None,None\n"
        "    jid=str(st.get('job_id') or '').strip(); node=(st.get('node') or '').strip()\n"
        "    if not jid or not node: return None,None,None\n"
        "    if sh(f\"squeue -j {jid} -h -t R -o '%i' 2>/dev/null\").strip() != jid:\n"
        "        return None,None,None  # jobid not RUNNING -> alias is stale\n"
        "    return jid, node, st.get('job_id')\n"
        "tunnels=[]\n"
        "for label,jobnames,port,state_file,local_port in TUNNELS:\n"
        "    jid=node=jname=None\n"
        "    if jobnames is None:\n"
        "        jid,node,jname=resolve_from_state(state_file)\n"
        "    else:\n"
        "        for jn in jobnames:\n"
        "            out=sh(f\"squeue -u $USER -h -n {jn} -t R -o '%i|%N'\")\n"
        "            for line in out.splitlines():\n"
        "                parts=line.strip().split('|')\n"
        "                if len(parts)>=2 and parts[1].strip(): jid,node,jname=parts[0].strip(),parts[1].strip().split()[0],jn; break\n"
        "            if jid: break\n"
        "    # port = the LOCAL laptop-side tunnel port (dashboard/healer key).\n"
        "    # remote_port = the HPC-side model port (differs for aliases).\n"
        "    tunnels.append({'label':label,'jobname':jname,'port':local_port,'remote_port':port,'jobid':jid,'node':node,'healthy':probe(node,port) if node else False,'wall_left':sh(f\"squeue -j {jid} -h -o '%L' 2>/dev/null\").strip() if jid else ''})\n"
        "our_out=sh(\"squeue -u $USER -h -o '%N|%j|%b|%T'\")\n"
        "our_jobs={}\n"
        "for line in our_out.splitlines():\n"
        "    parts=line.split('|')\n"
        "    if len(parts)<4 or parts[3]!='RUNNING' or not parts[0].strip(): continue\n"
        "    m=re.search(r'b200[=:](\\d+)',parts[2]); gpus=int(m.group(1)) if m else 0\n"
        "    if gpus==0: continue\n"
        "    our_jobs.setdefault(parts[0].strip(),[]).append({'jobname':parts[1],'gpus':gpus})\n"
        "print(json.dumps({'s':s,'e':e,'tunnels':tunnels,'our_jobs':our_jobs}))\n"
    )
    b64 = base64.b64encode(script.encode()).decode()
    remote = f"echo {b64} | base64 -d | python3"
    try:
        r = subprocess.run(
            SSH.split() + [remote],
            capture_output=True, text=True, timeout=20)
        d = json.loads(r.stdout)
        state = json.loads(d["s"]) if d.get("s") else {}
        state["_tunnels"] = d.get("tunnels", [])
        state["_our_jobs"] = d.get("our_jobs", {})
        return state, "\n".join(d.get("e", []))
    except Exception as e:
        return {"_error": str(e)}, ""


def _listener_pid(port):
    """Return the PID of the process listening on a local port, or None.
    Uses Get-NetTCPConnection (reliable on Win10+). None means NO listener —
    the ssh tunnel / proxy process has actually died, which is the only
    condition under which reopening is safe."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue "
             f"| Select-Object -First 1 -ExpandProperty OwningProcess"],
            capture_output=True, text=True, timeout=6)
        out = (r.stdout or "").strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


def probe_local_ports():
    """Probe local tunnel+proxy ports. Returns {port: "up"|"listening"|"down"}.

    THREE states, not two — this is the fix for the heal-closing-claude-glm bug.
    Previously probe returned a bool (up/down) from a single urllib /health call.
    A model that's listening but slow to answer /health (cold boot, long prefill,
    DeepGEMM JIT) timed out -> bool False -> heal killed the listener -> reopen
    failed -> tunnel dead. Now:
      "up"        = listener exists AND /health (or /) responds -> healthy
      "listening" = listener exists but /health timed out -> busy/booting, DO NOT TOUCH
      "down"      = NO listener -> ssh/proxy process died, safe to reopen/restart
    """
    import urllib.request
    result = {}
    # Tunnel + proxy ports (from LOCAL_PORTS) PLUS logger ports (from LOGGER_CONFIG).
    # Loggers are the client-facing layer for claude-glm / claude-glm-alt.
    ports_to_check = set()
    for tport, pport in LOCAL_PORTS:
        if tport is not None: ports_to_check.add(tport)
        if pport is not None: ports_to_check.add(pport)
    for cfg in LOGGER_CONFIG.values():
        ports_to_check.add(cfg["logger_port"])
    for port in ports_to_check:
        pid = _listener_pid(port)
        if pid is None:
            result[port] = "down"
            continue
        healthy = False
        for path in ("/health", "/"):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
                urllib.request.urlopen(req, timeout=3).read()
                healthy = True
                break
            except Exception:
                pass
        result[port] = "up" if healthy else "listening"
    return result


def log_event(level, msg):
    """Append a line to the HPC events.log so heal actions surface in the EVENTS
    panel. Laptop-independent + persistent across watch sessions. Best-effort."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    line = f"{ts}  {level:<6} {msg}"
    try:
        import base64
        # Append remotely (events.log lives on HPC shared storage). base64 the
        # line to dodge shell quoting. printf to append with newline.
        b64 = base64.b64encode((line + "\n").encode()).decode()
        remote = f"echo {b64} | base64 -d >> {EVENTS}"
        subprocess.run(SSH.split() + [remote], capture_output=True, timeout=10)
    except Exception:
        pass


def kill_port_listeners(port):
    """Kill any process listening on a local port (Windows). Returns count killed."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue "
             f"| ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"],
            capture_output=True, text=True, timeout=10)
        return 1  # best-effort
    except Exception:
        return 0


def reopen_tunnel(local_port, node, remote_port):
    """Open an SSH tunnel 127.0.0.1:local_port -> node:remote_port. Returns True/False.
    Launches ssh DIRECTLY as a subprocess with CREATE_NO_WINDOW — NOT via
    `cmd /c start "title" /MIN ssh ...`. The start/cmd wrapper caused two bugs:
      (1) Windows "cannot find /tunnel 8106/" popups — `start` mis-parsed the
          quoted title when launched detached from Python, surfacing a modal dialog
          each heal cycle.
      (2) Silent reopen failure -> infinite heal loop: the detached ssh either never
          bound or exited immediately (no console to print the error), so the next
          cycle saw "no listener" and reopened again. events.log showed 20+ reopen
          ALERTs for :8106 yet the port never came up. A direct `ssh -L ... -N`
          stays up reliably (verified: plain tunnel survived probes + HEALTH OK).
    Direct Popen with CREATE_NO_WINDOW backgrounds ssh as a real long-lived process
    with proper stdio, no shell, no `start`, no popup.

    NOTE: replace the SSH target below with your own login host/user."""
    try:
        import subprocess as sp
        sp.Popen(
            ["ssh", "-L", f"127.0.0.1:{local_port}:{node}:{remote_port}", "-N",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
             "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=8",
             "user@hpc-login.example.com"],
            creationflags=0x08000000,  # CREATE_NO_WINDOW — no console popup, real process
            stdin=sp.DEVNULL, stdout=sp.DEVNULL, stderr=sp.DEVNULL,
        )
        return True
    except Exception:
        return False


def restart_proxy(tunnel_port, cfg):
    """Restart the claude-code-proxy for a model. cfg = PROXY_CONFIG[tunnel_port]."""
    proxy_port = cfg["proxy_port"]
    model = cfg["model"]
    ctx = cfg["ctx"]
    extra = cfg["extra"]
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = f"{LOG_DIR}\\watch-heal-{proxy_port}.log"
    # Build the env string for the proxy subprocess.
    env_lines = [
        f"set OPENAI_API_KEY=none",
        f"set OPENAI_BASE_URL=http://localhost:{tunnel_port}/v1",
        f"set BIG_MODEL={model}",
        f"set SMALL_MODEL={model}",
        f"set PREFERRED_PROVIDER=openai",
        f"set PORT={proxy_port}",
        f"set HOST=127.0.0.1",
        f"set BACKEND_CONTEXT_LIMIT={ctx}",
    ]
    for k, v in extra.items():
        env_lines.append(f"set {k}={v}")
    env_lines.append("set PYTHONIOENCODING=utf-8")
    env_lines.append("set PYTHONUTF8=1")
    env_block = "&& ".join(env_lines)
    bat_cmd = f'@echo off\r\ncd /d "{PROXY_DIR}"\r\n{env_block}&& uv run server.py >> "{log_file}" 2>&1'
    bat_path = f"{LOG_DIR}\\heal-proxy-{proxy_port}.bat"
    try:
        with open(bat_path, "w") as f:
            f.write(bat_cmd)
        # CREATE_NO_WINDOW (0x08000000), NOT DETACHED_PROCESS (0x8). DETACHED
        # makes a NEW console for the child -> a popup window flashes every heal
        # cycle (the proxy-boot popup the user saw). This mirrors reopen_tunnel's
        # fix. We still use cmd /c the .bat because the env_block is batch syntax
        # (set X=Y&&), but CREATE_NO_WINDOW suppresses the window.
        subprocess.Popen(
            ["cmd", "/c", bat_path],
            creationflags=0x08000000)  # CREATE_NO_WINDOW — no console popup
        return True
    except Exception:
        return False


def restart_logger(tunnel_port, cfg):
    """Restart the usage_logger wedge (claude-code-proxy/llm-usage-logger/usage_logger.py).
    Mirrors restart_proxy's CREATE_NO_WINDOW .bat approach so no console popup
    flashes each heal cycle. cfg = LOGGER_CONFIG[tunnel_port]."""
    lport = cfg["logger_port"]
    upstream = cfg["upstream"]
    route = cfg["route"]
    usage_log = cfg["usage_log"]
    os.makedirs(USAGE_LOG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    stderr_log = f"{LOG_DIR}\\{route}-logger-{lport}.log"
    env_block = "&& ".join([
        f"set UPSTREAM_URL=http://127.0.0.1:{upstream}",
        f"set USAGE_LOG_PATH={USAGE_LOG_DIR}\\{usage_log}",
        f"set ROUTE_NAME={route}",
        f"set HOST=127.0.0.1",
        f"set PORT={lport}",
    ])
    bat_cmd = f'@echo off\r\ncd /d "{LOGGER_DIR}"\r\n{env_block}&& uv run usage_logger.py >> "{stderr_log}" 2>&1'
    bat_path = f"{LOG_DIR}\\heal-logger-{lport}.bat"
    try:
        with open(bat_path, "w") as f:
            f.write(bat_cmd)
        subprocess.Popen(
            ["cmd", "/c", bat_path],
            creationflags=0x08000000)  # CREATE_NO_WINDOW
        return True
    except Exception:
        return False


def _port_listening(port):
    """True if any process is listening on local `port` (127.0.0.1 or 0.0.0.0)."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"[bool](Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue)"],
            capture_output=True, text=True, timeout=6)
        return r.stdout.strip().lower() == "true"
    except Exception:
        return False


def _tunnel_node(port):
    """Return the remote node a local tunnel ssh process points at, or None.
    Parses `ssh -L 127.0.0.1:PORT:NODE:RPORT ...` from live ssh command lines.
    Used to detect a STALE tunnel after a hot-swap (listener alive but pointed
    at a dead/old node)."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='ssh.exe'\" | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8)
        pat = re.compile(rf"127\.0\.0\.1:{port}:([^:\s]+):\d+")
        for line in (r.stdout or "").splitlines():
            m = pat.search(line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def heal_local_stack(tunnels_remote, local_health):
    """Self-heal the LOCAL laptop-side stack (tunnel + proxy), NEVER the HPC job.
    VERIFIED fix-pass — every heal is followed by a re-probe; a reopen that fails
    to bind is retried once, then escalated to STUCK so the operator sees a real
    failure instead of a silent no-op loop.

    Three states per port (NON-DESTRUCTIVE — the fix for the heal-closing-claude-glm bug):
      "down"      = NO local listener -> ssh/proxy died -> reopen/restart (safe).
      "listening" = listener exists, /health timed out -> busy/booting. DO NOT TOUCH.
      "up"        = healthy, nothing to do.

    Two heal triggers:
      (a) "down"                 -> reopen tunnel / restart proxy, then VERIFY.
      (b) stale node             -> listener alive but pointed at a DIFFERENT node
                                    than the live HPC job (hot-swap moved it). Kill
                                    the stale tunnel + reopen to the new node.

    Requires HPC job RUNNING with a known node + jobid (else nothing to heal to).
    Each action logged to events.log so healing surfaces in the EVENTS panel."""
    actions = []
    for t in tunnels_remote:
        label = t["label"]
        tport = t["port"]                            # LAPTOP-side tunnel port (bind + probe)
        rport = t.get("remote_port", tport)          # HPC-side model port (usually same)
        node = t.get("node")
        jobid = t.get("jobid")
        if not jobid or not node:
            continue  # HPC job not running -> nothing to heal to
        tstate = local_health.get(tport, "down")

        # --- (b) stale-tunnel detection: listener alive but pointed at wrong
        # node OR at a dead job on the SAME node (same-node hot-swap). Two
        # sub-cases:
        #   (b1) live_node != node     -> hot-swap moved the model to a new node
        #   (b2) live_node == node BUT last-seen jobid != current jobid
        #                              -> job died + resubmitted on same node;
        #                                 tunnel still forwards to a dead port.
        # Both trigger repoint+reopen. Case (b2) added after a GLM wedge +
        # resubmit on the same node needed manual rewire because case (b1) alone
        # didn't fire.
        if tstate in ("up", "listening"):
            live_node = _tunnel_node(tport)
            last_jid = _last_tunnel_jobid.get(tport)
            node_moved = bool(live_node and live_node != node)
            job_swapped_same_node = bool(
                live_node and live_node == node and last_jid and last_jid != jobid
            )
            if node_moved or job_swapped_same_node:
                if node_moved:
                    reason = f"model moved {live_node}→{node}"
                else:
                    reason = f"job {last_jid}→{jobid} on same node {node}"
                actions.append(f"Repointed {label} tunnel ({reason})")
                log_event("ALERT", f"Repointed {label} tunnel :{tport} — {reason}, reopened")
                kill_port_listeners(tport)
                time.sleep(2)
                reopen_tunnel(tport, node, rport)
                time.sleep(4)
                if not _port_listening(tport):
                    log_event("WARN", f"{label} tunnel :{tport} still down after repoint to {node} - will retry next cycle.")
                    actions.append(f"STUCK: {label} tunnel did not bind after repoint")
                tstate = "up" if _port_listening(tport) else "down"
                local_health[tport] = tstate
        # Cache the current jobid AFTER the repoint decision (so next cycle's
        # comparison uses the value we just repointed to).
        _last_tunnel_jobid[tport] = jobid

        # --- (a) heal tunnel: only when NO listener (ssh died). "listening" = busy, leave it. ---
        if tstate == "down":
            actions.append(f"Reopened {label} tunnel (local tunnel had died, model up on {node})")
            log_event("ALERT", f"Reopened {label} tunnel :{tport} - local tunnel had died, model running on {node}.")
            reopen_tunnel(tport, node, rport)
            time.sleep(4)  # let ssh establish + bind
            # VERIFY the reopen actually bound. Retry once if not.
            if not _port_listening(tport):
                log_event("WARN", f"{label} tunnel :{tport} first reopen did not bind - retrying once.")
                reopen_tunnel(tport, node, rport)
                time.sleep(4)
            if _port_listening(tport):
                local_health[tport] = "up"
                tstate = "up"
            else:
                log_event("WARN", f"{label} tunnel :{tport} still down after 2 reopen attempts - ssh may be failing. Will retry.")
                actions.append(f"STUCK: {label} tunnel did not bind after 2 attempts")
                tstate = "down"

        # --- heal proxy: only when NO listener (proxy process died). ---
        cfg = PROXY_CONFIG.get(tport)
        if cfg:
            pport = cfg["proxy_port"]
            pstate = local_health.get(pport, "down")
            if pstate == "down":
                # Gate on a verified tunnel: a proxy bound to a dead tunnel spin-loops.
                tstate_now = local_health.get(tport, "down")
                if tstate_now == "down":
                    actions.append(f"Skipped {label} proxy (tunnel still down - healing tunnel first)")
                    log_event("WARN", f"Skipped {label} proxy :{pport} - tunnel :{tport} also down, will retry after tunnel heals.")
                    continue
                actions.append(f"Restarted {label} proxy (process had died)")
                log_event("ALERT", f"Restarted {label} proxy :{pport} - no local listener, tunnel up. Relaunching.")
                restart_proxy(tport, cfg)
                time.sleep(4)
                # VERIFY the proxy actually bound.
                if not _port_listening(pport):
                    log_event("WARN", f"{label} proxy :{pport} first restart did not bind - retrying once.")
                    restart_proxy(tport, cfg)
                    time.sleep(4)
                if _port_listening(pport):
                    local_health[pport] = "up"
                else:
                    log_event("WARN", f"{label} proxy did not bind after 2 attempts — check watch-heal-{pport}.log")
                    actions.append(f"STUCK: {label} proxy did not bind after 2 attempts")
            elif pstate == "listening":
                # Proxy process alive but /health slow — do NOT kill. Log a WARN so
                # it's visible if it stays in this state for many cycles.
                log_event("WARN", f"{label} proxy busy (listening, /health slow) — left alone, still booting.")

        # --- heal usage_logger (claude-glm / claude-glm-alt wedge, :5021 / :5023). ---
        # The proxy heal above can walk a process tree and take the logger down as
        # collateral (Windows job-object cascade). Restore it, gated on proxy being
        # up. Only glm + glm-alt tunnels are managed (kimi/minimax/etc. left alone
        # to keep local process count small — user's constraint).
        lcfg = LOGGER_CONFIG.get(tport)
        if lcfg and cfg:  # cfg from PROXY_CONFIG above; both must exist
            lport = lcfg["logger_port"]
            lstate = local_health.get(lport, "down")
            if lstate == "down":
                # Gate on proxy up — a logger pointed at a dead upstream spin-loops.
                pport_now = cfg["proxy_port"]
                pstate_now = local_health.get(pport_now, "down")
                if pstate_now != "up":
                    log_event("WARN", f"Skipped {label} logger :{lport} - upstream proxy :{pport_now} not up, will retry.")
                    continue
                actions.append(f"Restarted {label} logger (process had died)")
                log_event("ALERT", f"Restarted {label} logger :{lport} - no local listener. Relaunching.")
                restart_logger(tport, lcfg)
                time.sleep(4)
                if not _port_listening(lport):
                    log_event("WARN", f"{label} logger :{lport} first restart did not bind - retrying once.")
                    restart_logger(tport, lcfg)
                    time.sleep(4)
                if _port_listening(lport):
                    local_health[lport] = "up"
                else:
                    log_event("WARN", f"{label} logger :{lport} did not bind after 2 attempts — check {lcfg['route']}-logger-{lport}.log")
                    actions.append(f"STUCK: {label} logger did not bind after 2 attempts")
            elif lstate == "listening":
                log_event("WARN", f"{label} logger :{lport} busy (listening, / slow) — left alone.")
    if actions:
        log_event("INFO", f"heal cycle done: {len(actions)} action(s)")
    return actions


def render_once(state, events_text, local_health, countdown=None):
    """Render full dashboard. Reuses fleet_panel's box/render but injects the
    pre-fetched HPC tunnel + our-jobs data (the laptop can't run squeue/probe
    compute nodes, so fetch_hpc_state gathered them remotely)."""
    fp._LOCAL_HEALTH = local_health
    # Inject pre-fetched remote data so the panel's render functions don't try
    # to run squeue/curl locally (which would fail on the laptop).
    # NOTE: state["_tunnels"] is INTENTIONALLY LEFT IN PLACE so fp.render_tunnels
    # can look up rows by local tunnel port. Previously we .pop()'d it and
    # monkey-patched fp.tunnel_status to a jobname-keyed lookup — that broke
    # jobname=None rows (like GLM-old), which fell through as "not deployed"
    # despite the resolver returning healthy=True. render_tunnels now consumes
    # state["_tunnels"] directly; keep the tunnel_status fallback monkey-patch
    # only for legacy code paths that still call it (nothing in the current
    # render path does).
    tunnels_remote = state.get("_tunnels")
    our_jobs_remote = state.pop("_our_jobs", None)
    if tunnels_remote is not None:
        _tunnel_lookup = {t["jobname"]: t for t in tunnels_remote if t.get("jobname")}
        fp.tunnel_status = lambda sf, port, jobname: _tunnel_from_remote(_tunnel_lookup, jobname)
        # wall_left_for_job: return pre-fetched value, no local squeue.
        _wl = {str(t["jobid"]): t.get("wall_left", "") for t in tunnels_remote if t.get("jobid")}
        fp.wall_left_for_job = lambda jobid, use_cache=True: _wl.get(str(jobid), "") or "-"
    if our_jobs_remote is not None:
        fp.our_jobs_by_node = lambda use_cache=True: our_jobs_remote
    now = datetime.now().strftime("%H:%M:%S")
    cd = f"  {ANSI_DIM}next refresh {countdown}s{ANSI_RESET}" if countdown is not None else ""
    lines = [f"{ANSI_BOLD}proxy-ai fleet dashboard{ANSI_RESET}  {ANSI_DIM}updated {now}{ANSI_RESET}{cd}  {ANSI_DIM}(q quit, r refresh){ANSI_RESET}", ""]

    if not state or state.get("_error"):
        lines += fp.box("FLEET", [f"{ANSI_RED}* no state ({state.get('_error','ssh failed')}){ANSI_RESET}",
                                   "check mmm-serve job + monitor"])
        return "\n".join(lines)

    lines += fp.render_tunnels(state)
    lines += fp.render_fleet(state)
    lines += fp.render_node_grid(state)
    ev_rows = []
    for line in events_text.splitlines()[-8:]:
        line = line.strip()
        if not line:
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
            # Truncate so the FULL row (time + level + msg) fits the 70-char box
            # interior with a clean ellipsis. Row overhead = "HH:MM:SS LEVEL " = 15
            # chars (level max width 5), so msg gets ~52 before ellipsis. Without
            # this cap, box() cuts the row mid-word with no ellipsis -> "leak".
            msg = msg.replace("watch ", "", 1) if msg.startswith("watch ") else msg
            if len(msg) > 52:
                msg = msg[:51] + "…"
            ev_rows.append(f"{ANSI_DIM}{t}{ANSI_RESET} {lvl_c} {msg}")
        else:
            # Unknown format - still truncate so it never leaks past the box.
            ev_rows.append(line[:53] + ("…" if len(line) > 53 else ""))
    if not ev_rows:
        ev_rows = [ANSI_DIM + "(no events)" + ANSI_RESET]
    lines += fp.box("EVENTS", ev_rows)
    return "\n".join(lines)


def _tunnel_from_remote(lookup, jobname):
    """Translate a pre-fetched remote tunnel record into the (dot, word, jobid, node)
    tuple that fleet_panel.render_tunnels expects. Avoids the panel trying to run
    squeue/curl locally on the laptop.

    jobname may be a STRING (GLM rows) or a TUPLE of alternatives (e.g. a model
    with two serve variants sharing one row). The lookup is keyed by the
    RESOLVED jobname string, so for a tuple we try each alternative — a
    single-string .get(tuple) always misses, which was the 'not deployed' bug
    (tuple jobname != resolved string)."""
    if isinstance(jobname, (tuple, list)):
        names = list(jobname)
    else:
        names = [jobname]
    t = None
    for jn in names:
        t = lookup.get(jn)
        if t and t.get("jobid"):
            break
    if not t or not t.get("jobid"):
        return ("○", "not deployed", "-", "-")
    node = t.get("node") or "-"
    jobid = str(t["jobid"])
    if t.get("healthy"):
        return ("●", "up", jobid, node)
    return ("◐", "booting", jobid, node)


def draw(out):
    sys.stdout.write(ANSI_HOME)
    for line in out.split("\n"):
        sys.stdout.write(line + ANSI_CLEAR_LINE + "\r\n")
    for _ in range(4):
        sys.stdout.write(ANSI_CLEAR_LINE + " \r\n")
    sys.stdout.write(ANSI_HOME)
    sys.stdout.flush()


def main():
    colorama.init()
    interval = 5
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        interval = max(2, min(3600, int(sys.argv[1])))

    sys.stdout.write(ANSI_HIDE + ANSI_CLEAR + ANSI_HOME)
    sys.stdout.flush()
    last_state = {}
    last_events = ""
    last_alert_sig = None
    # Persist heal banners across cycles so the operator actually sees them.
    # Without this, the banner was drawn for ONE 1s tick then vanished on the
    # next render_once (which doesn't know about heal_actions). Keep the most
    # recent heal visible for HEAL_BANNER_TTL refresh cycles.
    HEAL_BANNER_TTL = 12  # ~60s at 5s refresh
    heal_banner = None
    heal_banner_ttl = 0

    try:
        while True:
            # Full refresh boundary: fetch HPC state + probe local ports.
            state, events = fetch_hpc_state()
            local_health = probe_local_ports()
            last_state, last_events = state, events

            # SELF-HEAL: if HPC job is RUNNING but local tunnel or proxy is down,
            # reopen the tunnel / restart the proxy. Each action logged to
            # events.log so it surfaces in the EVENTS panel (the design: watch
            # heals, and the healing shows up in events).
            tunnels_remote = state.get("_tunnels", [])
            heal_actions = heal_local_stack(tunnels_remote, local_health)
            if heal_actions:
                # Re-probe after healing so the dashboard reflects the new state.
                local_health = probe_local_ports()
                # Re-fetch events so the just-appended heal lines show.
                _, events = fetch_hpc_state()
                last_events = events
                last_alert_sig = None  # let the bell re-arm fresh

            # Bell on new ALERT
            new_sig = None
            for line in reversed(events.splitlines()):
                if "ALERT" in line:
                    new_sig = line
                    break
            if new_sig and new_sig != last_alert_sig:
                sys.stdout.write(ANSI_BELL)
            last_alert_sig = new_sig

            out = render_once(state, events, local_health, countdown=interval)
            if heal_actions:
                # Surface the heal banner at the top so the operator sees what happened.
                # Cap width so it never leaks past the viewport; the same actions are
                # logged to events.log (persistent record) by log_event inside heal.
                joined = "; ".join(heal_actions)
                if len(joined) > 66:
                    joined = joined[:65] + "…"
                heal_banner = ANSI_YELLOW + f"  >> HEALED {datetime.now().strftime('%H:%M:%S')}: {joined}" + ANSI_RESET
                heal_banner_ttl = HEAL_BANNER_TTL
            if heal_banner and heal_banner_ttl > 0:
                out = heal_banner + "\n" + out
                heal_banner_ttl -= 1
            elif heal_banner:
                heal_banner = None  # expired
            draw(out)

            # Countdown: 1s ticks, redraw from cached state, poll for r/q keys.
            # Local ports re-probed every 5s (not every 1s tick) to avoid
            # hammering localhost with urllib probes ~10x/s.
            lh_cached = local_health
            for cd in range(interval, 0, -1):
                if cd % 5 == 0:
                    lh_cached = probe_local_ports()
                out = render_once(last_state, last_events, lh_cached, countdown=cd)
                if heal_banner and heal_banner_ttl > 0:
                    out = heal_banner + "\n" + out
                draw(out)
                # Poll keys for ~1s in 0.1s slices for snappy r/q response.
                for _ in range(10):
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"r", b"R"):
                            # Immediate feedback BEFORE the ~15-25s SSH fetch —
                            # without this the keypress felt dead (fetch has no
                            # visible progress). Draw a banner, then break to the
                            # outer loop which does fetch_hpc_state + redraw.
                            sys.stdout.write(ANSI_HOME)
                            sys.stdout.write(ANSI_YELLOW + "  >> refreshing (fetching HPC state...)" + ANSI_RESET + ANSI_CLEAR_LINE + "\r\n")
                            for _ in range(3):
                                sys.stdout.write(ANSI_CLEAR_LINE + " \r\n")
                            sys.stdout.write(ANSI_HOME)
                            sys.stdout.flush()
                            break  # break countdown -> immediate full refresh
                        if ch in (b"q", b"Q", b"\x03"):  # q or Ctrl+C
                            return
                    time.sleep(0.1)
                else:
                    continue
                break  # r pressed -> restart outer loop (full refresh)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\r\n" * 30 + ANSI_SHOW)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
