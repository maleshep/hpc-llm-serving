#!/usr/bin/env python3
"""Fleet monitor — HPC observer (runs inside the mmm-serve cpu job).

OBSERVER ONLY. Never sbatches, never scancels. Every GPU action stays with the
operator, invoked via the /hot-swap skill. This daemon only:
  - polls sinfo/squeue with adaptive cadence (30s -> 15/10/5s as wall-left shrinks)
  - writes a state snapshot to state.json every cycle
  - appends ALERT lines to events.log (and optional ntfy.sh push)
  - detects opportunities (idle node + 7d budget room) and gaps (job died)

Launched by mmm-serve serve.sh as a background loop. Laptop-independent: keeps
running 7 days straight regardless of whether `proxy-ai watch` is open.

Optional env:
  NTFY_TOPIC   if set, pushes ALERT lines to ntfy.sh/<topic> (phone). Off if empty.
  POLL_OVERRIDE if set, fixes cadence to this many seconds (disables adaptive).

NOTE: This is a sanitized reference copy. Replace <account>, the login host, and
the jobname constants with your cluster's values before running.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

STATE_DIR = "/shared/project/<account>/llm/.fleet"
STATE_FILE = os.path.join(STATE_DIR, "state.json")
EVENTS_FILE = os.path.join(STATE_DIR, "events.log")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
POLL_OVERRIDE = os.environ.get("POLL_OVERRIDE", "").strip()

ACCOUNT = os.environ.get("USER", "your-user")
FAT_PARTITION = "fat"
B200_PER_FAT_NODE = 8
QOS_7D_CAP_B200 = 16  # cluster-wide GrpTRES gres/gpu:b200 cap on the 7d QoS

# jobname(s) of the primary GLM-5.2 FP8 job (8x B200, port 8103) — see serve-glm52-sglang-latest.sh
# Tuple: any of these names is treated as "primary". Diagnostic variants (e.g.
# a workaround-named job submitted to dodge an SGLang boot wedge) also count as
# primary so the panel/monitor don't flag a false GAP.
PRIMARY_JOBNAMES = ("glm52-serve", "glm52-noarfusion")
PRIMARY_JOBNAME = PRIMARY_JOBNAMES[0]  # canonical label for legacy readers
# jobname of the NVFP4 alt (4x B200, port 8106) — see serve-glm52-nvfp4.sh
ALT_JOBNAME = "glm52nvfp4"

# Adaptive cadence (seconds) by primary job wall-left. Tighter as runway shrinks
# so the operator gets swap-window alerts early enough to act (boot is ~15min).
# Returns (seconds, reason) so the panel can show WHY this cadence.
def cadence_for(wall_left_secs):
    if POLL_OVERRIDE:
        return int(POLL_OVERRIDE), f"override ({POLL_OVERRIDE}s)"
    if wall_left_secs is None:           # no primary running -> emergency fill scan
        return 5, "emergency (no GLM running)"
    if wall_left_secs < 12 * 3600:       # < 12h
        return 5, "critical (<12h left)"
    if wall_left_secs < 24 * 3600:       # < 1d
        return 10, "urgent (<1d left)"
    if wall_left_secs < 2 * 24 * 3600:  # < 2d
        return 15, "ramp (<2d left)"
    h = wall_left_secs / 3600
    return 30, f"comfortable ({h:.0f}h left)"

# Wall-left thresholds at which to post an EXPIRING alert (each fires once per crossing).
EXPIRY_THRESHOLDS = [
    (2 * 24 * 3600, "2 days"),
    (1 * 24 * 3600, "1 day"),
    (12 * 3600,     "12 hours"),
    (6 * 3600,      "6 hours"),
    (1 * 3600,      "1 hour"),
]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return r.stdout or ""


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_wall_left(s):
    """squeue TimeLeft -> seconds. '6-21:00:00' / '1:30:00' / '0:00' / 'UNLIMITED' / ''."""
    if not s or s in ("UNLIMITED", "0:00", "INVALID"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def daemon_wall_left_h(daemon_jobid):
    """The monitor's own wall-left in hours (it's a 7d cpu job). None if not a Slurm job."""
    if not daemon_jobid or daemon_jobid == "local":
        return None
    out = sh(f"squeue -j {daemon_jobid} -h -o '%L' 2>/dev/null")
    secs = parse_wall_left(out.strip())
    return round(secs / 3600, 1) if secs is not None else None


def pending_jobs():
    """Our PENDING jobs across all partitions. Returns [{jobid, jobname, qos, reason}].
    Reason comes from squeue's %R (e.g. (Priority), (Resources), (QOSGrpGRES))."""
    out = sh(f"squeue -u {ACCOUNT} -h -t PENDING -o '%i|%j|%q|%R'")
    pending = []
    for line in out.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        jid, jname, qos, reason = parts
        pending.append({"jobid": jid, "jobname": jname, "qos": qos,
                        "reason": reason.strip()[:40]})
    return pending


def our_fat_jobs():
    """Our GLM jobs on the fat partition. Returns (primaries: list, alt: dict|None).

    Primaries is a list because we can have multiple simultaneously during a
    hot-swap window (new job on fresh node coexists with old job until wall
    expires or operator cancels). Sorted by wall_left_h DESC so [0] = freshest.
    """
    out = sh(f"squeue -u {ACCOUNT} -h -p {FAT_PARTITION} "
             f"-o '%i|%j|%N|%T|%L|%q|%b'")
    primaries = []
    alt = None
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jobid, jobname, node, state, timeleft = parts[0], parts[1], parts[2], parts[3], parts[4]
        qos = parts[5] if len(parts) > 5 else ""
        gres = parts[6] if len(parts) > 6 else ""
        # PENDING jobs have empty node field — normalize to "-"
        if not node.strip():
            node = "-"
        gpus = 0
        m = re.search(r"b200[=:](\d+)", gres)
        if m:
            gpus = int(m.group(1))
        wl_secs = parse_wall_left(timeleft)
        rec = {
            "jobid": jobid,
            "jobname": jobname,
            "node": node,
            "state": state,
            "wall_left": wl_secs,
            "wall_left_h": round(wl_secs / 3600, 1) if wl_secs is not None else None,
            "qos": qos,
            "gpus": gpus,
        }
        if jobname in PRIMARY_JOBNAMES:
            primaries.append(rec)
        elif jobname == ALT_JOBNAME:
            alt = rec
    # Freshest primary first (longest wall_left = newest submission)
    primaries.sort(key=lambda r: r["wall_left"] or 0, reverse=True)
    return primaries, alt


def free_fat_nodes():
    """Fat nodes with free B200s. Returns [{node, state, free_b200, total_b200, used_b200}].
    Uses sinfo for node list + state, then scontrol show node for AllocTRES (sinfo -O AllocTRES
    returns blank on this Slurm version; scontrol is reliable)."""
    sinfo_out = sh(f"sinfo -p {FAT_PARTITION} -h -N -O 'NodeList,StateLong'")
    node_states = {}
    for line in sinfo_out.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        node = parts[0]
        state = parts[1] if len(parts) > 1 else "?"
        node_states[node] = state
    # One scontrol call for all nodes' AllocTRES — parse the per-node blocks.
    sctrl_out = sh("scontrol show nodes")
    nodes = []
    cur_node = None
    cur_alloc = 0
    for line in sctrl_out.splitlines():
        if line.startswith("NodeName="):
            # flush previous node
            if cur_node in node_states:
                state = node_states[cur_node]
                free = B200_PER_FAT_NODE - cur_alloc
                nodes.append({"node": cur_node, "state": state, "free_b200": free,
                              "total_b200": B200_PER_FAT_NODE, "used_b200": cur_alloc})
            cur_node = line.split("=", 1)[1].split()[0]
            cur_alloc = 0
        elif "AllocTRES=" in line and cur_node in node_states:
            m = re.search(r"gres/gpu:b200=(\d+)", line)
            if m:
                cur_alloc = int(m.group(1))
    # flush last node
    if cur_node in node_states:
        state = node_states[cur_node]
        free = B200_PER_FAT_NODE - cur_alloc
        nodes.append({"node": cur_node, "state": state, "free_b200": free,
                      "total_b200": B200_PER_FAT_NODE, "used_b200": cur_alloc})
    return nodes


def budget_7d():
    """Count B200s in use cluster-wide under qos=7d (running + pending). Cap=16."""
    out = sh("squeue --all -h -O 'JobName,QOS,GRES,State'")
    used = 0
    pending = 0
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        # parts: JobName QOS GRES State  (GRES has no spaces; State last)
        qos = parts[1] if len(parts) > 1 else ""
        gres = parts[2] if len(parts) > 2 else ""
        state = parts[-1]
        if qos != "7d":
            continue
        m = re.search(r"b200[=:](\d+)", gres)
        n = int(m.group(1)) if m else 0
        if state == "RUNNING":
            used += n
        elif state in ("PENDING", "CONFIGURING"):
            pending += n
    total_claimed = used + pending
    return {"used": used, "pending": pending, "cap": QOS_7D_CAP_B200,
            "free": max(0, QOS_7D_CAP_B200 - total_claimed)}


def append_event(level, msg, state):
    line = f"{now_iso()}  {level:<6} {msg}"
    try:
        with open(EVENTS_FILE, "a") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"failed to write events.log: {e}", file=sys.stderr)
    state["events_tail"] = (state.get("events_tail") or [])[-19:] + [line]
    if NTFY_TOPIC and level == "ALERT":
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=(level + ": " + msg).encode(),
                headers={"Title": "GLM fleet", "Tags": "satellite"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            print(f"ntfy push failed: {e}", file=sys.stderr)


def write_state(state):
    state["updated_at"] = now_iso()
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    daemon_jobid = os.environ.get("SLURM_JOB_ID", "local")
    # dedup memory: which alerts have already fired (so we don't spam each cycle)
    fired = set()        # expiry threshold keys already crossed
    last_opp_node = None # last opportunity node alerted (re-alert if node changes)
    last_gap = False     # whether we were in a gap last cycle

    state = {
        "daemon": {"jobid": daemon_jobid, "state": "running",
                   "started_at": now_iso(), "ntfy": bool(NTFY_TOPIC),
                   "wall_left_h": None},
        "current": None,      # freshest primary job snapshot or null
        "old": [],            # legacy primary jobs kept alive across a hot-swap
        "alt": None,          # NVFP4 alt snapshot or null
        "free_nodes": [],
        "budget": {},
        "pending": [],        # our PENDING jobs (jobid, jobname, qos, reason)
        "cadence": 30,
        "cadence_reason": "comfortable",
        "last_hit": None,
        "events_tail": [],
    }
    append_event("INFO", f"monitor started (job {daemon_jobid}, ntfy={'on' if NTFY_TOPIC else 'off'})", state)

    while True:
        try:
            primaries, alt = our_fat_jobs()
            # current = freshest primary (longest wall_left); old = the rest
            primary = primaries[0] if primaries else None
            old_list = primaries[1:] if len(primaries) > 1 else []
            nodes = free_fat_nodes()
            budget = budget_7d()
            pending = pending_jobs()

            state["current"] = primary
            state["old"] = old_list
            state["alt"] = alt
            state["free_nodes"] = nodes
            state["budget"] = budget
            state["pending"] = pending

            # Daemon's own wall-left (it's a 7d cpu job).
            state["daemon"]["wall_left_h"] = daemon_wall_left_h(daemon_jobid)

            wall_left = primary["wall_left"] if primary else None
            cad, reason = cadence_for(wall_left)
            state["cadence"] = cad
            state["cadence_reason"] = reason

            # --- gap detection: primary was running, now gone ---
            if primary is None and not last_gap:
                append_event("ALERT",
                             "GAP — no primary GLM job running (down or wall-killed). "
                             "Run /hot-swap to relaunch when ready.", state)
                last_gap = True
                fired.clear()  # reset expiry dedup; new job starts fresh
            elif primary is not None and last_gap:
                append_event("INFO", f"primary recovered — job {primary['jobid']} on {primary['node']}", state)
                last_gap = False

            # --- expiry alerts (each threshold fires once per job lifetime) ---
            if primary and wall_left is not None:
                for secs, label in EXPIRY_THRESHOLDS:
                    key = f"expiry_{primary['jobid']}_{label}"
                    if wall_left < secs and key not in fired:
                        append_event("ALERT",
                                     f"main job {primary['jobid']} EXPIRING — {label} left "
                                     f"({primary['wall_left_h']}h), swap window open. Run /hot-swap.", state)
                        fired.add(key)

            # --- opportunity detection (idle 8x node + 7d budget room) ---
            # Exclude non-schedulable states: drained/draining/down/maint/etc. can't accept jobs
            # even though they report free_b200==8 (nothing allocated on them).
            if primary is None or (wall_left is not None and wall_left < 2 * 86400):
                schedulable = [n for n in nodes if not any(s in n["state"].lower()
                                                           for s in ("drain", "down", "maint", "fail", "resv"))]
                idle_8x = [n for n in schedulable if n["free_b200"] == B200_PER_FAT_NODE]
                if idle_8x:
                    target = idle_8x[0]
                    if budget["free"] > 0:
                        msg = (f"opportunity — {target['node']} idle (8x B200 free), "
                               f"7d budget has room ({budget['free']}/{QOS_7D_CAP_B200} slots)")
                        if last_opp_node != target["node"]:
                            append_event("ALERT", msg + " — run /hot-swap", state)
                            last_opp_node = target["node"]
                    else:
                        msg = (f"opportunity — {target['node']} idle, but 7d budget FULL "
                               f"({budget['used']}/{QOS_7D_CAP_B200}). Would need --qos=3d.")
                        if last_opp_node != target["node"] + "-full":
                            append_event("ALERT", msg + " — run /hot-swap", state)
                            last_opp_node = target["node"] + "-full"
                else:
                    last_opp_node = None  # node taken -> re-arm for next idle

            write_state(state)
        except Exception as e:
            # Never die on a transient Slurm error — log and keep looping.
            append_event("WARN", f"probe error: {e}", state)
            write_state(state)

        time.sleep(state.get("cadence", 30))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("monitor stopped", file=sys.stderr)
    except Exception as e:
        # fatal: write a final state so the watch panel shows "daemon STOPPED"
        try:
            state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
            state["daemon"] = {"jobid": os.environ.get("SLURM_JOB_ID", "local"),
                               "state": "stopped", "error": str(e),
                               "started_at": state.get("daemon", {}).get("started_at", now_iso())}
            write_state(state)
            append_event("ALERT", f"monitor STOPPED — {e}", state)
        except Exception:
            pass
        raise
