"""Loader for the Rezaee IoT/Fog/Cloud DAG benchmark.

Ali Rezaee & Sahar Adabi, "Jobs (DAG workflow) and tasks dataset with near 50k
job instances and 1.3 Millions of tasks", Zenodo, CC-BY-4.0.
doi:10.5281/zenodo.4667690

This is the dataset the paper we extend cites in Sec. 3.2. Using it means the
DAG structures, task compute, memory, deadlines, arrival times and per-tier
machine speeds all come from outside this project.

Note on the paper: Sec. 3.2 describes this dataset as "about half a million
jobs". It contains 51,895. The task count (1.3M) is correct.

Three quirks make the task file awkward to read, all handled here:
  1. TaskDetails.txt is UTF-16 with a BOM, not UTF-8.
  2. The last three columns are bracketed adjacency lists containing commas.
  3. ListOfResourceNeeds is a variable-length UNBRACKETED comma list, so the
     column count differs from row to row. Fields are therefore anchored from
     both ends and the remainder is assigned to that column.

And one trap worth recording: JobID is NOT unique. The file holds several
simulation runs, each restarting JobID at 0, so only 21,895 of the 51,895 jobs
have a distinct JobID. JobID_InDB is the true key, but the task file links via
OwnerJobID (= JobID). Keying on JobID alone silently merges the tasks of
unrelated jobs -- it inflated task counts roughly 4x before we caught it.
We therefore key on (JobID, TimeSubmission), which is unique for 51,891 of
51,895 jobs; the 4 colliding jobs are dropped.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import zipfile

JOB_ZIP = "data/rezaee/JobDetails.zip"
TASK_ZIP = "data/rezaee/TaskDetails.zip"
CACHE = "data/rezaee/cache.pkl"

HEAD = ["TaskID_InDB", "TaskID", "OwnerJobID", "CPUNeed_Claimed",
        "RAMNeed_Claimed", "StorageNeed_Claimed"]
TAIL = ["TimeSubmission", "TimeDeadLinePrefered", "TimeDeadlineFinal", "Date",
        "SimID", "EST", "EFT", "NodeID_SubmittedOn", "LastStatus",
        "LastStatusTime", "PriorityNo", "CPUNeed_Real", "RAMNeed_Real",
        "StorageNeed_Real", "CPUNeed_Predicted", "RAMNeed_Predicted",
        "StorageNeed_Predicted", "LengthOnScheduledMachine",
        "LengthOnFogAndCloudBaseLineCPU", "BaseLineCloudAndFogCpuMIPS",
        "LengthOnBaseLineCloudMachine", "LengthOnBaseLineFogMachine",
        "LengthOnBaseLineIoTMachine", "BaseLineCpuFogMIPS",
        "BaseLineCpuIoTMIPS", "BaseLineCpuCloudMIPS", "BaseLineCloudBandwidth",
        "BaseLineCloudAndFogBandwidth", "BaseLineIoTBandwidth",
        "BaseLineFogBandwidth"]
BRACKET = ["SuccessorsImediate", "SuccessorsNotImmediate", "PredecessorsImediate"]
N_TAIL = len(TAIL) + len(BRACKET)   # 33


def tokenize(line: str):
    """Split on commas, treating [..] as atomic."""
    out, cur, i, n = [], "", 0, len(line)
    while i < n:
        c = line[i]
        if c == "[":
            j = line.find("]", i)
            if j < 0:
                cur += line[i:]
                break
            cur += line[i:j + 1]
            i = j + 1
        elif c == ",":
            out.append(cur)
            cur = ""
            i += 1
        else:
            cur += c
            i += 1
    out.append(cur)
    return out


def parse_task(line: str):
    t = tokenize(line)
    if len(t) < len(HEAD) + N_TAIL:
        return None
    d = dict(zip(HEAD, t[:len(HEAD)]))
    d.update(zip(TAIL, t[-N_TAIL:-len(BRACKET)]))
    d.update(zip(BRACKET, t[-len(BRACKET):]))
    d["ListOfResourceNeeds"] = ",".join(t[len(HEAD):len(t) - N_TAIL])
    return d


def _ids(s: str):
    s = (s or "").strip().strip("[]").strip()
    if not s:
        return ()
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            try:
                out.append(int(p))
            except ValueError:
                pass
    return tuple(out)


def _f(d, k, default=0.0):
    try:
        return float(d[k])
    except (KeyError, ValueError, TypeError):
        return default


def build_cache(window_start=0.0, window_len=900.0, max_jobs=6000,
                job_zip=JOB_ZIP, task_zip=TASK_ZIP, out=CACHE, verbose=True):
    """Extract one time window of the trace into a compact cache.

    The full task file is 670 MB of UTF-16 text; decoding it on every run would
    dominate the experiment. We stream it once and keep only the jobs whose real
    submission time falls in the requested window.
    """
    with zipfile.ZipFile(job_zip) as z:
        jobs_raw = json.loads(z.read("JobDetails.json").decode("utf-8-sig"))["Jobs"]
    if verbose:
        print(f"jobs in file: {len(jobs_raw):,}")

    # (JobID, TimeSubmission) composite key -- see the module docstring.
    import collections
    collide = {k for k, v in collections.Counter(
        (str(j["JobID"]), str(j["TimeSubmission"])) for j in jobs_raw).items() if v > 1}

    wanted = {}
    for j in jobs_raw:
        sub = _f(j, "TimeSubmission", -1)
        if sub < window_start or sub >= window_start + window_len:
            continue
        key = (str(j["JobID"]), str(j["TimeSubmission"]))
        if key in collide:
            continue
        wanted[key] = {
            "key": key,
            "indb": str(j["JobID_InDB"]),
            "job_id": int(j["JobID"]),
            "arrival": sub,
            "deadline": _f(j, "TimeDeadlineFinal"),
            "n_tasks": int(_f(j, "CountOfTasks")),
            "n_edges": int(_f(j, "CountOfEdges")),
            "sum_edge_w": _f(j, "SumOfEdgeWeight"),
            "max_edge_w": _f(j, "MaxEdgeWeight"),
            "min_edge_w": _f(j, "MinEdgeWeight"),
            "total_mi": _f(j, "JobSizeWithoutAccountForParallelism"),
            "max_par": int(_f(j, "MaxParallelExecutableTasks", 1)),
            "t_cloud": _f(j, "MinTimeForExecuteDagWithMaxParallelismOnCloud"),
            "t_fog": _f(j, "MinTimeForExecuteDagWithMaxParallelismOnFog"),
            "t_iot": _f(j, "MinTimeForExecuteDagWithMaxParallelismOnIoT"),
            "tasks": {},
        }
        if len(wanted) >= max_jobs:
            break
    if verbose:
        print(f"jobs in window [{window_start:.0f},{window_start+window_len:.0f}): "
              f"{len(wanted):,}")

    owners = {k[0] for k in wanted}
    mips = {}
    kept = 0
    with zipfile.ZipFile(task_zip) as z:
        with z.open("TaskDetails.txt") as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-16", errors="replace")
            header = True
            for line in stream:
                if header:
                    header = False
                    continue
                line = line.rstrip("\r\n")
                if not line:
                    continue
                # Cheap pre-filter before the expensive tokenize.
                c1 = line.find(",")
                c2 = line.find(",", c1 + 1)
                c3 = line.find(",", c2 + 1)
                if c3 < 0:
                    continue
                owner = line[c2 + 1:c3]
                if owner not in owners:
                    continue
                d = parse_task(line)
                if d is None:
                    continue
                key = (owner, str(int(_f(d, "TimeSubmission", -1))))
                if key not in wanted:
                    continue
                if not mips:
                    mips = {
                        "cloud": _f(d, "BaseLineCpuCloudMIPS"),
                        "fog": _f(d, "BaseLineCpuFogMIPS"),
                        "iot": _f(d, "BaseLineCpuIoTMIPS"),
                        "bw_cloud": _f(d, "BaseLineCloudBandwidth"),
                        "bw_fog": _f(d, "BaseLineFogBandwidth"),
                        "bw_iot": _f(d, "BaseLineIoTBandwidth"),
                    }
                tid = int(_f(d, "TaskID", -1))
                if tid < 0:
                    continue
                wanted[key]["tasks"][tid] = {
                    "tid": tid,
                    "mi": _f(d, "CPUNeed_Claimed"),
                    "ram_mb": _f(d, "RAMNeed_Claimed"),
                    "storage": _f(d, "StorageNeed_Claimed"),
                    "preds": _ids(d.get("PredecessorsImediate")),
                    "succs": _ids(d.get("SuccessorsImediate")),
                }
                kept += 1
                if verbose and kept % 20000 == 0:
                    print(f"  tasks read: {kept:,}", flush=True)

    # Drop jobs whose task rows are missing or inconsistent.
    good = {}
    for k, j in wanted.items():
        if not j["tasks"]:
            continue
        ids = set(j["tasks"])
        for t in j["tasks"].values():
            t["preds"] = tuple(p for p in t["preds"] if p in ids)
            t["succs"] = tuple(s for s in t["succs"] if s in ids)
        if _has_cycle(j["tasks"]):
            continue
        good[k] = j

    payload = {"jobs": list(good.values()), "mips": mips,
               "window": (window_start, window_len)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"kept {len(good):,} jobs / {kept:,} tasks -> {out}")
        print(f"tier MIPS from the file: {mips}")
    return payload


def _has_cycle(tasks):
    colour = {}

    def visit(n):
        colour[n] = 1
        for s in tasks[n]["succs"]:
            c = colour.get(s, 0)
            if c == 1:
                return True
            if c == 0 and visit(s):
                return True
        colour[n] = 2
        return False

    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(max(lim, 10000))
    try:
        return any(colour.get(n, 0) == 0 and visit(n) for n in tasks)
    except RecursionError:
        return True
    finally:
        sys.setrecursionlimit(lim)


def load_cache(path=CACHE):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--len", type=float, default=900.0)
    ap.add_argument("--max-jobs", type=int, default=6000)
    a = ap.parse_args()
    build_cache(window_start=a.start, window_len=a.len, max_jobs=a.max_jobs)


# ---------------------------------------------------------------------------
# Adapter: turn cached records into the simulator's Job/Task objects.
# ---------------------------------------------------------------------------

class RezaeeWorkload:
    """Trace-driven workload. Arrivals are REPLAYED, not sampled.

    What comes from the dataset: DAG structure, per-task compute (MI), memory,
    job deadlines, arrival times, and the per-tier machine speeds.

    What remains ours, and why:
      * Link bandwidths. The file reports 49,500 for cloud, fog AND IoT alike,
        so it does not differentiate the network at all. We keep the tiered
        bandwidths from the paper's own Sec. 6.1 (WiFi / MAN / WAN) and apply
        them to the dataset's data volumes.
      * Service classes. The dataset has none, so jobs are placed in three
        tightness terciles by deadline / cloud-execution-time ratio. This keeps
        the weighted-SLA machinery while the deadlines themselves stay real.
    """

    KB = 1024 * 8   # edge-weight unit -> bits

    def __init__(self, cache, rng, cfg, repeat=True):
        self.cfg = cfg
        self.rng = rng
        self.recs = sorted(cache["jobs"], key=lambda r: r["arrival"])
        self.mips = cache["mips"]
        self.repeat = repeat
        self._next_task = 0
        self._next_job = 0
        self._bursts = []
        self._assign_classes()

    def _assign_classes(self):
        """Tightness terciles: ratio of deadline budget to cloud execution time."""
        from .entities import JobClass
        ratios = []
        for r in self.recs:
            budget = max(1e-6, r["deadline"] - r["arrival"])
            ratios.append(budget / max(1e-6, r["t_cloud"]))
        lo = sorted(ratios)[len(ratios) // 3]
        hi = sorted(ratios)[2 * len(ratios) // 3]
        for r, x in zip(self.recs, ratios):
            r["jclass"] = (JobClass.INCIDENT if x <= lo
                           else JobClass.SIGNAL if x <= hi
                           else JobClass.FORECAST)

    # ---- trace playback ----

    def arrivals(self, t0, t1, origin_pool, scale=1.0):
        """Real (time, record) pairs mapped onto the simulated window."""
        base = self.recs[0]["arrival"]
        cap = getattr(self.cfg, "trace_max_jobs", 0)
        recs = self.recs
        if cap and cap < len(recs):
            step = len(recs) / cap          # even subsample, keeps the shape
            recs = [recs[int(i * step)] for i in range(cap)]
        out = []
        for r in recs:
            t = t0 + (r["arrival"] - base) * scale
            if t >= t1:
                break
            mult = 1.0
            for (b0, b1, m, _) in self._bursts:
                if b0 <= t <= b1:
                    mult *= m
            reps = int(mult) if mult > 1 else 1
            for _ in range(reps):
                out.append((t, r, origin_pool[self.rng.randrange(len(origin_pool))]))
        return out

    # ---- forecaster hooks, so MERSEM can plan against this workload ----

    def rate_at(self, t):
        """Jobs per second, measured from the trace itself."""
        if not self.recs:
            return 0.0
        span = max(1e-6, self.recs[-1]["arrival"] - self.recs[0]["arrival"])
        cap = getattr(self.cfg, "trace_max_jobs", 0)
        n = min(cap, len(self.recs)) if cap else len(self.recs)
        return n / (span * max(1e-6, getattr(self.cfg, "trace_scale", 1.0)))

    def class_weights(self, t):
        from .entities import JobClass
        c = [0, 0, 0]
        for r in self.recs:
            c[int(r["jclass"])] += 1
        tot = sum(c) or 1
        return [x / tot for x in c]

    def mean_deadline(self, jclass):
        """Mean deadline budget for a class, for planners that need one."""
        vals = [(r["deadline"] - r["arrival"]) for r in self.recs
                if int(r["jclass"]) == int(jclass)]
        scale = getattr(self.cfg, "deadline_scale", 1.0)
        return (sum(vals) / len(vals) * scale) if vals else 60.0

    def inject_burst(self, t_start, duration, mult=3.0, class_bias=None):
        self._bursts.append((t_start, t_start + duration, mult, class_bias))

    def clear_bursts(self):
        self._bursts.clear()

    # ---- job construction ----

    def make_job_from_record(self, rec, t_now, origin):
        from .entities import Job, Task
        jid = self._next_job
        self._next_job += 1
        budget = max(1e-3, (rec["deadline"] - rec["arrival"])
                    * getattr(self.cfg, "deadline_scale", 1.0))

        n_edges = max(1, rec["n_edges"])
        w_per_edge = rec["sum_edge_w"] / n_edges     # dataset units (KB)

        local, tmap = {}, {}
        for tid, t in rec["tasks"].items():
            task = Task(
                tid=self._next_task, job_id=jid, stage=f"s{tid % 8}",
                mi=t["mi"],
                in_bytes=max(1.0, len(t["preds"])) * w_per_edge * self.KB,
                out_bytes=max(1.0, len(t["succs"])) * w_per_edge * self.KB,
                memory_gb=max(0.05, t["ram_mb"] / 1024.0),
            )
            local[tid] = task
            self._next_task += 1
        for tid, t in rec["tasks"].items():
            local[tid].preds = tuple(local[p].tid for p in t["preds"] if p in local)
            local[tid].succs = tuple(local[s].tid for s in t["succs"] if s in local)
        for t in local.values():
            tmap[t.tid] = t

        job = Job(
            job_id=jid, origin_node=origin, jclass=rec["jclass"], arrival=t_now,
            deadline_abs=t_now + budget, tasks=tmap,
            entry=tuple(t.tid for t in tmap.values() if not t.preds),
            exits=tuple(t.tid for t in tmap.values() if not t.succs),
        )
        job.total_mi = sum(t.mi for t in tmap.values())
        self._annotate(job)
        return job

    def _annotate(self, job):
        """HEFT upward rank on the real DAG, using the dataset's cloud MIPS."""
        ref = self.mips.get("cloud", 2356000.0)
        order = self._topo(job)
        desc = {}
        for tid in reversed(order):
            t = job.tasks[tid]
            best, seen = 0.0, set()
            for s in t.succs:
                st = job.tasks[s]
                comm = t.out_bytes / self.cfg.ref_bw
                best = max(best, comm + st.upward_rank)
                seen.add(s)
                seen |= desc.get(s, set())
            t.upward_rank = t.mi / ref + best
            desc[tid] = seen
            t.n_descendants = len(seen)

    @staticmethod
    def _topo(job):
        indeg = {tid: len(t.preds) for tid, t in job.tasks.items()}
        q = [tid for tid, d in indeg.items() if d == 0]
        out = []
        while q:
            n = q.pop()
            out.append(n)
            for s in job.tasks[n].succs:
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
        return out
