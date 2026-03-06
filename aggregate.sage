# Assumptions about the log format:
# - Directory: ./lats/
# - Each file contains 1 Python dict per line (ast.literal_eval-able)
# - Lines starting with '#' are comments
# - Each dict has at least:
#     'algorithm' (e.g. 'HLLL', 'L2-Cholesky')
#     'success' (True/False)
#     'timings_s' (dict with numeric values; we use sum(timings_s.values()) as "runtime")
#     'prec_bits' (int)
#     'float' (e.g. 'd', 'dd', 'mpfr')
#     'n', 'm' (optional; if not present we parse n,m from filename q_n_m.txt)

import os
import re
import ast
from collections import defaultdict

from sage.all import Graphics, point, line, text, rainbow, log
# You can also use matplotlib if you prefer; Sage Graphics is convenient.

LATS_DIR = "lats"

# parse filenames like: lats/{q}_{n}_{m}.txt where q may be huge decimal
FNAME_RE = re.compile(r"^(?P<q>\d+)_(?P<n>\d+)_(?P<m>\d+)\.txt$")


def scan_lat_files(lats_dir=LATS_DIR):
    paths = []
    if not os.path.isdir(lats_dir):
        raise RuntimeError(f"Directory not found: {lats_dir}")
    for fn in os.listdir(lats_dir):
        p = os.path.join(lats_dir, fn)
        if os.path.isfile(p) and fn.endswith(".txt") and FNAME_RE.match(fn):
            paths.append(p)
    return sorted(paths)


def read_success_rows(path):
    """
    Returns list of (row_dict, n_from_fname, m_from_fname, q_from_fname)
    but keeps only successful runs.
    """
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
    if not m:
        return []
    q_s = m.group("q")
    n_f = int(m.group("n"))
    m_f = int(m.group("m"))

    out = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                d = ast.literal_eval(ln)
            except Exception:
                # skip malformed lines
                continue
            if not isinstance(d, dict):
                continue
            if d.get("success", False) is not True:
                continue
            out.append((d, n_f, m_f, q_s))
    return out

def read_fail_rows(path):
    """
    Returns list of (row_dict, n_from_fname, m_from_fname, q_from_fname)
    but keeps only failed runs.
    """
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
    if not m:
        return []
    q_s = m.group("q")
    n_f = int(m.group("n"))
    m_f = int(m.group("m"))

    out = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                d = ast.literal_eval(ln)
            except Exception:
                # skip malformed lines
                continue
            if not isinstance(d, dict):
                continue
            if d.get("success", False) is True:
                continue
            out.append((d, n_f, m_f, q_s))
    return out

def runtime_from_row(d):
    """
    Defines "runtime" as sum of all timings in d['timings_s'].
    This automatically includes:
      - 'gso' if present
      - 'householder_R' if present
      - 'alg' time
    """
    t = d.get("timings_s", {})
    if not isinstance(t, dict):
        return None
    vals = []
    for v in t.values():
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals)


def key_from_row(d):
    """
    Grouping key: (precision-label, algorithm, mod-q)
    precision-label includes float mode + prec bits, e.g. "d-53", "mpfr-256".
    """
    alg = d.get("algorithm", "unknown")
    fmode = d.get("float", "unknown")
    pbits = d.get("prec_bits", None)
    q = d.get("q",None)
    if isinstance(pbits, int):
        prec = f"{fmode}-{pbits}"
    else:
        prec = f"{fmode}"
    return prec, alg, q


def main():
    paths = scan_lat_files(LATS_DIR)
    if not paths:
        raise RuntimeError(f"No lattice log files found in ./{LATS_DIR}/")

    # Collect successful points grouped by (prec, alg)
    # store list of (n, runtime)
    groups = defaultdict(list)

    for p in paths:
        rows = read_success_rows(p)
        for (d, n_f, m_f, q_s) in rows:
            n = int(d.get("n", n_f))
            rt = runtime_from_row(d)
            if rt is None:
                continue
            key = key_from_row(d)
            groups[key].append((n, rt))

    if not groups:
        raise RuntimeError("No successful runs found in any file.")

    # For each (prec, alg), aggregate by dimension and plot mean runtime vs n
    plots = []
    keys = sorted(groups.keys())
    colors = rainbow(len(keys))

    for idx, key in enumerate(keys):
        prec, alg, q = key
        pts = groups[key]

        # aggregate by n -> mean runtime
        by_n = defaultdict(list)
        for n, rt in pts:
            by_n[n].append(rt)

        ns = sorted(by_n.keys())
        mean_pts = [(n, sum(by_n[n]) / len(by_n[n])) for n in ns]

        g = Graphics()
        g += point(mean_pts, color=colors[idx], size=25, legend_label=f"{alg} @ {prec} | {q}")
        g += line(mean_pts, color=colors[idx], thickness=1.5)
        plots.append(g)

    # Combine plots
    G = sum(plots, Graphics())

    # Labels and style
    G.axes_labels(["dimension n", "runtime (seconds)"])
    G.set_legend_options(loc="upper left", fontsize="small")
    G.show(title="Runtime vs dimension (successful runs only)")

    # Save to file as well
    out_png = os.path.join(LATS_DIR, "runtime_vs_dimension.png")
    G.save(out_png)
    print(f"Saved plot to: {out_png}")

    # Optional: also dump a small summary table
    print("\nSummary (successful points):")
    for key in keys:
        prec, alg, q = key
        pts = groups[key]
        ns = sorted(set(n for n, _ in pts))
        print(f"  {alg} @ {prec} | q: {len(pts)} points, n in {ns[:5]}{'...' if len(ns)>5 else ''}")

if __name__ == "__main__":
    main()