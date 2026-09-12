'''
Student Name: Gbenga Dairo

Optimal Strategy: A* in Belief Space

The functions below return the following:
1. path to localization
2. Total moves to localize
3. Nodes expanded
4. Peak frontier 
5. Runtime, memory usage, and effective branching factor (b*)

'''

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import heapq, time, tracemalloc

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

Coord = Tuple[int, int]
Belief = frozenset[Coord]


#Heuristic: relaxed problem (span-based)

def heuristic_span_max(belief: Belief) -> int:
    if len(belief) <= 1:
        return 0
    rows = [r for r, _ in belief]
    cols = [c for _, c in belief]
    return max(max(rows) - min(rows), max(cols) - min(cols))


# Belief Initialization & Goal

#Initial belief over all open cells in the grid
def _all_open_belief(grid: List[List[int]]) -> Belief:
    D = len(grid)
    return frozenset((r, c) for r in range(D) for c in range(D) if grid[r][c] == 1)

#Goal test: belief has collpsed to a single cell
def _goal(belief: Belief) -> bool:
    return len(belief) == 1


# Priority Queue Node
@dataclass(order=True)
class _PQItem:
    f: int
    h: int
    belief_size: int
    path_len: int
    tie: int
    g: int
    belief: Belief
    path: Tuple[str, ...]



# Effective Branching Factor (b*)
'''
This help solve for effective branching factor b* such that:
N + 1 = 1 + b + b^2 + ... + b^d = (b^(d+1) - 1)/(b - 1).
    
'''

def _effective_b_star(N: int, d: int, tol: float = 1e-6) -> float:
    if d <= 0:
        return 1.0

    def series(b: float) -> float:
        if abs(b - 1.0) < 1e-12:
            return d + 1.0
        return (b ** (d + 1) - 1.0) / (b - 1.0)

    target = N + 1.0
    lo, hi = 1.0, max(2.0, 1.0 + (N / max(d, 1)))
    while series(hi) < target:
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if series(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)



# A* Search over Belief Space
def _a_star_core(
    grid: List[List[int]],
    skip_noop: bool = True,
    heuristic = heuristic_span_max,
    initial_belief: Optional[Belief] = None,   
) -> Tuple[List[str], int, int]:
    D = len(grid)

    # Precompute single-cell transitions (bounce if blocked)
    move_vec = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
    succ_cell: Dict[str, Dict[Coord, Coord]] = {mv: {} for mv in move_vec}
    for r in range(D):
        for c in range(D):
            if grid[r][c] != 1:
                continue
            for mv, (dr, dc) in move_vec.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < D and 0 <= nc < D and grid[nr][nc] == 1:
                    succ_cell[mv][(r, c)] = (nr, nc)
                else:
                    succ_cell[mv][(r, c)] = (r, c)

    # Transition and heuristic caches
    trans_cache: Dict[Tuple[Belief, str], Belief] = {}
    h_cache: Dict[Belief, int] = {}

    # Initial node setup 
    B0 = initial_belief if (initial_belief is not None) else _all_open_belief(grid)
    h0 = heuristic(B0)
    node0 = _PQItem(f=h0, h=h0, belief_size=len(B0), path_len=0, tie=0, g=0, belief=B0, path=tuple())
    frontier: List[_PQItem] = [node0]
    heapq.heapify(frontier)
    best_g: Dict[int, int] = {hash(B0): 0}
    nodes_expanded = 0
    peak_frontier = 1
    tie = 1

    # Main A* loop
    while frontier:
        item = heapq.heappop(frontier)
        B, g, path = item.belief, item.g, item.path
        nodes_expanded += 1

        if _goal(B):
            return list(path), nodes_expanded, peak_frontier

        # Order actions by which dimension has larger spread
        rows = [r for r, _ in B]
        cols = [c for _, c in B]
        span_r = (max(rows) - min(rows)) if rows else 0
        span_c = (max(cols) - min(cols)) if cols else 0
        order = ("N", "S", "E", "W") if span_r >= span_c else ("E", "W", "N", "S")

        for mv in order:
            # Cached transition
            key_t = (B, mv)
            Bp = trans_cache.get(key_t)
            if Bp is None:
                Bp = frozenset({succ_cell[mv][rc] for rc in B})
                trans_cache[key_t] = Bp

            if skip_noop and Bp == B:
                continue

            gp = g + 1
            key = hash(Bp)
            if key in best_g and gp >= best_g[key]:
                continue
            best_g[key] = gp

            # Cached heuristic
            hp = h_cache.get(Bp)
            if hp is None:
                hp = heuristic(Bp)
                h_cache[Bp] = hp

            fp = gp + hp
            heapq.heappush(frontier, _PQItem(
                f=fp, h=hp, belief_size=len(Bp), path_len=len(path)+1,
                tie=tie, g=gp, belief=Bp, path=path+(mv,)
            ))
            tie += 1
            peak_frontier = max(peak_frontier, len(frontier))

    return [], nodes_expanded, peak_frontier


# Runtime and Memory 
def a_star_belief_stats(
    grid: List[List[int]],
    use_rss: bool = True,
    initial_belief: Optional[Belief] = None,    
):
    #This help run A* on the belief grid, with runtime & memory stats.
    t0 = time.perf_counter()
    tracemalloc.start()

    path, N, peak = _a_star_core(grid, initial_belief=initial_belief)

    current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()

    rss1 = psutil.Process().memory_info().rss if (_HAS_PSUTIL and use_rss) else None
    runtime_sec = t1 - t0
    peak_py_mem_mb = peak_bytes / (1024**2)
    rss_mb = (rss1 / (1024**2)) if rss1 is not None else None

    return path, N, peak, runtime_sec, peak_py_mem_mb, rss_mb


#This help return a full report
def a_star_output(
    grid: List[List[int]],
    initial_belief: Optional[Belief] = None,     # <-- NEW
) -> Dict[str, object]:
    path, N, peak, runtime, py_mb, rss_mb = a_star_belief_stats(
        grid, use_rss=True, initial_belief=initial_belief
    )
    d = len(path)
    b_star = _effective_b_star(N, d)

    return {
        "path": path,
        "moves": d,
        "nodes_expanded": N,
        "peak_frontier": peak,
        "runtime_sec": runtime,
        "peak_py_mem_mb": py_mb,
        "rss_mb": rss_mb,
        "b_star": b_star,
    }
