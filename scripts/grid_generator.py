'''
Student Name: Gbenga Dairo

This creates a D × D grid of blocked cells, opens a random interior seed,
and iteratively opens new cells.

'''

from __future__ import annotations
import random
from typing import List, Tuple

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

Coord = Tuple[int, int]

def generate_ship(D: int, seed: int | None = None) -> List[List[int]]:
   
    if seed is not None:
        random.seed(seed)

    grid = [[0 for _ in range(D)] for _ in range(D)]

#Helper to find cardinal neighbors
    def neighbors(r: int, c: int) -> List[Coord]:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [
            (r + dr, c + dc)
            for dr, dc in dirs
            if 0 <= r + dr < D and 0 <= c + dc < D
        ]

#Step 1: choose random interior seed
    r, c = random.randint(1, D - 2), random.randint(1, D - 2)
    grid[r][c] = 1

#Step 2: iteratively open cells with exactly one open neighbor
    changed = True
    while changed:
        changed = False
        candidates = []
        for i in range(1, D - 1):
            for j in range(1, D - 1):
                if grid[i][j] == 0:
                    open_neighbors = sum(grid[nr][nc] for nr, nc in neighbors(i, j))
                    if open_neighbors == 1:
                        candidates.append((i, j))
        if candidates:
            x, y = random.choice(candidates)
            grid[x][y] = 1
            changed = True

#Step 3: identify dead ends
    dead_ends = []
    for i in range(1, D - 1):
        for j in range(1, D - 1):
            if grid[i][j] == 1:
                if sum(grid[nr][nc] for nr, nc in neighbors(i, j)) == 1:
                    dead_ends.append((i, j))

#Step 4: open some extra cells near dead ends
    for (i, j) in random.sample(dead_ends, k=len(dead_ends) // 2):
        closed_neighbors = [(nr, nc) for nr, nc in neighbors(i, j) if grid[nr][nc] == 0]
        if closed_neighbors:
            nr, nc = random.choice(closed_neighbors)
            grid[nr][nc] = 1

    return grid

#Text visualization of the ship grid.
def display_ship(grid: List[List[int]]) -> None:
    for row in grid:
        print("".join("⬜" if cell else "⬛" for cell in row))
 
