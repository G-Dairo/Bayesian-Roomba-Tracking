'''
Student name: Gbenga Dairo

Developed Strategy: Decision-Theoretic Bayesian Localization (Utility-Based HMM)

Task:
To extend Baseline Strategy 2 by performing full recursive Bayesian filtering with a
decomposable utility function. The agent reasons over uncertainty and balances
sensing and motion to minimize total actions.

'''

from __future__ import annotations
from typing import List, Tuple, Dict
import math, random

Coord = Tuple[int, int]

# Utility helpers

def manhattan_distance(a: Coord, b: Coord) -> int:
    '''Compute Manhattan distance between two cells.'''
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def all_open_cells(grid: List[List[int]]) -> List[Coord]:
    '''Return all open cells (grid value == 1).'''
    D = len(grid)
    return [(r, c) for r in range(D) for c in range(D) if grid[r][c] == 1]

def move_command(cell: Coord, target: Coord, grid: List[List[int]]) -> Coord:
    '''Move the Roomba one step toward the LocatorBot, respecting walls.'''
    r, c = cell
    rt, ct = target
    new_r, new_c = r, c

    if r < rt and grid[r+1][c] == 1:
        new_r += 1
    elif r > rt and grid[r-1][c] == 1:
        new_r -= 1
    elif c < ct and grid[r][c+1] == 1:
        new_c += 1
    elif c > ct and grid[r][c-1] == 1:
        new_c -= 1
    return (new_r, new_c)

def choose_direction_toward(a: Coord, b: Coord) -> str:
    '''Return N/S/E/W direction from cell a toward cell b.'''
    ra, ca = a
    rb, cb = b
    if ra < rb: return 'S'
    if ra > rb: return 'N'
    if ca < cb: return 'E'
    if ca > cb: return 'W'
    return random.choice(['N','S','E','W'])


# Transition and Sensor models

def transition_probability(a: Coord, b: Coord, u: str, grid: List[List[int]]) -> float:
    '''Probability that the Roomba moves from cell a to cell b after executing command u.'''
    D = len(grid)
    dr, dc = {'N': (-1,0), 'S': (1,0), 'E': (0,1), 'W': (0,-1)}[u]
    intended = (a[0]+dr, a[1]+dc)
    if not (0 <= intended[0] < D and 0 <= intended[1] < D and grid[intended[0]][intended[1]] == 1):
        intended = a
    if b == intended:
        return 0.8
    elif b == a:
        return 0.2
    else:
        return 0.0

def sensor_likelihood(beep: bool, locator: Coord, cell: Coord, alpha: float) -> float:
    '''Compute sensor likelihood under exponential decay model.'''
    dist = manhattan_distance(locator, cell)
    p_beep = math.exp(-alpha * (dist - 1))
    return p_beep if beep else 1 - p_beep


# Developed Strategy main loop

def developed_strategy(
    grid: List[List[int]],
    locator_pos: Coord,
    true_roomba_pos: Coord,
    alpha: float = 0.3,
    max_steps: int = 200,
    w_m: float = 1.0,   # weight for move cost
    w_s: float = 1.0,   # weight for sensing cost
    w_u: float = 10.0   # weight for uncertainty penalty
) -> Dict[str, int]:
    '''
    Perform recursive Bayesian localization (Prediction + Correction + Utility).
    Returns counts of moves, senses, and total actions.
    '''
    opens = all_open_cells(grid)
    belief: Dict[Coord, float] = {cell: 1/len(opens) for cell in opens}

    total_moves = 0
    total_sense = 0
    utility = 0.0

    for step in range(max_steps):

        ''' Prediction (motion update using transition model)'''
        max_cell = max(belief, key=belief.get)
        belief_certainty = belief[max_cell]
        # Exploration bias when uncertainty is high
        if belief_certainty < 0.6:
            u = random.choice(['N','S','E','W'])
        else:
            u = choose_direction_toward(max_cell, locator_pos)

        predicted_belief = {cell: 0.0 for cell in opens}
        for b in opens:
            for a in opens:
                predicted_belief[b] += transition_probability(a, b, u, grid) * belief[a]
        belief = predicted_belief

        ''' Correction (sensor update based on new reading)'''
        dist_true = manhattan_distance(locator_pos, true_roomba_pos)
        p_beep = math.exp(-alpha * (dist_true - 1))
        heard_beep = random.random() < p_beep
        total_sense += 1

        for cell in opens:
            belief[cell] *= sensor_likelihood(heard_beep, locator_pos, cell, alpha)

        # Normalize
        norm = sum(belief.values())
        if norm == 0:
            norm = 1e-9
        for cell in opens:
            belief[cell] /= norm

        ''' Compute clamped certainty and utility'''
        belief_certainty = max(belief.values())
        belief_certainty = min(max(belief_certainty, 0.0), 1.0)
        utility = - (w_m * total_moves + w_s * total_sense + w_u * (1 - belief_certainty))

        ''' Simulate true Roomba motion'''
        # Deterministic motion toward locator for fairness
        true_roomba_pos = move_command(true_roomba_pos, locator_pos, grid)
        total_moves += 1

        ''' Termination conditions'''
        if true_roomba_pos == locator_pos:
            print(f"Localized at {locator_pos} after {total_moves} moves and {total_sense} senses. | Utility: {utility:.3f}")
            break

        elif belief_certainty > 0.9:
            print(f"Belief collapsed to {max_cell} (P={belief_certainty:.2f}) | Utility: {utility:.3f}")
            break

    return {
        'moves': total_moves,
        'senses': total_sense,
        'total_actions': total_moves + total_sense,
        'utility': utility
    }
