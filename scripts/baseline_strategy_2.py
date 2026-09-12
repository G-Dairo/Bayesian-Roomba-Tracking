'''
Student Name: Gbenga Dairo

Baseline Strategy 2: Probabilistic Localization with Detector

Task:
Use a simple Bayesian belief update combined with a "greedy" action rule:
always move as if the Roomba were at the cell with the highest posterior probability.

'''

from __future__ import annotations
from typing import List, Tuple, Dict
import math, random

Coord = Tuple[int, int]


# Utility functions

def manhattan_distance(a: Coord, b: Coord) -> int:
    '''Return the Manhattan distance between two grid cells.'''
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def all_open_cells(grid: List[List[int]]) -> List[Coord]:
    '''Return all open cells where grid value == 1.'''
    D = len(grid)
    opens = []
    for r in range(D):
        for c in range(D):
            if grid[r][c] == 1:
                opens.append((r, c))
    return opens


def move_command(cell: Coord, target: Coord, grid: List[List[int]]) -> Coord:
    '''
    Move one step toward the target (LocatorBot).
    If the direction is blocked by a wall, the bot stays in place.
    '''
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


# Sensor model

def sensor_likelihood(beep: bool, locator: Coord, cell: Coord, alpha: float) -> float:
    '''
    Compute P(beep | locator=a, roomba=b) or P(no beep | a,b)
    using the exponential decay model of detector sensitivity.
    '''
    dist = manhattan_distance(locator, cell)
    p_beep = math.exp(-alpha * (dist - 1))
    if beep:
        return p_beep
    else:
        return 1 - p_beep



# Baseline Strategy 2 main algorithm

def baseline_strategy_2(
    grid: List[List[int]],
    locator_pos: Coord,
    true_roomba_pos: Coord,
    alpha: float = 0.3,
    max_steps: int = 200
) -> Dict[str, int]:
    '''
    Simulate the localization process under Baseline Strategy 2

    Returns a dictionary with total moves, senses, and total actions.
    '''

    # Initialize uniform belief over all open cells
    opens = all_open_cells(grid)
    belief: Dict[Coord, float] = {cell: 1 / len(opens) for cell in opens}

    total_moves = 0
    total_sense = 0

    # Main loop
    for step in range(max_steps):

        # Detector reading (simulate sensor)
        dist_true = manhattan_distance(locator_pos, true_roomba_pos)
        p_beep = math.exp(-alpha * (dist_true - 1))
        heard_beep = random.random() < p_beep
        total_sense += 1

        
        # Sensor update (Bayesian correction)
        new_belief = {}
        total_prob = 0.0
        for cell in opens:
            likelihood = sensor_likelihood(heard_beep, locator_pos, cell, alpha)
            new_belief[cell] = likelihood * belief[cell]
            total_prob += new_belief[cell]

        # Normalize belief distribution
        if total_prob == 0:
            total_prob = 1e-9  # avoid divide-by-zero errors
        for cell in opens:
            new_belief[cell] /= total_prob
        belief = new_belief

        # Identify the most probable cell
        max_cell = max(belief, key=belief.get)
        max_prob = belief[max_cell]

        # Issue a movement command (based on belief)
        # Move the commanded cell toward the LocatorBot
        intended_move = move_command(max_cell, locator_pos, grid)

        # The true Roomba follows the same command (may differ due to walls)
        true_roomba_pos = move_command(true_roomba_pos, locator_pos, grid)
        total_moves += 1


        # Termination conditions
        if true_roomba_pos == locator_pos:
            print(f"Roomba localized at {locator_pos} after {total_moves} moves and {total_sense} senses.")
            break

        elif max_prob > 0.9:   # belief collapse threshold
            print(f"Belief collapsed to {max_cell} with P={max_prob:.2f}")
            break

    total_actions = total_moves + total_sense

    return {
        "moves": total_moves,
        "senses": total_sense,
        "total_actions": total_actions
    }
