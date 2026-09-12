'''
Student Name: Gbenga Dairo

Mobile Decision-Theoretic Bayesian Localization Strategy


Task:
TO extend the developed strategy by allowing the LocatorBot to move.
The agent uses recursive Bayesian filtering and entropy-based
decision control to decide whether to:
- move the LocatorBot,
- move the Roomba, or
- run the detector.

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
    '''Move one step toward target if possible.'''
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
    return random.choice(['N', 'S', 'E', 'W'])


# Probability models

def transition_probability(a: Coord, b: Coord, u: str, grid: List[List[int]]) -> float:
    '''Probability that the Roomba moves from a→b given action u.'''
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
    '''Compute sensor likelihood given beep outcome.'''
    dist = manhattan_distance(locator, cell)
    p_beep = math.exp(-alpha * (dist - 1))
    return p_beep if beep else 1 - p_beep


# Helper: Shannon Entropy

def entropy(belief: Dict[Coord, float]) -> float:
    '''
    Compute Shannon entropy.
    '''
    H = 0.0
    for p in belief.values():
        if p > 0:
            H -= p * math.log(p)
    return H


# Mobile Decision-Theoretic Strategy

def mobile_developed_strategy(
    grid: List[List[int]],
    locator_pos: Coord,
    true_roomba_pos: Coord,
    alpha: float = 0.3,
    max_steps: int = 200,
    w_mR: float = 1.0,  # Roomba move cost
    w_mL: float = 1.2,  # Locator move cost (slightly higher)
    w_s: float = 0.8,   # Sensing cost
    w_u: float = 8.0    # Uncertainty penalty
) -> Dict[str, float]:
    '''
    Perform recursive Bayesian localization with a mobile LocatorBot.
    '''
    opens = all_open_cells(grid)
    belief = {cell: 1/len(opens) for cell in opens}

    total_moves_R = 0
    total_moves_L = 0
    total_sense = 0
    utility = 0.0

    for step in range(max_steps):

        # Prediction (Roomba motion)
        max_cell = max(belief, key=belief.get)
        belief_certainty = belief[max_cell]
        if belief_certainty < 0.6:
            u = random.choice(['N','S','E','W'])
        else:
            u = choose_direction_toward(max_cell, locator_pos)

        predicted_belief = {cell: 0.0 for cell in opens}
        for b in opens:
            for a in opens:
                predicted_belief[b] += transition_probability(a, b, u, grid) * belief[a]
        belief = predicted_belief

        # Sensing (beep/no-beep update)
        dist_true = manhattan_distance(locator_pos, true_roomba_pos)
        p_beep = math.exp(-alpha * (dist_true - 1))
        heard_beep = random.random() < p_beep
        total_sense += 1

        for cell in opens:
            belief[cell] *= sensor_likelihood(heard_beep, locator_pos, cell, alpha)

        # Normalize
        norm = sum(max(0.0, v) for v in belief.values())
        if norm == 0:
            norm = 1e-9
        for cell in opens:
            belief[cell] = max(0.0, belief[cell]) / norm

        # Entropy-based decision for LocatorBot movement
        prior_entropy = entropy(belief)
        expected_entropy = 0.0

        # Approximate expected information gain
        for outcome in [True, False]:
            pseudo_belief = {c: belief[c] * sensor_likelihood(outcome, locator_pos, c, alpha)
                             for c in opens}
            norm = sum(max(0.0, v) for v in pseudo_belief.values())
            if norm == 0:
                norm = 1e-9
            for c in opens:
                pseudo_belief[c] = max(0.0, pseudo_belief[c]) / norm
            expected_entropy += 0.5 * entropy(pseudo_belief)  # assume equal outcome chance

        delta_H_sense = prior_entropy - expected_entropy

        # Evaluate expected entropy reduction if Locator moves one step toward belief peak
        new_locator = move_command(locator_pos, max_cell, grid)
        pseudo_belief = {c: belief[c] * sensor_likelihood(True, new_locator, c, alpha)
                         for c in opens}
        norm = sum(max(0.0, v) for v in pseudo_belief.values())
        if norm == 0:
            norm = 1e-9
        for c in opens:
            pseudo_belief[c] = max(0.0, pseudo_belief[c]) / norm
        delta_H_move = prior_entropy - entropy(pseudo_belief)

        # Compare expected utilities of sensing vs moving
        EU_sense = - (w_s + w_u * (1 - delta_H_sense))
        EU_moveL = - (w_mL + w_u * (1 - delta_H_move))

        if EU_moveL > EU_sense and new_locator != locator_pos:
            locator_pos = new_locator
            total_moves_L += 1
        else:
            # Roomba moves one step toward locator
            true_roomba_pos = move_command(true_roomba_pos, locator_pos, grid)
            total_moves_R += 1

        # Update utility
        belief_certainty = max(belief.values())
        utility = - (w_mR * total_moves_R + w_mL * total_moves_L +
                     w_s * total_sense + w_u * (1 - belief_certainty))

        # Termination conditions
        if true_roomba_pos == locator_pos:
            print(f"Localized at {locator_pos} after {total_moves_R} R-moves, "
                  f"{total_moves_L} L-moves, and {total_sense} senses. | U={utility:.3f}")
            break
        elif belief_certainty > 0.9:
            print(f"Belief collapsed to {max_cell} (P={belief_certainty:.2f}) | U={utility:.3f}")
            break

    return {
        'moves_R': total_moves_R,
        'moves_L': total_moves_L,
        'senses': total_sense,
        'total_actions': total_moves_R + total_moves_L + total_sense,
        'utility': utility
    }
