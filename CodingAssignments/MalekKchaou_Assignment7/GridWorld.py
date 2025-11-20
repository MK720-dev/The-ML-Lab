# -*- coding: utf-8 -*-
"""
================================================================================
Name of Program: EECS 658 Assignment 7 – Gridworld Policy and Value Iteration

Brief Description:
    This program implements Model-Based Reinforcement Learning algorithms on a 
    5x5 Gridworld environment with two terminal states. The assignment is divided
    into two parts:

    Part 1: Policy Iteration
        - Starts with an initial randomized policy
        - Uses iterative policy evaluation (Bellman backups)
        - Uses policy improvement to create a greedy policy w.r.t. V
        - Prints V at iterations 0, 1, 10, and final
        - Produces a policy grid and plots convergence error vs iteration

    Part 2: Value Iteration
        - Uses Bellman optimality updates
        - Prints V at each iteration, including the final optimal V*
        - Extracts the optimal policy π*
        - Plots |V_k - V_{k-1}| vs iteration

Inputs:
    - No external input files required.
      The Gridworld is defined internally as a 5x5 state space.

Outputs:
    - Printed value grids for all major iterations
    - Printed optimal policy grids (U/D/L/R/T)
    - Convergence plots (saved locally)

Author: Malek Kchaou
Collaborators: None 
Other sources: 
    - https://gibberblot.github.io/rl-notes/index.html
    - Prof. David O. Johnson, EECS 658 Lecture Slides, University of Kansas (Fall 2025).
    - NumPy documentation: https://numpy.org/doc/

Creation Date: November 18, 2025
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# Gridworld Setup
# ==============================================================================

# Build a 5×5 grid where each state is represented by an integer ID (0–24)
grid = np.array([[(j + 5*i) for j in range(5)] for i in range(5)])

# Terminal states: start (0) and goal (24)
terminal_states = [0, 24]

# Action set (deterministic)
actions = ['up', 'down', 'left', 'right']

# RL Parameters
gamma = 1.0          # discount factor (episodic environment, so γ=1 is fine)
theta = 1e-4         # convergence tolerance for value updates



# ==============================================================================
# next_state(): Transition Function
# ==============================================================================

def next_state(s, a):
    """
    Deterministic Markov transition model.
    Takes state index s and action string a, returns integer ID of next state.
    """
    row, col = divmod(s, 5)   # convert state index → (row, col)

    # Apply movement according to chosen action
    if a == 'up':
        row = max(row - 1, 0)     # prevent leaving grid
    elif a == 'down':
        row = min(row + 1, 4)
    elif a == 'left':
        col = max(col - 1, 0)
    elif a == 'right':
        col = min(col + 1, 4)

    # Convert updated grid location back to state ID
    return row * 5 + col



# ==============================================================================
# reward(): Reward Function
# ==============================================================================

def reward(s, a, s_next):
    """
    Reward model:
        - Terminal states have zero reward.
        - All other transitions cost -1 per step.
    """
    if s in terminal_states:
        return 0
    return -1



# ==============================================================================
#                            PART 1 — POLICY ITERATION
# ==============================================================================

print("\n==============================")
print("        PART 1: POLICY ITERATION")
print("==============================\n")



# ==============================================================================
# policy_evaluation(): Evaluate current policy π
# ==============================================================================

def policy_evaluation(policy, V_init, theta=1e-4):
    """
    Performs iterative policy evaluation.
    Computes V^π by repeatedly applying Bellman EXPECTATION backup until change
    in value (delta) is less than theta, OR until safety cap of 100 iterations.
    """
    V = V_init.copy()      # start from initial value function
    errors = []            # track delta per iteration
    it = 0                 # iteration counter

    print(f"Policy Evaluation - iteration {it}:")
    print(V, "\n")

    while True:
        delta = 0.0                # track max difference across all states
        V_new = V.copy()           # new value function buffer
        it += 1

        # Loop through all states in the 5x5 grid
        for i in range(5):
            for j in range(5):
                s = i*5 + j        # convert (row, col) → state ID

                # Terminal states always stay zero
                if s in terminal_states:
                    V_new[i, j] = 0.0
                    continue

                # Read action chosen by current policy π(s)
                a = policy[i, j]

                # Compute next state according to deterministic dynamics
                s_next = next_state(s, a)

                # Reward from taking action a at state s
                r = reward(s, a, s_next)

                # Extract coordinates of successor state
                row_next, col_next = divmod(s_next, 5)

                # Bellman expectation backup for POLICY EVALUATION:
                #     V_new(s) = R + γ·V(s_next)
                v_new = r + gamma * V[row_next, col_next]

                # Track largest update difference
                delta = max(delta, abs(v_new - V[i, j]))

                # Store updated state value
                V_new[i, j] = v_new

        # Replace value function with updated version
        V = V_new.copy()

        # Save error for convergence plot
        errors.append(delta)

        # Required printouts
        if it == 1 or it == 10:
            print(f"Policy Evaluation - iteration {it}:")
            print(V, "\n")

        # Convergence condition met
        if delta < theta:
            print(f"Policy Evaluation - FINAL iteration {it}:")
            print(V, "\n")
            break

        # Safety cutoff — prevents infinite loops for bad policies
        if it == 100:
            print(f"Stopping after {it} iterations (policy failed to converge).")
            print(V, "\n")
            break

    return V, errors



# ==============================================================================
# policy_improvement(): Improve current policy greedily
# ==============================================================================

def policy_improvement(V, policy):
    """
    Given a value function V, improve the policy by choosing the action that
    maximizes expected return at each state (Bellman optimality).
    """
    policy_stable = True
    new_policy = policy.copy()

    # Loop through all grid states
    for i in range(5):
        for j in range(5):
            s = i*5 + j

            if s in terminal_states:
                new_policy[i, j] = 'T'
                continue

            old_action = policy[i, j]

            best_val = -1e9
            best_action = None

            # Evaluate all possible actions a ∈ A
            for a in actions:
                s_next = next_state(s, a)
                r = reward(s, a, s_next)
                row_next, col_next = divmod(s_next, 5)

                # Standard Bellman optimality for checking action quality
                val = r + gamma * V[row_next, col_next]

                # Keep best action so far
                if val > best_val:
                    best_val = val
                    best_action = a

            # Assign greedy action
            new_policy[i, j] = best_action

            # Detect change in policy
            if best_action != old_action:
                policy_stable = False

    return new_policy, policy_stable



# ==============================================================================
# run_policy_iteration(): Full algorithm for Policy Iteration
# ==============================================================================

def run_policy_iteration():
    """
    Runs Policy Iteration by alternating:
        1. Policy Evaluation
        2. Policy Improvement
    until the policy becomes stable (no changes).
    """
    # Start from RANDOM policy for non-terminals
    policy = np.random.choice(actions, size=(5,5))

    # Override terminal states with 'T'
    for s in terminal_states:
        i, j = divmod(s, 5)
        policy[i, j] = 'T'

    # Initialize state-values to zero
    V = np.zeros((5, 5))
    all_errors = []
    outer_iter = 0

    # Repeatedly evaluate and improve policy until stable
    while True:
        outer_iter += 1

        print(f"=== Policy Iteration Outer Loop {outer_iter} ===")
        print("Current policy:")
        print(policy, "\n")

        # Evaluate V^π
        V, eval_errors = policy_evaluation(policy, V, theta)
        all_errors.extend(eval_errors)

        # Improve π using greedy update
        policy, stable = policy_improvement(V, policy)
        print("Improved policy:")
        print(policy, "\n")

        # Stop if no changes (policy converged)
        if stable:
            print("Policy is stable — stopping policy iteration.\n")
            break

    return V, policy, all_errors



# Run Part 1 — (Policy Iteration)
V_pi, policy_pi, errors_pi = run_policy_iteration()

print("Optimal Value Function from Policy Iteration:")
print(V_pi, "\n")

# Convert policy to friendly symbols (U/D/L/R/T)
policy_symbols = np.full((5,5), '', dtype=object)
for i in range(5):
    for j in range(5):
        a = policy_pi[i, j]
        policy_symbols[i, j] = 'T' if a == 'T' else a[0].upper()

print("Optimal Policy (U/D/L/R/T):")
print(policy_symbols)

# Convergence plot for Policy Iteration
plt.figure(figsize=(8,5))
plt.plot(errors_pi, linewidth=2)
plt.axhline(theta, color='red', linestyle='--', label=f'ε = {theta}')
plt.xlabel("Iteration (t)")
plt.ylabel("Max Error |V_k - V_{k-1}|")
plt.title("Policy Iteration – Convergence of Policy Evaluation")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()



# ==============================================================================
#                           PART 2 — VALUE ITERATION
# ==============================================================================

print("\n==============================")
print("        PART 2: VALUE ITERATION")
print("==============================\n")



# ==============================================================================
# run_value_iteration(): Value Iteration Algorithm
# ==============================================================================

def run_value_iteration():
    """
    Runs Value Iteration using the Bellman optimality update:
        V(s) ← max_a [ R + γ·V(s') ]
    Continues until change in value is less than theta.
    """
    V = np.zeros_like(grid)     # initial value function
    errors = []
    iteration = 0

    print(f"Iteration {iteration}:")
    print(V, "\n")

    while True:
        delta = 0                 # track largest update
        V_new = V.copy()
        iteration += 1

        # Loop over all grid cells
        for i in range(5):
            for j in range(5):
                s = i*5 + j

                if s in terminal_states:
                    V_new[i,j] = 0
                    continue

                # For VALUE ITERATION we evaluate ALL actions
                values = []

                for a in actions:
                    s_next = next_state(s, a)
                    row_next, col_next = divmod(s_next, 5)
                    values.append(reward(s, a, s_next) + gamma * V[row_next, col_next])

                # Apply Bellman optimality (take max over actions)
                V_new[i,j] = max(values)

                # Track convergence error
                delta = max(delta, abs(V_new[i,j] - V[i,j]))

        V = V_new.copy()
        errors.append(delta)

        # Print intermediate results every iteration
        print(f"Iteration {iteration}:")
        print(V, "\n")

        # Stop once updates are small
        if delta < theta:
            break

    return V, errors



# Run Value Iteration
V_vi, errors_vi = run_value_iteration()



# ==============================================================================
# extract_policy(): Greedy policy derived from V*
# ==============================================================================

def extract_policy(V):
    """
    Extract the greedy optimal policy derived from the final V produced by VI.
    """
    policy = np.full((5,5), '', dtype=object)

    for i in range(5):
        for j in range(5):
            s = i*5 + j

            if s in terminal_states:
                policy[i,j] = 'T'
                continue

            best_val = -1e9
            best_action = None

            # Evaluate all actions
            for a in actions:
                s_next = next_state(s, a)
                row_next, col_next = divmod(s_next, 5)
                val = reward(s, a, s_next) + gamma * V[row_next, col_next]

                if val > best_val:
                    best_val = val
                    best_action = a

            # Convert to symbolic representation
            policy[i,j] = best_action[0].upper()

    return policy



# Display the optimal policy extracted from Value Iteration
print("Optimal Policy (from Value Iteration):")
print(extract_policy(V_vi), "\n")

# Plot convergence curve for Value Iteration
plt.figure(figsize=(8,5))
plt.plot(errors_vi, linewidth=2)
plt.axhline(theta, color='red', linestyle='--', label=f'ε = {theta}')
plt.xlabel("Iteration")
plt.ylabel("Max Error |V_k - V_{k-1}|")
plt.title("Value Iteration Convergence")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

