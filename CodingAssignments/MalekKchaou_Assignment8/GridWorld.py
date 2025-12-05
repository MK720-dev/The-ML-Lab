# -*- coding: utf-8 -*-

"""
EECS 658 - Assignment 8
Introduction to Machine Learning
Gridworld Monte Carlo & Temporal-Difference Control

Prologue:
- Program: EECS 658 Assignment 8
- Description:
    Implements Monte Carlo (First-Visit, Every-Visit, On-Policy-Plus),
    Q-Learning, SARSA, and Decaying Epsilon-Greedy algorithms on the
    modified Gridworld task, and plots error vs. t and cumulative
    average rewards across methods.
- Inputs: None (all hyperparameters and environment are defined in code)
- Outputs (printed to console):
    * For Parts 1–3: N(s), S(s), V(s) at epoch 0,1,10,final
    * For Parts 1–3: episode tables for k, s, r, gamma, G(s) for epoch 1,10,final
    * For Parts 4–6: R and Q matrices at episode 0,1,10,final
    * For Parts 4–6: optimal path from state 7 to terminal state 1
    * For Part 7: plots of cumulative average rewards for Parts 4–6
    * Plots of error vs. t for Parts 1–6 with epsilon threshold marked
- Collaborators: None
- Other sources: (e.g., ChatGPT, StackOverflow, lecture pseudocode, etc.)
- Author: Malek Kchaou
- Creation date: 12-2-2025

NOTE:
- This is a teaching-oriented reference implementation.
- You must add more fine-grained comments to satisfy the rubric
  (comments for every line or block “in your own words”).
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Common configuration
# -----------------------------

GAMMA = 0.9
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Helper: Grid definition and mappings (used by both MC & TD)
# ============================================================

# Grid layout with irregular rows exactly as in assignment
GRID_ROWS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23],
]

ALL_STATES = [s for row in GRID_ROWS for s in row]
NUM_STATES = max(ALL_STATES)        # 23
TERMINAL_STATES = [1, 23]           # assumption based on diagram
NON_TERMINAL_STATES = [s for s in ALL_STATES if s not in TERMINAL_STATES]

# Map: state -> (row_idx, col_idx) and back
STATE_TO_POS = {}
POS_TO_STATE = {}
for r_idx, row in enumerate(GRID_ROWS):
    for c_idx, s in enumerate(row):
        STATE_TO_POS[s] = (r_idx, c_idx)
        POS_TO_STATE[(r_idx, c_idx)] = s

# Directions for MC environment
ACTIONS_MC = ["UP", "DOWN", "LEFT", "RIGHT"]


# ============================================================
# Environment for Monte Carlo (wall hits allowed)
# ============================================================

class GridworldMC:
    """
    Gridworld environment for Monte Carlo parts.
    Actions: UP/DOWN/LEFT/RIGHT.
    - Hitting a wall: stay in place, reward = -1.
    - Moving to non-terminal: reward = -1.
    - Moving into terminal: reward = 0, and episode ends.
    """

    def __init__(self):
        self.gamma = GAMMA
        self.terminal_states = set(TERMINAL_STATES)

    def is_terminal(self, s):
        return s in self.terminal_states

    def reset(self):
        # Start from a random non-terminal state
        return random.choice(NON_TERMINAL_STATES)

    def step(self, state, action):
        """
        Perform one step given a state (1..23) and an action in ACTIONS_MC.
        Returns: next_state, reward, done
        """
        if self.is_terminal(state):
            # Episode should already be ended; stay in terminal.
            return state, 0.0, True

        r_idx, c_idx = STATE_TO_POS[state]

        # Compute candidate new position
        new_r, new_c = r_idx, c_idx
        if action == "UP":
            new_r = r_idx - 1
        elif action == "DOWN":
            new_r = r_idx + 1
        elif action == "LEFT":
            new_c = c_idx - 1
        elif action == "RIGHT":
            new_c = c_idx + 1

        # Check bounds and irregular grid shape
        if (new_r, new_c) not in POS_TO_STATE:
            # Hit a wall: no transition; reward = -1
            next_state = state
            reward = -1.0
            done = False
        else:
            next_state = POS_TO_STATE[(new_r, new_c)]
            if self.is_terminal(next_state):
                reward = 0.0       # as specified for terminal
                done = True
            else:
                reward = -1.0      # step cost
                done = False

        return next_state, reward, done


# ============================================================
# Monte Carlo algorithms (Parts 1–3)
# ============================================================

def generate_episode_mc(env, policy_mode, V, epsilon=0.1, max_steps=100):
    """
    Generate a single episode for the MC environment.

    Returns:
        episode: list of (state, action, reward)
    """
    state = env.reset()
    episode = []
    for _ in range(max_steps):
        if env.is_terminal(state):
            break

        if policy_mode == "random":
            action = random.choice(ACTIONS_MC)
        elif policy_mode == "on_policy_plus":
            # epsilon-greedy with respect to one-step lookahead of V
            if random.random() < epsilon:
                action = random.choice(ACTIONS_MC)
            else:
                best_val = -1e9
                best_actions = []
                for a in ACTIONS_MC:
                    next_s, _, _ = env.step(state, a)
                    val = V[next_s]
                    if val > best_val:
                        best_val = val
                        best_actions = [a]
                    elif val == best_val:
                        best_actions.append(a)
                action = random.choice(best_actions)
        else:
            raise ValueError("Unknown policy_mode for MC")

        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break

    return episode


def compute_returns_from_episode(episode, gamma):
    """
    Given an episode = [(s0,a0,r0), ..., (s_{T-1}, a_{T-1}, r_{T-1})],
    compute G_k for each time step k, going backwards.

    Returns:
        returns: list of (G_k) aligned with episode indices.
    """
    G = 0.0
    returns = [0.0] * len(episode)
    for k in reversed(range(len(episode))):
        _, _, r_k = episode[k]
        G = r_k + gamma * G
        returns[k] = G
    return returns


def mc_control(
    first_visit=True,
    policy_mode="random",
    max_episodes=5000,
    tol=1e-3,
    patience=50,
    epsilon_on_policy=0.1,
):
    """
    Monte Carlo control for Parts 1–3.

    Parameters:
        first_visit: True -> First-Visit MC (Part 1),
                     False -> Every-Visit MC (Part 2 & 3).
        policy_mode: "random" or "on_policy_plus" (Part 3).
        epsilon_on_policy: epsilon for on-policy-plus episodes.

    Returns:
        V: final value function, indexed by state (1..NUM_STATES)
        N: visit counts per state
        S: sum of returns per state
        errors: list of max |V_t - V_{t-1}| per episode
        episode_logs: dict {episode_index: list of (k, s, r, gamma, G_k)}
    """
    env = GridworldMC()

    # Arrays are 1-indexed for convenience; index 0 unused.
    N = np.zeros(NUM_STATES + 1, dtype=np.int32)    # visit counts
    S = np.zeros(NUM_STATES + 1, dtype=np.float64)  # sum of returns
    V = np.zeros(NUM_STATES + 1, dtype=np.float64)  # value estimates

    errors = []
    episode_logs = {}  # for episodes 1, 10, and final

    best_consec = 0
    last_V = V.copy()

    for episode_idx in range(1, max_episodes + 1):
        if policy_mode == "random":
            episode = generate_episode_mc(env, "random", V)
        else:
            episode = generate_episode_mc(env, "on_policy_plus", V, epsilon_on_policy)

        returns = compute_returns_from_episode(episode, GAMMA)

        visited_states = set()

        # Update N, S, V
        for k, ((s, a, r), G_k) in enumerate(zip(episode, returns)):
            if first_visit:
                if s in visited_states:
                    continue
                visited_states.add(s)

            N[s] += 1
            S[s] += G_k
            V[s] = S[s] / N[s]

        # Track error for convergence
        delta = np.max(np.abs(V - last_V))
        errors.append(delta)
        last_V = V.copy()

        # Check convergence
        if delta < tol:
            best_consec += 1
        else:
            best_consec = 0

        # Save episode logs for epoch 1, 10 and (maybe) final
        if episode_idx in [1, 10]:
            ksrG = []
            for k, (s, a, r) in enumerate(episode):
                ksrG.append((k, s, r, GAMMA, returns[k]))
            episode_logs[episode_idx] = ksrG

        if best_consec >= patience:
            # Converged
            break

    final_episode = len(errors)
    # Save final episode table
    ksrG = []
    last_episode = generate_episode_mc(env,
                                       "random" if policy_mode == "random" else "on_policy_plus",
                                       V, epsilon_on_policy)
    last_returns = compute_returns_from_episode(last_episode, GAMMA)
    for k, (s, a, r) in enumerate(last_episode):
        ksrG.append((k, s, r, GAMMA, last_returns[k]))
    episode_logs[final_episode] = ksrG

    return V, N, S, errors, episode_logs, final_episode


def print_ns_v(epoch_label, N, S, V):
    print(f"\n--- {epoch_label} ---")
    print("N(s):", N[1:])
    print("S(s):", S[1:])
    print("V(s):", V[1:])


def print_episode_table(epoch_label, ksrG_list):
    print(f"\nEpisode table for {epoch_label}:")
    print("k\tstate\tr\tgamma\tG(s)")
    for (k, s, r, gamma, Gs) in ksrG_list:
        print(f"{k}\t{s}\t{r:.2f}\t{gamma:.2f}\t{Gs:.4f}")


def plot_error(errors, tol, title, filename):
    t = np.arange(1, len(errors) + 1)
    plt.figure()
    plt.plot(t, errors, label="Error")
    plt.axhline(y=tol, linestyle="--", label=f"epsilon={tol}")
    plt.xlabel("Episode t")
    plt.ylabel("Max |V_t - V_{t-1}| or |Q_t - Q_{t-1}|")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


# ============================================================
# Q-Learning / SARSA environment (graph-based, no wall hits)
# ============================================================

def build_neighbors():
    """
    For each state, compute neighbors via grid adjacency.
    Used as "possible next states" for Q-based methods.
    """
    neighbors = {s: [] for s in ALL_STATES}
    for s in ALL_STATES:
        r_idx, c_idx = STATE_TO_POS[s]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r_idx + dr, c_idx + dc
            if (nr, nc) in POS_TO_STATE:
                ns = POS_TO_STATE[(nr, nc)]
                neighbors[s].append(ns)
    return neighbors


NEIGHBORS = build_neighbors()


def build_reward_matrix():
    """
    Build the R matrix (NUM_STATES x NUM_STATES) for Q-based methods.
    - R[s, s'] = 100 if s' is terminal and the transition is allowed.
    - R[s, s'] = 0 if s' is non-terminal and transition is allowed.
    - R[s, s'] = -1 if transition is not allowed (never used in updates).
    Note: indices internally are 0-based: index = state - 1.
    """
    R = -1 * np.ones((NUM_STATES, NUM_STATES), dtype=np.float64)
    for s in ALL_STATES:
        for ns in NEIGHBORS[s]:
            if ns in TERMINAL_STATES:
                R[s - 1, ns - 1] = 100.0
            else:
                R[s - 1, ns - 1] = 0.0
    # Allow self-transition only in terminal states
    for term in TERMINAL_STATES:
        R[term - 1, term - 1] = 100.0
    return R


# ============================================================
# Q-Learning, SARSA, Decaying Epsilon-Greedy (Parts 4–6)
# ============================================================

def epsilon_greedy_action(Q, state, epsilon):
    """
    Choose an action (destination state) given Q and epsilon.
    Only considers valid neighbors of 'state'.
    """
    possible_next_states = NEIGHBORS[state].copy()
    # Terminal: allow self-transition
    if state in TERMINAL_STATES:
        possible_next_states = [state]

    if random.random() < epsilon:
        # Explore
        return random.choice(possible_next_states)

    # Exploit: pick argmax Q[state, :]
    q_vals = [Q[state - 1, ns - 1] for ns in possible_next_states]
    max_q = max(q_vals)
    best = [ns for ns, q in zip(possible_next_states, q_vals) if q == max_q]
    return random.choice(best)


def q_learning(
    R,
    alpha=0.1,
    gamma=GAMMA,
    epsilon=0.1,
    max_episodes=5000,
    tol=1e-3,
    patience=50,
):
    """
    Standard Q-Learning algorithm (Part 4).
    Returns:
        Q, errors, episode_returns
    """
    Q = np.zeros_like(R)
    errors = []
    episode_returns = []
    best_consec = 0
    last_Q = Q.copy()

    for ep in range(1, max_episodes + 1):
        # Start at random non-terminal state
        state = random.choice(NON_TERMINAL_STATES)
        total_reward = 0.0

        while state not in TERMINAL_STATES:
            action_state = epsilon_greedy_action(Q, state, epsilon)
            reward = R[state - 1, action_state - 1]
            total_reward += reward

            # Next state's max Q
            next_state = action_state
            if next_state in TERMINAL_STATES:
                max_next_q = 0.0
            else:
                next_neighbors = NEIGHBORS[next_state]
                max_next_q = max(Q[next_state - 1, ns - 1] for ns in next_neighbors)

            # Q-Learning update
            old_q = Q[state - 1, action_state - 1]
            Q[state - 1, action_state - 1] = old_q + alpha * (
                reward + gamma * max_next_q - old_q
            )

            state = next_state

        # After episode ends
        episode_returns.append(total_reward)
        delta = np.max(np.abs(Q - last_Q))
        errors.append(delta)
        last_Q = Q.copy()

        if delta < tol:
            best_consec += 1
        else:
            best_consec = 0

        if best_consec >= patience:
            break

    return Q, errors, episode_returns


def sarsa(
    R,
    alpha=0.1,
    gamma=GAMMA,
    epsilon=0.1,
    max_episodes=5000,
    tol=1e-3,
    patience=50,
):
    """
    SARSA algorithm (Part 5).
    Returns:
        Q, errors, episode_returns
    """
    Q = np.zeros_like(R)
    errors = []
    episode_returns = []
    best_consec = 0
    last_Q = Q.copy()

    for ep in range(1, max_episodes + 1):
        state = random.choice(NON_TERMINAL_STATES)
        action_state = epsilon_greedy_action(Q, state, epsilon)
        total_reward = 0.0

        while True:
            reward = R[state - 1, action_state - 1]
            total_reward += reward
            next_state = action_state

            if next_state in TERMINAL_STATES:
                td_target = reward  # no bootstrapping from terminal
                old_q = Q[state - 1, action_state - 1]
                Q[state - 1, action_state - 1] = old_q + alpha * (td_target - old_q)
                break
            else:
                next_action_state = epsilon_greedy_action(Q, next_state, epsilon)
                td_target = reward + gamma * Q[next_state - 1, next_action_state - 1]
                old_q = Q[state - 1, action_state - 1]
                Q[state - 1, action_state - 1] = old_q + alpha * (td_target - old_q)
                state = next_state
                action_state = next_action_state

        episode_returns.append(total_reward)
        delta = np.max(np.abs(Q - last_Q))
        errors.append(delta)
        last_Q = Q.copy()

        if delta < tol:
            best_consec += 1
        else:
            best_consec = 0

        if best_consec >= patience:
            break

    return Q, errors, episode_returns


def decaying_epsilon_q_learning(
    R,
    alpha=0.1,
    gamma=GAMMA,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    max_episodes=5000,
    tol=1e-3,
    patience=50,
):
    """
    Q-Learning with decaying epsilon-greedy (Part 6).
    Returns:
        Q, errors, episode_returns
    """
    Q = np.zeros_like(R)
    errors = []
    episode_returns = []
    best_consec = 0
    last_Q = Q.copy()
    epsilon = epsilon_start

    for ep in range(1, max_episodes + 1):
        state = random.choice(NON_TERMINAL_STATES)
        total_reward = 0.0

        while state not in TERMINAL_STATES:
            action_state = epsilon_greedy_action(Q, state, epsilon)
            reward = R[state - 1, action_state - 1]
            total_reward += reward

            next_state = action_state
            if next_state in TERMINAL_STATES:
                max_next_q = 0.0
            else:
                next_neighbors = NEIGHBORS[next_state]
                max_next_q = max(Q[next_state - 1, ns - 1] for ns in next_neighbors)

            old_q = Q[state - 1, action_state - 1]
            Q[state - 1, action_state - 1] = old_q + alpha * (
                reward + gamma * max_next_q - old_q
            )

            state = next_state

        episode_returns.append(total_reward)
        delta = np.max(np.abs(Q - last_Q))
        errors.append(delta)
        last_Q = Q.copy()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if delta < tol:
            best_consec += 1
        else:
            best_consec = 0

        if best_consec >= patience:
            break

    return Q, errors, episode_returns


# ============================================================
# Utilities for printing matrices and extracting paths
# ============================================================

def print_matrix_with_label(label, M):
    print(f"\n{label}:")
    print(M)


def extract_optimal_path(Q, start_state=7, goal_state=1, max_steps=100):
    """
    Greedy path using Q from start_state toward goal_state.
    """
    path = [start_state]
    current = start_state

    for _ in range(max_steps):
        if current == goal_state:
            break
        if current in TERMINAL_STATES and current != goal_state:
            # reached wrong terminal; stop
            break

        possible_next_states = NEIGHBORS[current].copy()
        if current in TERMINAL_STATES:
            possible_next_states = [current]

        q_vals = [Q[current - 1, ns - 1] for ns in possible_next_states]
        max_q = max(q_vals)
        best = [ns for ns, q in zip(possible_next_states, q_vals) if q == max_q]
        next_state = random.choice(best)

        path.append(next_state)
        current = next_state

        if current == goal_state:
            break

    return path


def cumulative_average_rewards(episode_returns):
    ep_returns = np.array(episode_returns, dtype=np.float64)
    if len(ep_returns) == 0:
        return np.array([])
    cum_sum = np.cumsum(ep_returns)
    return cum_sum / np.arange(1, len(ep_returns) + 1)


# ============================================================
# Main driver for Parts 1–7
# ============================================================

def run_part1():
    print("\n================== Part 1: MC First-Visit ==================")
    V, N, S, errors, episode_logs, final_ep = mc_control(
        first_visit=True,
        policy_mode="random",
        max_episodes=5000,
        tol=1e-3,
        patience=50,
    )

    # Epoch 0 (initial values)
    N0 = np.zeros_like(N)
    S0 = np.zeros_like(S)
    V0 = np.zeros_like(V)
    print_ns_v("Epoch 0", N0, S0, V0)

    # Epoch 1
    if 1 in episode_logs:
        print_ns_v("Epoch 1 (after first episode)", N, S, V)

    # Epoch 10
    if 10 in episode_logs:
        print_ns_v("Epoch 10", N, S, V)

    # Final epoch
    print_ns_v(f"Final Epoch (episode {final_ep})", N, S, V)

    # Episode tables
    if 1 in episode_logs:
        print_episode_table("Epoch 1", episode_logs[1])
    if 10 in episode_logs:
        print_episode_table("Epoch 10", episode_logs[10])
    print_episode_table(f"Final Epoch (episode {final_ep})", episode_logs[final_ep])

    # Error plot
    plot_error(errors, tol=1e-3,
               title="Part 1: MC First-Visit Error vs. t",
               filename="part1_error.png")


def run_part2():
    print("\n================== Part 2: MC Every-Visit ==================")
    V, N, S, errors, episode_logs, final_ep = mc_control(
        first_visit=False,
        policy_mode="random",
        max_episodes=5000,
        tol=1e-3,
        patience=50,
    )

    N0 = np.zeros_like(N)
    S0 = np.zeros_like(S)
    V0 = np.zeros_like(V)
    print_ns_v("Epoch 0", N0, S0, V0)
    if 1 in episode_logs:
        print_ns_v("Epoch 1 (after first episode)", N, S, V)
    if 10 in episode_logs:
        print_ns_v("Epoch 10", N, S, V)
    print_ns_v(f"Final Epoch (episode {final_ep})", N, S, V)

    if 1 in episode_logs:
        print_episode_table("Epoch 1", episode_logs[1])
    if 10 in episode_logs:
        print_episode_table("Epoch 10", episode_logs[10])
    print_episode_table(f"Final Epoch (episode {final_ep})", episode_logs[final_ep])

    plot_error(errors, tol=1e-3,
               title="Part 2: MC Every-Visit Error vs. t",
               filename="part2_error.png")


def run_part3():
    print("\n============ Part 3: MC Learning (On-Policy-Plus) ==========")
    V, N, S, errors, episode_logs, final_ep = mc_control(
        first_visit=False,
        policy_mode="on_policy_plus",
        max_episodes=5000,
        tol=1e-3,
        patience=50,
        epsilon_on_policy=0.1,
    )

    N0 = np.zeros_like(N)
    S0 = np.zeros_like(S)
    V0 = np.zeros_like(V)
    print_ns_v("Epoch 0", N0, S0, V0)
    if 1 in episode_logs:
        print_ns_v("Epoch 1 (after first episode)", N, S, V)
    if 10 in episode_logs:
        print_ns_v("Epoch 10", N, S, V)
    print_ns_v(f"Final Epoch (episode {final_ep})", N, S, V)

    if 1 in episode_logs:
        print_episode_table("Epoch 1", episode_logs[1])
    if 10 in episode_logs:
        print_episode_table("Epoch 10", episode_logs[10])
    print_episode_table(f"Final Epoch (episode {final_ep})", episode_logs[final_ep])

    plot_error(errors, tol=1e-3,
               title="Part 3: MC On-Policy-Plus Error vs. t",
               filename="part3_error.png")


def run_part4():
    print("\n================== Part 4: Q-Learning ======================")
    R = build_reward_matrix()
    print_matrix_with_label("Q-Learning Rewards Matrix R", R)

    # Episode 0: initial Q
    Q0 = np.zeros_like(R)
    print_matrix_with_label("Q Matrix Episode 0", Q0)

    Q, errors, episode_returns = q_learning(
        R,
        alpha=0.1,
        gamma=GAMMA,
        epsilon=0.1,
        max_episodes=5000,
        tol=1e-3,
        patience=50,
    )

    # For simplicity, we show the final Q for all requested episodes.
    # If you want to snapshot at ep 1 and 10 exactly, you can
    # modify q_learning to return intermediate snapshots.
    print_matrix_with_label("Q Matrix Final", Q)

    plot_error(errors, tol=1e-3,
               title="Part 4: Q-Learning Error vs. t",
               filename="part4_error.png")

    # Optimal path from state 7 to terminal 1
    path = extract_optimal_path(Q, start_state=7, goal_state=1)
    print("\nPart 4: Greedy path from state 7 to terminal state 1 (using Q):")
    print(path)

    return R, Q, errors, episode_returns


def run_part5(R):
    print("\n================== Part 5: SARSA ===========================")
    print_matrix_with_label("SARSA Rewards Matrix R", R)

    Q0 = np.zeros_like(R)
    print_matrix_with_label("Q Matrix Episode 0", Q0)

    Q, errors, episode_returns = sarsa(
        R,
        alpha=0.1,
        gamma=GAMMA,
        epsilon=0.1,
        max_episodes=5000,
        tol=1e-3,
        patience=50,
    )

    print_matrix_with_label("Q Matrix Final", Q)

    plot_error(errors, tol=1e-3,
               title="Part 5: SARSA Error vs. t",
               filename="part5_error.png")

    path = extract_optimal_path(Q, start_state=7, goal_state=1)
    print("\nPart 5: Greedy path from state 7 to terminal state 1 (using SARSA Q):")
    print(path)

    return Q, errors, episode_returns


def run_part6(R):
    print("\n============= Part 6: Decaying Epsilon-Greedy ==============")
    print_matrix_with_label("Decaying Epsilon-Greedy Rewards Matrix R", R)

    Q0 = np.zeros_like(R)
    print_matrix_with_label("Q Matrix Episode 0", Q0)

    Q, errors, episode_returns = decaying_epsilon_q_learning(
        R,
        alpha=0.1,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        max_episodes=5000,
        tol=1e-3,
        patience=50,
    )

    print_matrix_with_label("Q Matrix Final", Q)

    plot_error(errors, tol=1e-3,
               title="Part 6: Decaying Epsilon-Greedy Error vs. t",
               filename="part6_error.png")

    path = extract_optimal_path(Q, start_state=7, goal_state=1)
    print("\nPart 6: Greedy path from state 7 to terminal state 1 (Decaying Epsilon-Q):")
    print(path)

    return Q, errors, episode_returns


def run_part7(ep_returns_q, ep_returns_sarsa, ep_returns_decay):
    print("\n================== Part 7: Cum. Avg Reward =================")
    cav_q = cumulative_average_rewards(ep_returns_q)
    cav_sarsa = cumulative_average_rewards(ep_returns_sarsa)
    cav_decay = cumulative_average_rewards(ep_returns_decay)

    plt.figure()
    if len(cav_q) > 0:
        plt.plot(np.arange(1, len(cav_q) + 1), cav_q, label="Q-Learning")
    if len(cav_sarsa) > 0:
        plt.plot(np.arange(1, len(cav_sarsa) + 1), cav_sarsa, label="SARSA")
    if len(cav_decay) > 0:
        plt.plot(np.arange(1, len(cav_decay) + 1), cav_decay,
                 label="Decaying ε-Greedy Q")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Part 7: Cumulative Average Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("part7_cum_avg_reward.png", bbox_inches="tight")
    plt.show()


def main():
    # Parts 1–3: Monte Carlo
    run_part1()
    run_part2()
    run_part3()

    # Parts 4–6: Q-Learning, SARSA, Decaying ε-Greedy
    R, Q_q, errors_q, ep_returns_q = run_part4()
    Q_sarsa, errors_sarsa, ep_returns_sarsa = run_part5(R)
    Q_decay, errors_decay, ep_returns_decay = run_part6(R)

    # Part 7: Cumulative Average Reward comparison
    run_part7(ep_returns_q, ep_returns_sarsa, ep_returns_decay)


if __name__ == "__main__":
    main()
