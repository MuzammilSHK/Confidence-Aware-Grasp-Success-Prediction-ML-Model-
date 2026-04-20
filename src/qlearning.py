"""
Q-Learning for simple robotic navigation.

A robot on a 7x7 grid must find a path from start (top-left) to goal
(bottom-right) while avoiding obstacles. The agent learns a policy via
tabular Q-Learning: classic epsilon-greedy action selection with a
Bellman update on each step.

This complements the grasping models: together they cover the pipeline
of "decide to grasp" (classification/regression) and "decide where to
move next" (reinforcement learning) — the two halves of a robot that
picks up objects in an environment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")


# =============================================================================
# Grid Environment
# =============================================================================

class GridWorld:
    """
    Layout:
      S = start (0, 0), G = goal (n-1, n-1), X = obstacle
      Reward: +10 at goal, -10 at obstacle (terminal), -1 per step
    """

    def __init__(self, size=7, obstacles=None, seed=0):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        if obstacles is None:
            rng = np.random.default_rng(seed)
            # Pepper about 20% of the grid with obstacles, keeping start/goal free
            obstacles = set()
            target_count = int(size * size * 0.2)
            while len(obstacles) < target_count:
                cell = (int(rng.integers(0, size)), int(rng.integers(0, size)))
                if cell != self.start and cell != self.goal:
                    obstacles.add(cell)
        self.obstacles = set(obstacles)

        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # U D L R
        self.action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
        self.n_actions = 4
        self.n_states = size * size

    def state_to_idx(self, state):
        return state[0] * self.size + state[1]

    def reset(self):
        return self.start

    def step(self, state, action):
        dr, dc = self.actions[action]
        new_r = max(0, min(self.size - 1, state[0] + dr))
        new_c = max(0, min(self.size - 1, state[1] + dc))
        new_state = (new_r, new_c)

        if new_state in self.obstacles:
            return new_state, -10.0, True
        if new_state == self.goal:
            return new_state, 10.0, True
        return new_state, -1.0, False


# =============================================================================
# Q-Learning Agent
# =============================================================================

class QLearningAgent:

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=42):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state_idx):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s_idx, a, reward, s_next_idx, done):
        best_next = 0.0 if done else np.max(self.Q[s_next_idx])
        td_target = reward + self.gamma * best_next
        self.Q[s_idx, a] += self.alpha * (td_target - self.Q[s_idx, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# Training Loop
# =============================================================================

def train(env, agent, n_episodes=1000, max_steps=100):
    rewards_history = []
    steps_history = []
    success_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            s_idx = env.state_to_idx(state)
            action = agent.choose_action(s_idx)
            next_state, reward, done = env.step(state, action)
            s_next_idx = env.state_to_idx(next_state)
            agent.update(s_idx, action, reward, s_next_idx, done)
            state = next_state
            total_reward += reward
            steps += 1

        rewards_history.append(total_reward)
        steps_history.append(steps)
        success_history.append(1 if state == env.goal else 0)
        agent.decay_epsilon()

    return {
        "rewards": rewards_history,
        "steps": steps_history,
        "success": success_history,
    }


# =============================================================================
# Evaluation
# =============================================================================

def extract_policy(env, agent):
    policy = np.zeros((env.size, env.size), dtype=int)
    for r in range(env.size):
        for c in range(env.size):
            idx = env.state_to_idx((r, c))
            policy[r, c] = int(np.argmax(agent.Q[idx]))
    return policy


def rollout_greedy(env, agent, max_steps=50):
    state = env.reset()
    path = [state]
    total_reward = 0.0
    for _ in range(max_steps):
        s_idx = env.state_to_idx(state)
        action = int(np.argmax(agent.Q[s_idx]))
        state, reward, done = env.step(state, action)
        path.append(state)
        total_reward += reward
        if done:
            break
    return path, total_reward, state == env.goal


# =============================================================================
# Visualization
# =============================================================================

def plot_learning_curves(history, save_path, smoothing=25):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    rewards = np.array(history["rewards"])
    smooth_rewards = np.convolve(rewards,
                                 np.ones(smoothing) / smoothing, mode="valid")
    axes[0].plot(rewards, alpha=0.3, label="Raw")
    axes[0].plot(np.arange(smoothing - 1, len(rewards)), smooth_rewards,
                 linewidth=2, label=f"Smoothed ({smoothing}-ep)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Reward per Episode")
    axes[0].legend()

    steps = np.array(history["steps"])
    smooth_steps = np.convolve(steps,
                               np.ones(smoothing) / smoothing, mode="valid")
    axes[1].plot(steps, alpha=0.3, color="darkorange", label="Raw")
    axes[1].plot(np.arange(smoothing - 1, len(steps)), smooth_steps,
                 color="darkorange", linewidth=2, label=f"Smoothed")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Steps to Terminate")
    axes[1].set_title("Episode Length")
    axes[1].legend()

    success = np.array(history["success"])
    smooth_success = np.convolve(success,
                                 np.ones(smoothing) / smoothing, mode="valid")
    axes[2].plot(np.arange(smoothing - 1, len(success)), smooth_success,
                 color="seagreen", linewidth=2)
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Success Rate (rolling)")
    axes[2].set_title(f"Goal Reached — Rolling Mean ({smoothing}-ep)")
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_policy_and_path(env, policy, path, save_path):
    grid_display = np.zeros((env.size, env.size))
    for obs in env.obstacles:
        grid_display[obs] = -1
    grid_display[env.goal] = 2
    grid_display[env.start] = 1

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.colors.ListedColormap(["black", "white", "blue", "green"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(grid_display, cmap=cmap, norm=norm)

    action_arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles:
                ax.text(c, r, "X", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
            elif (r, c) == env.goal:
                ax.text(c, r, "G", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
            elif (r, c) == env.start:
                ax.text(c, r, "S", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
            else:
                ax.text(c, r, action_arrows[policy[r, c]],
                        ha="center", va="center", color="black", fontsize=16)

    # Overlay learned path
    if len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, "r-", linewidth=2.5, alpha=0.7, marker="o",
                markersize=6)

    ax.set_xticks(np.arange(env.size))
    ax.set_yticks(np.arange(env.size))
    ax.set_title("Learned Policy and Greedy Path\n(S=Start, G=Goal, X=Obstacle)")
    ax.grid(True, color="grey", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_q_heatmap(env, agent, save_path):
    """Show the max Q-value at each state — brighter cells have higher value."""
    value = np.max(agent.Q, axis=1).reshape(env.size, env.size)
    for obs in env.obstacles:
        value[obs] = np.nan

    plt.figure(figsize=(7, 6))
    sns.heatmap(value, annot=True, fmt=".1f", cmap="viridis",
                cbar_kws={"label": "max Q(s, a)"})
    plt.title("State-Value Heatmap (max Q per state)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# Convergence Analysis
# =============================================================================

def analyze_convergence(history, window=50):
    """Find the first episode where the rolling mean stays above a success rate of 90%."""
    success = np.array(history["success"])
    if len(success) < window:
        return None

    rolling = np.convolve(success, np.ones(window) / window, mode="valid")
    above = np.where(rolling >= 0.9)[0]
    convergence_ep = int(above[0]) + window if len(above) > 0 else None

    final_success_rate = float(rolling[-1])
    final_avg_reward = float(np.mean(history["rewards"][-window:]))

    return {
        "convergence_episode": convergence_ep,
        "final_success_rate": final_success_rate,
        "final_avg_reward": final_avg_reward,
    }


# =============================================================================
# Full Q-Learning Pipeline
# =============================================================================

def run_qlearning(n_episodes=1500, grid_size=7, seed=0):
    print(f"\n[Q-Learning] Grid {grid_size}x{grid_size}, {n_episodes} episodes...")
    env = GridWorld(size=grid_size, seed=seed)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)

    history = train(env, agent, n_episodes=n_episodes)
    convergence = analyze_convergence(history)
    policy = extract_policy(env, agent)
    path, greedy_reward, reached_goal = rollout_greedy(env, agent)

    plot_learning_curves(history, os.path.join(FIG_DIR, "qlearning_curves.png"))
    plot_policy_and_path(env, policy, path,
                         os.path.join(FIG_DIR, "qlearning_policy.png"))
    plot_q_heatmap(env, agent, os.path.join(FIG_DIR, "qlearning_q_heatmap.png"))

    results = {
        "n_episodes": n_episodes,
        "grid_size": grid_size,
        "n_obstacles": len(env.obstacles),
        "final_greedy_reward": float(greedy_reward),
        "reached_goal_under_greedy_policy": bool(reached_goal),
        "greedy_path_length": len(path),
        "convergence": convergence,
    }

    with open(os.path.join(METRIC_DIR, "qlearning_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Q-Learning] Greedy rollout: reached_goal={reached_goal}, "
          f"reward={greedy_reward:.1f}, path_length={len(path)}")
    if convergence and convergence["convergence_episode"]:
        print(f"[Q-Learning] Converged at episode "
              f"{convergence['convergence_episode']} "
              f"(final success rate: {convergence['final_success_rate']:.2%})")
    return agent, history, results


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    run_qlearning()
