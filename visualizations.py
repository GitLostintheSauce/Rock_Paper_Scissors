"""
visualizations.py — All matplotlib plotting functions for the RPS project.

This file creates 5 different graphs:
    1. Grouped bar chart — choice frequencies per user
    2. Heatmaps — transition probabilities (what follows each move)
    3. Bar chart — prediction accuracy vs. random baseline
    4. Bar chart — entropy (randomness) per user
    5. Line chart — choice trends over time (regression visualization)
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
VALID_CHOICES = ["rock", "paper", "scissors"]
USERS = ["Dana", "Jarin", "Eleanor", "Krew"]
COLORS = ["#4A90D9", "#E8915A", "#6BBF6B", "#D94A7A"]


def plot_choice_frequencies(freq_dict):
    """GRAPH 1: Grouped bar chart — how often each user picks each move.

    Creates three bars per user (one for rock, paper, scissors) so you
    can visually compare who favors which move.

    Parameters:
        freq_dict (dict): Output from compute_choice_frequencies().
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(USERS))
    bar_width = 0.25

    colors = ["#4A90D9", "#E8915A", "#6BBF6B"]
    for i, choice in enumerate(VALID_CHOICES):
        counts = [freq_dict[user][choice] for user in USERS]
        ax.bar(x + i * bar_width, counts, bar_width,
               label=choice.capitalize(), color=colors[i])

    ax.set_xlabel("Player", fontsize=12)
    ax.set_ylabel("Number of Times Chosen", fontsize=12)
    ax.set_title("Rock-Paper-Scissors Choice Frequency by Player", fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(USERS)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph1_choice_frequencies.png", dpi=150)
    plt.show()
    print("  Saved: graph1_choice_frequencies.png")


def plot_transition_heatmaps(transition_matrices):
    """GRAPH 2: Heatmaps — transition probabilities for each user.

    Each heatmap is a 3x3 grid. Row = previous move, column = next move.
    The color and number show how likely that transition is (0.0 to 1.0).

    Parameters:
        transition_matrices (dict): Output from build_transition_matrix().
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, user in enumerate(USERS):
        ax = axes_flat[idx]
        matrix = transition_matrices[user]

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i][j]:.2f}",
                        ha="center", va="center", fontsize=12,
                        color="black" if matrix[i][j] < 0.6 else "white")

        labels = ["Rock", "Paper", "Scissors"]
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Next Move")
        ax.set_ylabel("Previous Move")
        ax.set_title(f"{user}'s Transitions")

    fig.suptitle("Transition Probabilities: What Comes After Each Move?",
                 fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Probability")

    plt.savefig("graph2_transition_heatmaps.png", dpi=150)
    plt.show()
    print("  Saved: graph2_transition_heatmaps.png")


def plot_prediction_accuracy(accuracy_dict):
    """GRAPH 3: Bar chart — prediction accuracy with random baseline.

    Draws a red dashed line at 33.3% (what you'd get from random guessing).
    Any bar above that line means the player has detectable patterns.

    Parameters:
        accuracy_dict (dict): Output from calculate_prediction_accuracy().
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    users = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    bars = ax.bar(users, accuracies, color=COLORS, edgecolor="black")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc}%", ha="center", fontsize=12, fontweight="bold")

    ax.axhline(y=33.3, color="red", linestyle="--", linewidth=2,
               label="Random Guess (33.3%)")

    ax.set_xlabel("Player", fontsize=12)
    ax.set_ylabel("Prediction Accuracy (%)", fontsize=12)
    ax.set_title("Can We Predict the Next Move? (Classification Results)",
                 fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph3_prediction_accuracy.png", dpi=150)
    plt.show()
    print("  Saved: graph3_prediction_accuracy.png")


def plot_entropy(stats):
    """GRAPH 4: Bar chart — entropy (randomness measure) per user.

    Entropy measures how unpredictable a player is. A perfectly random
    player scores 1.585 (log2(3)). Lower = more predictable patterns.

    Parameters:
        stats (dict): Output from compute_stats().
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    users = list(stats.keys())
    entropies = [stats[user]["entropy"] for user in users]
    max_entropy = np.log2(3)

    bars = ax.bar(users, entropies, color=COLORS, edgecolor="black")

    for bar, ent in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ent:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.axhline(y=max_entropy, color="red", linestyle="--", linewidth=2,
               label=f"Perfectly Random ({max_entropy:.3f})")

    ax.set_xlabel("Player", fontsize=12)
    ax.set_ylabel("Entropy (bits)", fontsize=12)
    ax.set_title("How Random Is Each Player? (Entropy Analysis)", fontsize=14)
    ax.set_ylim(0, 2.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph4_entropy.png", dpi=150)
    plt.show()
    print("  Saved: graph4_entropy.png")


def plot_trends(df, regression_results):
    """GRAPH 5: Line chart — choice trends over time (regression).

    For each user, shows the running percentage of 'rock' picks over
    rounds, with a trend line from the regression. This visualizes
    whether a player's habits change over the course of the game.

    Parameters:
        df (pd.DataFrame): The cleaned data.
        regression_results (dict): Output from regression_analysis().
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for idx, user in enumerate(USERS):
        ax = axes_flat[idx]
        user_data = df[df["user"] == user].sort_values("round")
        rounds = list(user_data["round"])

        # Plot running percentage for each choice
        for c_idx, choice in enumerate(VALID_CHOICES):
            running_pct = []
            count = 0
            for i, c in enumerate(user_data["choice"]):
                if c == choice:
                    count += 1
                running_pct.append(count / (i + 1) * 100)

            color = ["#4A90D9", "#E8915A", "#6BBF6B"][c_idx]
            ax.plot(rounds, running_pct, marker="o", markersize=3,
                    label=choice.capitalize(), color=color, linewidth=1.5)

        # Draw the 33.3% baseline (what perfectly random would look like)
        ax.axhline(y=33.3, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Round")
        ax.set_ylabel("Running %")
        ax.set_title(f"{user}'s Choice Trends Over Time")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Trend Analysis: Do Habits Change Over Time? (Regression)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graph5_trends.png", dpi=150)
    plt.show()
    print("  Saved: graph5_trends.png")
