"""
analysis.py — All analysis and algorithm functions for the RPS project.

This file contains:
    - Choice frequency analysis
    - Transition matrix builder (for pattern detection)
    - Move prediction (CLASSIFICATION algorithm)
    - Prediction accuracy calculator
    - Summary statistics with entropy
    - K-means CLUSTERING of users by play style
    - Linear REGRESSION to detect trends over time
"""

import numpy as np

# Constants
VALID_CHOICES = ["rock", "paper", "scissors"]
USERS = ["Dana", "Jarin", "Eleanor", "Krew"]


# =========================================================================
# Basic Analysis Functions
# =========================================================================

def compute_choice_frequencies(df):
    """Count how often each user picks rock, paper, and scissors.

    Builds a dictionary where each key is a user name and each value
    is another dictionary with the counts for rock, paper, and scissors.

    Parameters:
        df (pd.DataFrame): The cleaned data with 'user' and 'choice' columns.

    Returns:
        dict: Nested dictionary like {'Dana': {'rock': 12, 'paper': 8, ...}}
    """
    frequency_dict = {}

    for user in USERS:
        user_data = df[df["user"] == user]
        counts = {}
        for choice in VALID_CHOICES:
            counts[choice] = int((user_data["choice"] == choice).sum())
        frequency_dict[user] = counts

    return frequency_dict


def build_transition_matrix(df):
    """Build a transition matrix for each user.

    A transition matrix shows: given that the previous move was X, what
    is the probability that the next move is Y?

    For example, if after playing 'rock' a user plays 'paper' 5 out of
    10 times, then the transition probability from rock -> paper is 0.5.

    Parameters:
        df (pd.DataFrame): The cleaned data.

    Returns:
        dict: Maps each user to a 3x3 numpy array.
              Rows = previous move, Columns = next move.
              Order: rock, paper, scissors.
    """
    transition_matrices = {}

    for user in USERS:
        user_data = df[df["user"] == user].sort_values("round")
        choices_list = list(user_data["choice"])

        # Create a 3x3 matrix of zeros using numpy
        matrix = np.zeros((3, 3))

        # Walk through consecutive pairs using a while loop
        round_num = 0
        while round_num < len(choices_list) - 1:
            current = choices_list[round_num]
            next_choice = choices_list[round_num + 1]

            row = VALID_CHOICES.index(current)
            col = VALID_CHOICES.index(next_choice)
            matrix[row][col] += 1

            round_num += 1

        # Convert counts to probabilities (each row sums to 1.0)
        for i in range(3):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] = matrix[i] / row_sum

        transition_matrices[user] = matrix

    return transition_matrices


# =========================================================================
# ALGORITHM 1: CLASSIFICATION — Predict next move from previous move
# =========================================================================

def predict_next_move(transition_matrices, user, previous_move):
    """Predict a user's next move based on their transition matrix.

    This is a CLASSIFICATION algorithm: given an input (the previous move),
    it classifies the output into one of three categories (rock, paper,
    or scissors) based on learned probabilities.

    Parameters:
        transition_matrices (dict): From build_transition_matrix().
        user (str): The user's name.
        previous_move (str): The move that was just played.

    Returns:
        str: The predicted next move ('rock', 'paper', or 'scissors').
    """
    matrix = transition_matrices[user]
    row = VALID_CHOICES.index(previous_move)
    probabilities = matrix[row]

    # Pick the move with the highest probability
    best_index = int(np.argmax(probabilities))
    return VALID_CHOICES[best_index]


def calculate_prediction_accuracy(df, transition_matrices):
    """Test how well our classification predictions work.

    For each user, go through every round starting from round 2.
    Use the previous move to predict the current move, then check
    if the prediction matched the actual move.

    Parameters:
        df (pd.DataFrame): The cleaned data.
        transition_matrices (dict): The transition matrices.

    Returns:
        dict: Maps each user to their accuracy percentage.
    """
    accuracy_dict = {}

    for user in USERS:
        user_data = df[df["user"] == user].sort_values("round")
        choices_list = list(user_data["choice"])

        correct = 0
        total = 0

        for i in range(1, len(choices_list)):
            previous = choices_list[i - 1]
            actual = choices_list[i]
            predicted = predict_next_move(transition_matrices, user, previous)

            if predicted == actual:
                correct += 1
            total += 1

        if total > 0:
            accuracy = (correct / total) * 100
        else:
            accuracy = 0.0

        accuracy_dict[user] = round(accuracy, 1)

    return accuracy_dict


# =========================================================================
# ALGORITHM 2: CLUSTERING — Group users by play style using K-Means
# =========================================================================

def cluster_users_kmeans(freq_dict, k=2):
    """Cluster users into groups based on their choice frequencies.

    Uses a basic K-Means algorithm (implemented manually with numpy).
    Each user is represented as a 3D point: (% rock, % paper, % scissors).
    Users with similar play styles end up in the same cluster.

    K-Means works by:
        1. Pick k random starting center points.
        2. Assign each user to the nearest center.
        3. Move each center to the average of its assigned users.
        4. Repeat steps 2-3 until centers stop moving.

    Parameters:
        freq_dict (dict): Output from compute_choice_frequencies().
        k (int): Number of clusters (default 2).

    Returns:
        dict: Contains 'labels' (which cluster each user belongs to),
              'centers' (the center point of each cluster),
              'user_points' (the data point for each user).
    """
    # Convert each user's counts to percentages -> a 3D point
    user_points = {}
    data_points = []

    for user in USERS:
        counts = freq_dict[user]
        total = sum(counts.values())
        point = np.array([counts[c] / total for c in VALID_CHOICES])
        user_points[user] = point
        data_points.append(point)

    data = np.array(data_points)  # shape: (4, 3)

    # Initialize centers using first k points (deterministic for reproducibility)
    centers = data[:k].copy()

    # Run K-Means for up to 20 iterations
    labels = np.zeros(len(USERS), dtype=int)
    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        old_labels = labels.copy()

        # Step A: Assign each user to the nearest center
        for i in range(len(USERS)):
            distances = []
            for j in range(k):
                # Euclidean distance between the user's point and each center
                dist = np.sqrt(np.sum((data[i] - centers[j]) ** 2))
                distances.append(dist)
            labels[i] = int(np.argmin(distances))

        # Step B: Update centers to the average of their members
        for j in range(k):
            members = data[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)

        # Check if anything changed; if not, we've converged
        if np.array_equal(labels, old_labels):
            break

        iteration += 1

    return {
        "labels": {USERS[i]: int(labels[i]) for i in range(len(USERS))},
        "centers": centers,
        "user_points": user_points,
        "iterations": iteration + 1
    }


# =========================================================================
# ALGORITHM 3: REGRESSION — Detect trends in choices over time
# =========================================================================

def regression_analysis(df):
    """Perform linear regression to detect trends in each user's choices.

    For each user and each choice (rock, paper, scissors), we look at
    whether they pick that choice more or less as rounds go on.

    We use the least-squares formula for a line y = mx + b:
        m = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)

    A positive slope means the user picks that choice MORE over time.
    A negative slope means they pick it LESS over time.

    Parameters:
        df (pd.DataFrame): The cleaned data.

    Returns:
        dict: For each user, a dict of slopes and descriptions per choice.
    """
    results = {}

    for user in USERS:
        user_data = df[df["user"] == user].sort_values("round")
        rounds = np.array(user_data["round"], dtype=float)
        user_results = {}

        for choice in VALID_CHOICES:
            # Create a binary array: 1 if they picked this choice, 0 if not
            y = np.array([1.0 if c == choice else 0.0
                          for c in user_data["choice"]])
            x = rounds

            # Least-squares linear regression formula
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x ** 2)

            denominator = (n * sum_x2 - sum_x ** 2)
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            else:
                slope = 0.0

            # Describe the trend
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"

            user_results[choice] = {
                "slope": round(slope, 4),
                "trend": trend
            }

        results[user] = user_results

    return results


# =========================================================================
# Summary Statistics
# =========================================================================

def compute_stats(df):
    """Compute summary statistics for each user.

    Calculates: most/least common choice, longest streak of the same
    move in a row, and entropy (a measure of randomness).

    Parameters:
        df (pd.DataFrame): The cleaned data.

    Returns:
        dict: Statistics for each user.
    """
    stats = {}

    for user in USERS:
        user_data = df[df["user"] == user].sort_values("round")
        choices_list = list(user_data["choice"])
        user_stats = {}

        # Most and least common choice
        counts = user_data["choice"].value_counts()
        user_stats["most_common"] = counts.index[0]
        user_stats["least_common"] = counts.index[-1]
        user_stats["total_rounds"] = len(choices_list)

        # Longest streak of the same choice in a row
        max_streak = 1
        current_streak = 1
        for i in range(1, len(choices_list)):
            if choices_list[i] == choices_list[i - 1]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 1
        user_stats["longest_streak"] = max_streak

        # Entropy: measures randomness. Max for 3 choices = log2(3) ≈ 1.585
        probs = []
        for choice in VALID_CHOICES:
            p = (user_data["choice"] == choice).sum() / len(user_data)
            if p > 0:
                probs.append(p)
        entropy = -sum(p * np.log2(p) for p in probs)
        user_stats["entropy"] = round(entropy, 3)

        stats[user] = user_stats

    return stats
