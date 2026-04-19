"""
=============================================================================
Can AI Predict Human Randomness?
Rock-Paper-Scissors Pattern Analysis
ENGR 010 Group Project - Spring 2026
Team Members: Dana, Jarin, Eleanor, Krew
=============================================================================

This is the MAIN file that ties the project together. It imports functions
from our other files (data_loader.py, analysis.py, visualizations.py) and
provides an interactive menu so the user can explore the data.

To run:  python3 rps_analysis.py
"""

# =============================================================================
# Import our custom modules (the other Python files we wrote)
# =============================================================================
from data_loader import load_csv, load_json, export_to_json, validate_data
from analysis import (compute_choice_frequencies, build_transition_matrix,
                      predict_next_move, calculate_prediction_accuracy,
                      compute_stats, cluster_users_kmeans,
                      regression_analysis, USERS, VALID_CHOICES)
from visualizations import (plot_choice_frequencies, plot_transition_heatmaps,
                            plot_prediction_accuracy, plot_entropy,
                            plot_trends)

# Constants
CSV_FILE = "rps_data.csv"
JSON_FILE = "rps_data.json"


# =============================================================================
# Display Functions (for printing results to the terminal)
# =============================================================================

def display_frequencies(freq_dict):
    """Print choice frequency results in a readable format.

    Parameters:
        freq_dict (dict): Output from compute_choice_frequencies().
    """
    print("\n  --- Choice Frequencies ---")
    for user in USERS:
        counts = freq_dict[user]
        print(f"  {user:>8}: Rock={counts['rock']}, "
              f"Paper={counts['paper']}, Scissors={counts['scissors']}")


def display_transitions(transition_matrices):
    """Print transition matrices, highlighting strong tendencies.

    Parameters:
        transition_matrices (dict): Output from build_transition_matrix().
    """
    print("\n  --- Transition Probabilities ---")
    for user in USERS:
        print(f"\n  {user}:")
        matrix = transition_matrices[user]
        for i, prev in enumerate(VALID_CHOICES):
            for j, nxt in enumerate(VALID_CHOICES):
                prob = matrix[i][j]
                if prob > 0.45:
                    print(f"    After {prev:>8} -> {nxt:>8}: "
                          f"{prob:.0%}  <-- strong tendency!")
                elif prob > 0:
                    print(f"    After {prev:>8} -> {nxt:>8}: {prob:.0%}")


def display_accuracy(accuracy_dict):
    """Print prediction accuracy with verdicts.

    Parameters:
        accuracy_dict (dict): Output from calculate_prediction_accuracy().
    """
    print("\n  --- Classification: Prediction Accuracy ---")
    print(f"  {'Player':<10} {'Accuracy':>10}  {'Verdict'}")
    print(f"  {'-'*40}")
    for user in USERS:
        acc = accuracy_dict[user]
        if acc > 50:
            verdict = "Very predictable!"
        elif acc > 33.3:
            verdict = "Somewhat predictable"
        else:
            verdict = "Hard to predict"
        print(f"  {user:<10} {acc:>9.1f}%  {verdict}")


def display_stats(stats):
    """Print summary statistics for each user.

    Parameters:
        stats (dict): Output from compute_stats().
    """
    print("\n  --- Summary Statistics ---")
    for user in USERS:
        s = stats[user]
        print(f"  {user}:")
        print(f"    Favorite move:  {s['most_common']}")
        print(f"    Least used:     {s['least_common']}")
        print(f"    Longest streak: {s['longest_streak']} in a row")
        print(f"    Entropy:        {s['entropy']} / 1.585 "
              f"({'very random' if s['entropy'] > 1.5 else 'has patterns'})")


def display_clusters(cluster_results):
    """Print clustering results.

    Parameters:
        cluster_results (dict): Output from cluster_users_kmeans().
    """
    print("\n  --- K-Means Clustering: Play Style Groups ---")
    labels = cluster_results["labels"]
    points = cluster_results["user_points"]

    # Group users by cluster
    clusters = {}
    for user, label in labels.items():
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(user)

    for cluster_id, members in clusters.items():
        print(f"\n  Cluster {cluster_id + 1}: {', '.join(members)}")
        for user in members:
            p = points[user]
            print(f"    {user}: Rock={p[0]:.0%}, Paper={p[1]:.0%}, "
                  f"Scissors={p[2]:.0%}")

    print(f"\n  K-Means converged in {cluster_results['iterations']} "
          f"iterations.")


def display_regression(regression_results):
    """Print regression results showing trends.

    Parameters:
        regression_results (dict): Output from regression_analysis().
    """
    print("\n  --- Linear Regression: Trends Over Time ---")
    for user in USERS:
        print(f"\n  {user}:")
        for choice in VALID_CHOICES:
            r = regression_results[user][choice]
            slope = r["slope"]
            trend = r["trend"]
            print(f"    {choice:>8}: slope={slope:+.4f}  ({trend})")


def display_conclusion(accuracy_dict):
    """Print the final conclusion.

    Parameters:
        accuracy_dict (dict): Output from calculate_prediction_accuracy().
    """
    print("\n" + "=" * 60)
    print("  FINAL CONCLUSION")
    print("=" * 60)
    overall_avg = sum(accuracy_dict.values()) / len(accuracy_dict)
    print(f"\n  Average prediction accuracy: {overall_avg:.1f}%")
    print(f"  Random guessing would give:  33.3%")
    if overall_avg > 33.3:
        improvement = overall_avg - 33.3
        print(f"\n  Our model beats random guessing by {improvement:.1f} "
              f"percentage points!")
        print("  Conclusion: Humans are NOT perfectly random. Their")
        print("  patterns can be detected and partially predicted!")
    else:
        print("\n  Our model didn't beat random guessing.")
        print("  These players are pretty unpredictable!")


# =============================================================================
# Interactive Menu
# =============================================================================

def show_menu():
    """Display the interactive menu options."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MENU")
    print("=" * 60)
    print("  1. View choice frequencies")
    print("  2. View transition analysis")
    print("  3. View prediction accuracy (Classification)")
    print("  4. View player statistics")
    print("  5. View clustering results (K-Means)")
    print("  6. View regression trends")
    print("  7. Generate all graphs")
    print("  8. Export data to JSON")
    print("  9. Run full analysis (everything)")
    print("  0. Exit")
    print("=" * 60)


def run_interactive(df, freq_dict, transition_matrices, accuracy_dict,
                    stats, cluster_results, regression_results):
    """Run the interactive menu loop.

    Lets the user choose which analysis to view. Keeps running until
    the user picks '0' to exit.

    Parameters:
        All parameters are pre-computed analysis results.
    """
    running = True
    while running:
        show_menu()
        try:
            choice = input("  Enter your choice (0-9): ").strip()
        except EOFError:
            # If running non-interactively, just run everything
            choice = "9"
            running = False

        if choice == "1":
            display_frequencies(freq_dict)
        elif choice == "2":
            display_transitions(transition_matrices)
        elif choice == "3":
            display_accuracy(accuracy_dict)
        elif choice == "4":
            display_stats(stats)
        elif choice == "5":
            display_clusters(cluster_results)
        elif choice == "6":
            display_regression(regression_results)
        elif choice == "7":
            print("\n  Generating graphs...")
            plot_choice_frequencies(freq_dict)
            plot_transition_heatmaps(transition_matrices)
            plot_prediction_accuracy(accuracy_dict)
            plot_entropy(stats)
            plot_trends(df, regression_results)
            print("\n  All 5 graphs saved!")
        elif choice == "8":
            export_to_json(df, JSON_FILE)
        elif choice == "9":
            display_frequencies(freq_dict)
            display_transitions(transition_matrices)
            display_accuracy(accuracy_dict)
            display_stats(stats)
            display_clusters(cluster_results)
            display_regression(regression_results)
            print("\n  Generating all graphs...")
            plot_choice_frequencies(freq_dict)
            plot_transition_heatmaps(transition_matrices)
            plot_prediction_accuracy(accuracy_dict)
            plot_entropy(stats)
            plot_trends(df, regression_results)
            display_conclusion(accuracy_dict)
            print("\n  5 graphs saved to the current folder.")
        elif choice == "0":
            print("\n  Thanks for exploring! Goodbye.")
            running = False
        else:
            print("  Invalid choice. Please enter a number 0-9.")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main entry point — loads data, runs all analyses, starts the menu."""

    print("=" * 60)
    print("  CAN AI PREDICT HUMAN RANDOMNESS?")
    print("  Rock-Paper-Scissors Pattern Analysis")
    print("=" * 60)

    # --- Load and validate data ---
    print("\n[1] Loading and cleaning data...")
    df = load_csv(CSV_FILE)
    if df is None:
        print("  Cannot continue without data. Exiting.")
        return

    print(f"  Loaded {len(df)} rows for {df['user'].nunique()} users.")
    print("\n[2] Validating data...")
    validate_data(df)

    # --- Run all analyses up front ---
    print("\n[3] Running analyses...")
    freq_dict = compute_choice_frequencies(df)
    print("  Choice frequencies computed.")

    transition_matrices = build_transition_matrix(df)
    print("  Transition matrices built.")

    accuracy_dict = calculate_prediction_accuracy(df, transition_matrices)
    print("  Classification predictions tested.")

    stats = compute_stats(df)
    print("  Summary statistics calculated.")

    cluster_results = cluster_users_kmeans(freq_dict, k=2)
    print("  K-Means clustering complete.")

    regression_results = regression_analysis(df)
    print("  Linear regression complete.")

    print("\n  All analyses ready!")

    # --- Launch interactive menu ---
    run_interactive(df, freq_dict, transition_matrices, accuracy_dict,
                    stats, cluster_results, regression_results)


if __name__ == "__main__":
    main()
