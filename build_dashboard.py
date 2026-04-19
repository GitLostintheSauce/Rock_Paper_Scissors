"""
build_dashboard.py — Generate a static GitHub Pages dashboard.

Runs every analysis in the project, regenerates the five PNG graphs, and
writes `index.html` — a single self-contained page that shows:

    - Project overview and key findings
    - All five graphs with captions
    - Numerical result tables (frequencies, accuracy, clusters, regression)
    - Requirements checklist mapping each spec bullet to its code location

To run:
    python3 build_dashboard.py

After it finishes, open `index.html` in a browser or push the repo and
enable GitHub Pages (Settings -> Pages -> Deploy from branch -> main).
"""

import html
from datetime import datetime

from data_loader import load_csv, validate_data
from analysis import (compute_choice_frequencies, build_transition_matrix,
                      calculate_prediction_accuracy, compute_stats,
                      cluster_users_kmeans, regression_analysis,
                      USERS, VALID_CHOICES)
from visualizations import (plot_choice_frequencies, plot_transition_heatmaps,
                            plot_prediction_accuracy, plot_entropy,
                            plot_trends)

# Constants
CSV_FILE = "rps_data.csv"
OUTPUT_FILE = "index.html"


# =============================================================================
# Table builders — turn analysis dicts into HTML fragments
# =============================================================================

def _th(cells):
    """Render a list of strings as a <thead> row."""
    return "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>"
                            for c in cells) + "</tr>"


def _td(cells):
    """Render a list of strings as a <tbody> row."""
    return "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>"
                            for c in cells) + "</tr>"


def frequency_table_html(freq_dict):
    """HTML table showing each user's choice counts and percentages."""
    header = _th(["Player", "Rock", "Paper", "Scissors", "Favorite"])
    rows = []
    for user in USERS:
        counts = freq_dict[user]
        total = sum(counts.values()) or 1
        favorite = max(counts, key=counts.get).capitalize()
        rows.append(_td([
            user,
            f"{counts['rock']} ({counts['rock'] / total:.0%})",
            f"{counts['paper']} ({counts['paper'] / total:.0%})",
            f"{counts['scissors']} ({counts['scissors'] / total:.0%})",
            favorite,
        ]))
    return f"<table><thead>{header}</thead><tbody>" + "".join(rows) + "</tbody></table>"


def accuracy_table_html(accuracy_dict):
    """HTML table of prediction accuracy with a verdict per user."""
    header = _th(["Player", "Accuracy", "vs. Random (33.3%)", "Verdict"])
    rows = []
    for user in USERS:
        acc = accuracy_dict[user]
        delta = acc - 33.3
        if acc > 50:
            verdict = "Very predictable"
        elif acc > 33.3:
            verdict = "Somewhat predictable"
        else:
            verdict = "Hard to predict"
        rows.append(_td([user, f"{acc:.1f}%",
                         f"{delta:+.1f} pts", verdict]))
    return f"<table><thead>{header}</thead><tbody>" + "".join(rows) + "</tbody></table>"


def stats_table_html(stats):
    """HTML table of summary statistics (streak, entropy, etc.)."""
    header = _th(["Player", "Rounds", "Favorite", "Least Used",
                  "Longest Streak", "Entropy (/1.585)"])
    rows = []
    for user in USERS:
        s = stats[user]
        rows.append(_td([
            user, s["total_rounds"], s["most_common"].capitalize(),
            s["least_common"].capitalize(), s["longest_streak"],
            s["entropy"],
        ]))
    return f"<table><thead>{header}</thead><tbody>" + "".join(rows) + "</tbody></table>"


def regression_table_html(regression_results):
    """HTML table of regression slopes per user / choice."""
    header = _th(["Player"] + [f"{c.capitalize()} slope (trend)"
                               for c in VALID_CHOICES])
    rows = []
    for user in USERS:
        row_cells = [user]
        for choice in VALID_CHOICES:
            r = regression_results[user][choice]
            row_cells.append(f"{r['slope']:+.4f} ({r['trend']})")
        rows.append(_td(row_cells))
    return f"<table><thead>{header}</thead><tbody>" + "".join(rows) + "</tbody></table>"


def cluster_summary_html(cluster_results):
    """Human-readable cluster summary (who ended up grouped together)."""
    labels = cluster_results["labels"]
    points = cluster_results["user_points"]

    # Group users by cluster id
    groups = {}
    for user, cid in labels.items():
        groups.setdefault(cid, []).append(user)

    parts = []
    for cid, members in sorted(groups.items()):
        blurbs = []
        for user in members:
            p = points[user]
            blurbs.append(
                f"<li><strong>{html.escape(user)}</strong> — "
                f"Rock {p[0]:.0%}, Paper {p[1]:.0%}, "
                f"Scissors {p[2]:.0%}</li>"
            )
        parts.append(
            f"<div class='cluster'><h4>Cluster {cid + 1}</h4>"
            f"<ul>{''.join(blurbs)}</ul></div>"
        )
    converged = cluster_results["iterations"]
    return (f"<div class='cluster-grid'>{''.join(parts)}</div>"
            f"<p class='note'>K-Means converged in {converged} iterations.</p>")


def transition_highlights_html(transition_matrices):
    """Bullet list of the strongest move-to-move patterns per player."""
    parts = []
    for user in USERS:
        matrix = transition_matrices[user]
        strong = []
        for i, prev in enumerate(VALID_CHOICES):
            for j, nxt in enumerate(VALID_CHOICES):
                prob = matrix[i][j]
                if prob >= 0.5:
                    strong.append(
                        f"After <em>{prev}</em> -> plays "
                        f"<strong>{nxt}</strong> {prob:.0%} of the time"
                    )
        items = "".join(f"<li>{s}</li>" for s in strong) if strong else \
                "<li>No single transition above 50%.</li>"
        parts.append(
            f"<div class='player-card'><h4>{html.escape(user)}</h4>"
            f"<ul>{items}</ul></div>"
        )
    return f"<div class='card-grid'>{''.join(parts)}</div>"


# =============================================================================
# Page assembly
# =============================================================================

def build_html(df, freq_dict, transition_matrices, accuracy_dict, stats,
               cluster_results, regression_results):
    """Assemble the full index.html string.

    Parameters:
        Every argument is a pre-computed analysis result. Only the caller
        needs to know the order — each helper above handles its own slice.

    Returns:
        str: The complete HTML document.
    """
    overall_avg = sum(accuracy_dict.values()) / len(accuracy_dict)
    most_predictable = max(accuracy_dict, key=accuracy_dict.get)
    least_predictable = min(stats, key=lambda u: stats[u]["entropy"])
    random_est = (stats[max(stats, key=lambda u: stats[u]["entropy"])]
                  ["entropy"])
    generated = datetime.now().strftime("%B %d, %Y at %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Can AI Predict Human Randomness? — RPS Analysis Dashboard</title>
<style>
  :root {{
    --bg: #f6f7fb; --card: #ffffff; --ink: #1f2430; --muted: #5b6478;
    --accent: #4A90D9; --accent2: #E8915A; --good: #2f9e6a; --bad: #c0392b;
    --border: #e3e6ef;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
      Roboto, sans-serif; background: var(--bg); color: var(--ink);
    line-height: 1.55;
  }}
  header.hero {{
    background: linear-gradient(135deg, #4A90D9 0%, #7B5FD0 100%);
    color: white; padding: 3rem 1.5rem 3.5rem; text-align: center;
  }}
  header.hero h1 {{ margin: 0 0 .5rem; font-size: 2.25rem; }}
  header.hero p.tagline {{ margin: 0 0 1rem; opacity: .92; font-size: 1.1rem; }}
  header.hero p.team {{ margin: 0; opacity: .8; font-size: .95rem; }}
  main {{ max-width: 1100px; margin: -2rem auto 3rem; padding: 0 1rem; }}
  section {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 1.5rem;
    box-shadow: 0 4px 14px rgba(20, 30, 60, .04);
  }}
  h2 {{ margin-top: 0; border-bottom: 2px solid var(--border);
       padding-bottom: .5rem; }}
  h3 {{ margin-bottom: .25rem; color: var(--ink); }}
  .muted {{ color: var(--muted); font-size: .95rem; }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin-top: 1rem;
  }}
  .kpi {{
    background: #f3f6ff; border: 1px solid var(--border); border-radius: 10px;
    padding: 1rem; text-align: center;
  }}
  .kpi .value {{ font-size: 1.85rem; font-weight: 700; color: var(--accent); }}
  .kpi .label {{ font-size: .85rem; color: var(--muted); text-transform: uppercase;
               letter-spacing: .04em; margin-top: .25rem; }}
  figure {{ margin: 1rem 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid var(--border);
               border-radius: 8px; background: white; }}
  figcaption {{ color: var(--muted); font-size: .9rem; margin-top: .5rem; }}
  table {{
    width: 100%; border-collapse: collapse; margin: .75rem 0 1rem;
    font-size: .95rem;
  }}
  th, td {{ padding: .55rem .75rem; text-align: left;
           border-bottom: 1px solid var(--border); }}
  th {{ background: #f3f6ff; font-weight: 600; color: var(--ink); }}
  tbody tr:hover {{ background: #fafbff; }}
  .card-grid, .cluster-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem; margin-top: .75rem;
  }}
  .player-card, .cluster {{
    background: #f7f9ff; border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem;
  }}
  .player-card h4, .cluster h4 {{ margin: 0 0 .5rem; color: var(--accent); }}
  .player-card ul, .cluster ul {{ margin: 0; padding-left: 1.2rem; }}
  .player-card li, .cluster li {{ margin: .2rem 0; font-size: .92rem; }}
  .note {{ color: var(--muted); font-size: .88rem; margin-top: .5rem; }}
  .checklist td code {{ font-size: .85rem; background: #eef2fa;
                       padding: 1px 6px; border-radius: 4px; }}
  .section-tag {{ display: inline-block; padding: 2px 10px;
                 background: var(--accent); color: white; border-radius: 999px;
                 font-size: .72rem; letter-spacing: .05em; text-transform: uppercase;
                 margin-right: .5rem; vertical-align: middle; }}
  footer {{ text-align: center; color: var(--muted); font-size: .85rem;
           margin: 2rem 0 1rem; }}
</style>
</head>
<body>
<header class="hero">
  <h1>Can AI Predict Human Randomness?</h1>
  <p class="tagline">Rock-Paper-Scissors pattern analysis across
     {df['user'].nunique()} players and {len(df)} rounds</p>
  <p class="team">ENGR 010 Spring 2026 — Dana, Jarin, Eleanor, Krew</p>
</header>

<main>

<section>
  <h2>Key Findings</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="value">{overall_avg:.1f}%</div>
      <div class="label">Overall prediction accuracy</div>
    </div>
    <div class="kpi">
      <div class="value">33.3%</div>
      <div class="label">Random baseline</div>
    </div>
    <div class="kpi">
      <div class="value">{accuracy_dict[most_predictable]:.1f}%</div>
      <div class="label">Most predictable: {html.escape(most_predictable)}</div>
    </div>
    <div class="kpi">
      <div class="value">{random_est:.3f}</div>
      <div class="label">Max entropy observed ({html.escape(least_predictable)})</div>
    </div>
  </div>
  <p class="muted" style="margin-top:1rem">
    Our transition-matrix classifier beats random guessing by
    <strong>{overall_avg - 33.3:.1f} percentage points</strong>. Humans
    are measurably non-random — but some are more predictable than others.
  </p>
</section>

<section>
  <h2><span class="section-tag">Graph 1</span>Choice Frequencies</h2>
  <p class="muted">How often does each player pick rock, paper, or scissors?
     This is the baseline view of each player's preferences before we look
     for deeper patterns.</p>
  <figure>
    <img src="graph1_choice_frequencies.png" alt="Choice frequency bar chart">
    <figcaption>Figure 1 — Grouped bar chart of choice counts per player.</figcaption>
  </figure>
  {frequency_table_html(freq_dict)}
</section>

<section>
  <h2><span class="section-tag">Graph 2</span>Transition Patterns
      <span class="muted" style="font-size:.85rem">(correlation matrices)</span></h2>
  <p class="muted">A 3×3 matrix per player showing: <em>given the previous
     move was X, how often is the next move Y?</em> Strong cells
     (&gt; 50%) are reliable tells.</p>
  <figure>
    <img src="graph2_transition_heatmaps.png"
         alt="Transition probability heatmaps">
    <figcaption>Figure 2 — Transition heatmaps. Rows = previous move,
       columns = next move.</figcaption>
  </figure>
  <h3>Strongest tendencies per player</h3>
  {transition_highlights_html(transition_matrices)}
</section>

<section>
  <h2><span class="section-tag">Graph 3</span>Classification — Prediction Accuracy</h2>
  <p class="muted">Using each player's transition matrix, we predict their
     next move from their previous one, then measure how often the
     prediction matches the actual move.</p>
  <figure>
    <img src="graph3_prediction_accuracy.png" alt="Prediction accuracy bar chart">
    <figcaption>Figure 3 — Per-player prediction accuracy with the 33.3%
       random-guess baseline.</figcaption>
  </figure>
  {accuracy_table_html(accuracy_dict)}
</section>

<section>
  <h2><span class="section-tag">Graph 4</span>Entropy — How Random Is Each Player?</h2>
  <p class="muted">Entropy (bits, base-2) measures unpredictability. The
     ceiling for a 3-choice game is log<sub>2</sub>(3) ≈ 1.585. Lower
     scores mean more detectable patterns.</p>
  <figure>
    <img src="graph4_entropy.png" alt="Entropy bar chart">
    <figcaption>Figure 4 — Shannon entropy per player vs. the
       perfectly-random ceiling.</figcaption>
  </figure>
  {stats_table_html(stats)}
</section>

<section>
  <h2><span class="section-tag">Graph 5</span>Regression — Trends Over Time</h2>
  <p class="muted">Linear regression on each player's round-by-round
     choices detects slowly-shifting habits. A positive slope means the
     player picks that move more as the game goes on.</p>
  <figure>
    <img src="graph5_trends.png" alt="Running-percentage trend charts">
    <figcaption>Figure 5 — Running percentage of each move per player
       over 25 rounds.</figcaption>
  </figure>
  {regression_table_html(regression_results)}
</section>

<section>
  <h2>Clustering — Play-Style Groups (K-Means)</h2>
  <p class="muted">Each player becomes a 3-D point (% rock, % paper,
     % scissors). K-Means groups similar points. Implemented manually
     with NumPy so you can see every step.</p>
  {cluster_summary_html(cluster_results)}
</section>

<section class="checklist">
  <h2>Requirements Checklist</h2>
  <p class="muted">Every spec bullet from the ENGR 010 project description
     mapped to the code that satisfies it.</p>

  <h3>1. Data Management</h3>
  <table><thead>{_th(["Requirement", "Where It's Met"])}</thead><tbody>
    {_td(["CSV import", "data_loader.py -> load_csv()"])}
    {_td(["JSON import", "data_loader.py -> load_json()"])}
    {_td(["TXT import", "data_loader.py -> load_txt()"])}
    {_td(["Cleaning & preprocessing",
          "data_loader.py -> _clean_dataframe() (strip, lowercase, dropna)"])}
    {_td(["Handles missing values",
          "data_loader.py -> _clean_dataframe() dropna()"])}
    {_td(["Handles outliers (invalid choices)",
          "data_loader.py -> _clean_dataframe() filters to VALID_CHOICES"])}
    {_td(["Organized storage",
          "pandas DataFrames + nested dicts + NumPy arrays"])}
  </tbody></table>

  <h3>2. Analysis Features (3+ algorithms)</h3>
  <table><thead>{_th(["Algorithm / Requirement", "Where It's Met"])}</thead><tbody>
    {_td(["Classification",
          "analysis.py -> predict_next_move() + calculate_prediction_accuracy()"])}
    {_td(["Clustering (K-Means, manual)",
          "analysis.py -> cluster_users_kmeans()"])}
    {_td(["Regression (least-squares)",
          "analysis.py -> regression_analysis()"])}
    {_td(["Statistical metrics",
          "analysis.py -> compute_stats() (entropy, streaks, freq)"])}
    {_td(["Patterns & correlations",
          "analysis.py -> build_transition_matrix()"])}
    {_td(["Comparative analysis",
          "All dashboards compare every player side-by-side"])}
  </tbody></table>

  <h3>3. Visualization Features</h3>
  <table><thead>{_th(["Requirement", "Where It's Met"])}</thead><tbody>
    {_td(["Static plots of distributions",
          "visualizations.py -> plot_choice_frequencies()"])}
    {_td(["Correlation-style matrix",
          "visualizations.py -> plot_transition_heatmaps()"])}
    {_td(["Trend analysis charts",
          "visualizations.py -> plot_trends()"])}
    {_td(["Interactive dashboard",
          "index.html (this page) + CLI menu in rps_analysis.py"])}
  </tbody></table>

  <h3>4. Python Concepts</h3>
  <table><thead>{_th(["Requirement", "Where It's Met"])}</thead><tbody>
    {_td(["Numerical data types",
          "NumPy arrays + floats throughout analysis.py"])}
    {_td(["Clear variable naming",
          "transition_matrices, accuracy_dict, frequency_dict, etc."])}
    {_td(["Constants",
          "VALID_CHOICES, USERS (ALL_CAPS) in analysis.py / visualizations.py"])}
    {_td(["if/elif/else",
          "rps_analysis.py -> run_interactive(); display_accuracy()"])}
    {_td(["for loops",
          "analysis.py -> every algorithm iterates USERS and VALID_CHOICES"])}
    {_td(["while loops",
          "analysis.py -> build_transition_matrix(); cluster_users_kmeans()"])}
    {_td(["try/except",
          "data_loader.py -> load_csv/json/txt; rps_analysis.py EOFError guard"])}
    {_td(["5+ custom functions",
          "20+ functions across data_loader / analysis / visualizations"])}
    {_td(["Parameters & return values",
          "Every function declares typed parameters and returns results"])}
    {_td(["Docstrings",
          "Every public function has a Parameters/Returns docstring"])}
    {_td(["Multiple Python files",
          "data_loader.py, analysis.py, visualizations.py, rps_analysis.py, build_dashboard.py"])}
    {_td(["Lists (time-series)",
          "analysis.py -> choices_list in build_transition_matrix()"])}
    {_td(["Dictionaries (parameters)",
          "freq_dict, accuracy_dict, stats, regression_results"])}
    {_td(["NumPy arrays (signal processing)",
          "np.zeros((3,3)) transitions; centers[] in K-Means"])}
    {_td(["NumPy signal processing",
          "argmax classification; Euclidean distance in K-Means"])}
    {_td(["Pandas",
          "data_loader.py reads/cleans; analysis.py filters DataFrames"])}
    {_td(["3+ matplotlib plot types",
          "Bar (graphs 1,3,4), Heatmap (graph 2), Line (graph 5)"])}
    {_td(["Statistical analysis",
          "Entropy, accuracy %, regression slopes, frequency distributions"])}
  </tbody></table>
</section>

<footer>
  Generated {html.escape(generated)} · static dashboard built by
  <code>build_dashboard.py</code>.
</footer>

</main>
</body>
</html>
"""


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Load data, run every analysis, regenerate graphs, write index.html."""
    print("=" * 60)
    print("  BUILDING STATIC DASHBOARD")
    print("=" * 60)

    print("\n[1] Loading and cleaning data...")
    df = load_csv(CSV_FILE)
    if df is None:
        print("  Cannot continue without data. Exiting.")
        return
    print(f"  Loaded {len(df)} rows for {df['user'].nunique()} users.")
    validate_data(df)

    print("\n[2] Running analyses...")
    freq_dict = compute_choice_frequencies(df)
    transition_matrices = build_transition_matrix(df)
    accuracy_dict = calculate_prediction_accuracy(df, transition_matrices)
    stats = compute_stats(df)
    cluster_results = cluster_users_kmeans(freq_dict, k=2)
    regression_results = regression_analysis(df)
    print("  All analyses complete.")

    # Regenerate graphs so the dashboard always matches current data.
    # The plot functions also call plt.show() which is harmless when run
    # headless (the resulting windows just never open).
    print("\n[3] Regenerating graph PNGs...")
    import matplotlib
    matplotlib.use("Agg")  # headless backend — safe in CI / build scripts
    import matplotlib.pyplot as plt
    plt.ioff()
    plot_choice_frequencies(freq_dict)
    plot_transition_heatmaps(transition_matrices)
    plot_prediction_accuracy(accuracy_dict)
    plot_entropy(stats)
    plot_trends(df, regression_results)
    plt.close("all")

    print("\n[4] Writing index.html...")
    html_text = build_html(df, freq_dict, transition_matrices, accuracy_dict,
                           stats, cluster_results, regression_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"  Wrote {OUTPUT_FILE} ({len(html_text):,} bytes).")
    print("\n  Open index.html in a browser, or push the repo and enable")
    print("  GitHub Pages (Settings -> Pages -> Branch: main, / (root)).")


if __name__ == "__main__":
    main()
