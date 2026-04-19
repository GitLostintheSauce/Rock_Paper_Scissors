# Can AI Predict Human Randomness?

## Rock-Paper-Scissors Pattern Analysis

**ENGR 010 Group Project - Computer Science Option, Spring 2026**
**Team Members:** Dana, Jarin, Eleanor, Krew

---

## Project Overview

This project analyzes rock-paper-scissors choice patterns to answer one question: **are humans actually random?** We collected 25 rounds of rock-paper-scissors choices from 4 people, then used Python to find patterns, build a prediction model, and test how well it works.

**Spoiler:** Our model predicts the next move with **61.5% accuracy** — nearly double the 33.3% you'd get from random guessing. Humans are not as random as they think!

---

## How to Run

### Option A — Interactive CLI menu

```bash
cd Rock_Paper_Scissors
python3 rps_analysis.py
```

An interactive menu will appear. Type a number (1-9) to view different analyses, or press 9 to run everything at once.

### Option B — Static web dashboard (GitHub Pages)

```bash
cd Rock_Paper_Scissors
python3 build_dashboard.py
```

This regenerates the five PNG graphs and writes a self-contained `index.html`. Open it in any browser — every graph, table, and the full requirements checklist are on one page.

To publish it online for free:

1. Push the repo to GitHub (already done for this project).
2. On GitHub: **Settings → Pages → Source: Deploy from a branch → Branch: `main`, Folder: `/ (root)`**.
3. GitHub gives you a public URL like `https://<username>.github.io/Rock_Paper_Scissors/` — the dashboard is hosted for free, never sleeps, and stays up as long as the repo exists.

The `.nojekyll` file in the repo root tells GitHub Pages to skip Jekyll processing and just serve the files as-is.

---

## Project Structure

```
Rock_Paper_Scissors/
├── rps_analysis.py          # CLI entry point — interactive menu
├── build_dashboard.py       # Builds the static web dashboard (index.html)
├── data_loader.py           # CSV / JSON / TXT loading + cleaning + export
├── analysis.py              # Algorithms: classification, clustering, regression, stats
├── visualizations.py        # All matplotlib plot functions
├── rps_data.csv             # Raw data (100 rows, 4 users x 25 rounds)
├── index.html               # Generated static dashboard (GitHub Pages entry)
├── .nojekyll                # Tells GitHub Pages to serve files as-is
├── README.md                # This file
├── graph1_choice_frequencies.png
├── graph2_transition_heatmaps.png
├── graph3_prediction_accuracy.png
├── graph4_entropy.png
└── graph5_trends.png
```

---

## How It Works (Step by Step)

### 1. Load and Clean the Data
We read `rps_data.csv` using pandas. The cleaning step strips whitespace, lowercases all text, drops missing values, and removes any invalid choices. This ensures our analysis isn't thrown off by typos or formatting issues.

### 2. Choice Frequency Analysis
We count how many times each player picks rock, paper, and scissors. This reveals biases — for example, Dana picks rock 52% of the time, while Krew and Jarin both favor paper at 44%.

### 3. Transition Matrix (Pattern Detection)
For each player, we build a 3x3 matrix showing: "after move X, what move comes next?" This is the heart of the project. For example, Jarin picks paper after rock 100% of the time — a very strong pattern.

### 4. Prediction (Classification)
Using the transition matrix, we predict each player's next move based on their previous move. We pick whichever next move has the highest probability.

### 5. Accuracy Testing
We test the prediction on the actual data (rounds 2-25) and calculate what percentage of predictions were correct.

### 6. Clustering (K-Means)
We group players by play style. Each player becomes a 3D point (% rock, % paper, % scissors). K-Means groups them into clusters. Result: Dana is in her own cluster (rock-heavy), while Jarin, Eleanor, and Krew form a paper-favoring cluster.

### 7. Regression (Trend Detection)
We use linear regression to check if players' habits change over time. For example, Eleanor's rock usage increases over time (slope = +0.0185) while her paper usage decreases (slope = -0.0262).

### 8. Visualizations
Five graphs are generated (see the Graphs section below).

---

## Graphs

| Graph | File | Type | What It Shows |
|-------|------|------|---------------|
| 1 | `graph1_choice_frequencies.png` | Grouped bar chart | How often each player picks each move |
| 2 | `graph2_transition_heatmaps.png` | Heatmap (2x2 grid) | Transition probabilities per player |
| 3 | `graph3_prediction_accuracy.png` | Bar chart + baseline | Prediction accuracy vs. 33.3% random |
| 4 | `graph4_entropy.png` | Bar chart + baseline | How random each player is (entropy) |
| 5 | `graph5_trends.png` | Line chart (2x2 grid) | Running choice percentages over rounds |

---

## Requirements Checklist

Below is every requirement from the ENGR 010 project description, paired with the exact file and function in our code that satisfies it. Every row links a requirement to concrete evidence.

### 1. Data Management

| Requirement | Where It's Met |
|-------------|---------------|
| Imports data from **CSV** | `data_loader.py` -> `load_csv()` — `pd.read_csv`, then `_clean_dataframe()` |
| Imports data from **JSON** | `data_loader.py` -> `load_json()` — accepts a path or buffer, then `_clean_dataframe()` |
| Imports data from **TXT** | `data_loader.py` -> `load_txt()` — auto-detects delimiter (tab / space / comma), then `_clean_dataframe()` |
| Unified dispatcher | `data_loader.py` -> `load_any(filepath, filename)` — picks the right loader by file extension |
| Data cleaning and preprocessing | `data_loader.py` -> `_clean_dataframe()` strips whitespace, lowercases choices, drops nulls, filters invalid rows |
| Stores and organizes datasets | pandas DataFrames (raw data) + nested dicts (per-user results) + NumPy arrays (matrices, cluster points) |
| Handles missing values | `data_loader.py` -> `_clean_dataframe()` calls `dropna()` and reports the count removed |
| Handles outliers / bad data | `data_loader.py` -> `_clean_dataframe()` filters anything outside `VALID_CHOICES` and reports the count removed |

### 2. Analysis Features — Algorithms (3 required, we implement 3)

| Algorithm | Where It's Met |
|-----------|---------------|
| **Classification** | `analysis.py` -> `predict_next_move()` classifies the next move (rock/paper/scissors) from the transition matrix; `calculate_prediction_accuracy()` scores the classifier against the real data |
| **Clustering (K-Means)** | `analysis.py` -> `cluster_users_kmeans()` — manual K-Means implemented with NumPy (assign-nearest-center + update-means, iterated until convergence) |
| **Regression** | `analysis.py` -> `regression_analysis()` — least-squares linear regression for each user × choice to detect trends over rounds |

| Other Analysis Requirement | Where It's Met |
|----------------------------|---------------|
| Calculates statistical metrics | `analysis.py` -> `compute_stats()` — entropy, longest streak, most/least common, round counts |
| Identifies patterns & correlations | `analysis.py` -> `build_transition_matrix()` — 3×3 per-user move-to-move probability matrices (this is the "correlation matrix" of move pairs) |
| Performs comparative analysis | `rps_analysis.py` display functions and `index.html` tables compare every player side-by-side |

### 3. Visualization Features

| Requirement | Where It's Met |
|-------------|---------------|
| Static plots of data distributions | `visualizations.py` -> `plot_choice_frequencies()` — grouped bar chart of choice distribution per player (graph 1) |
| Correlation matrices | `visualizations.py` -> `plot_transition_heatmaps()` — 3×3 heatmap grid showing move-to-move correlations (graph 2) |
| Trend analysis charts | `visualizations.py` -> `plot_trends()` — line chart of running percentages vs. round (graph 5) |
| Additional static plots | `plot_prediction_accuracy()` (graph 3) and `plot_entropy()` (graph 4) — bar charts with reference baselines |
| Interactive dashboard | **`index.html`** (generated by `build_dashboard.py`) — web dashboard with every graph, every table, and the requirements map, publishable free to GitHub Pages; plus the CLI menu in `rps_analysis.py` -> `run_interactive()` |

### 4. Technical Requirements

#### Variables and Data Types

| Requirement | Where It's Met |
|-------------|---------------|
| Numerical data types | NumPy arrays for matrices and cluster points; floats for probabilities, slopes, and entropy across `analysis.py` |
| Clear variable naming | `transition_matrices`, `accuracy_dict`, `frequency_dict`, `choices_list`, `cluster_results`, `regression_results` — every name describes what it holds |
| Constants | `VALID_CHOICES` and `USERS` (ALL_CAPS) in `analysis.py`, `data_loader.py`, `visualizations.py`; `CSV_FILE`, `JSON_FILE`, `OUTPUT_FILE` in the entry-point files |

#### Control Structures

| Requirement | Where It's Met |
|-------------|---------------|
| if / elif / else | `rps_analysis.py` -> `run_interactive()` menu dispatch; `display_accuracy()` verdicts; `analysis.py` -> `regression_analysis()` classifies slope as increasing / decreasing / stable |
| for loop | `analysis.py` -> every algorithm iterates `USERS` and `VALID_CHOICES`; `compute_choice_frequencies()`, `calculate_prediction_accuracy()`, `regression_analysis()`, `compute_stats()` |
| while loop | `analysis.py` -> `build_transition_matrix()` walks consecutive pairs; `cluster_users_kmeans()` runs K-Means iterations until convergence; `rps_analysis.py` -> `run_interactive()` loops the menu |
| try / except | `data_loader.py` -> `load_csv`, `load_json`, `load_txt` catch `FileNotFoundError`; `rps_analysis.py` -> `run_interactive()` catches `EOFError` for non-interactive runs |

#### Functions and Organization

| Requirement | Where It's Met |
|-------------|---------------|
| Minimum 5 custom functions | 25+ documented functions spanning `data_loader.py`, `analysis.py`, `visualizations.py`, `rps_analysis.py`, and `build_dashboard.py` |
| Parameters and return values | Every function declares named parameters and returns a typed result — see the docstrings |
| Docstrings | Every public function has a docstring with a Parameters and Returns section |
| Multiple Python files | 5 files: `rps_analysis.py`, `build_dashboard.py`, `data_loader.py`, `analysis.py`, `visualizations.py` |

#### Data Structures

| Requirement | Where It's Met |
|-------------|---------------|
| Lists (time-series) | `analysis.py` -> `choices_list = list(user_data["choice"])` in `build_transition_matrix()`, `calculate_prediction_accuracy()`, `compute_stats()` — the per-user move sequence |
| Dictionaries (parameters) | `frequency_dict`, `accuracy_dict`, `stats`, `regression_results`, `cluster_results` — all nested dictionaries keyed by player |
| NumPy arrays (signal processing) | `analysis.py` -> `np.zeros((3, 3))` transition matrices; `cluster_users_kmeans()` uses NumPy arrays for data points, centers, and distances; `regression_analysis()` uses NumPy arrays for the x/y vectors |

#### Scientific Computing

| Requirement | Where It's Met |
|-------------|---------------|
| NumPy signal processing | Transition-matrix normalization, `np.argmax` classification, `np.sqrt(np.sum(...))` Euclidean distances in K-Means, least-squares sums in `regression_analysis()` |
| Pandas data manipulation | `data_loader.py` loads + cleans with pandas; `analysis.py` filters and sorts DataFrames in every function |
| 3+ Matplotlib plot types | **Bar charts** (graphs 1, 3, 4), **heatmap** (graph 2), **line chart** (graph 5) — three distinct plot types |
| Statistical analysis | Shannon entropy, prediction accuracy percentages, least-squares regression slopes, frequency distributions, longest-streak detection |

### 5. Creativity

| Feature | Description |
|---------|-------------|
| Entropy analysis | Measures how "random" each player actually is using information theory (graph 4) |
| Hosted web dashboard | `build_dashboard.py` generates a self-contained `index.html` we publish free on GitHub Pages — no server required |
| Multi-format data import | One unified `load_any()` handles CSV, JSON, and TXT uploads with auto-detected delimiters |
| CLI interactive menu | Users can explore individual analyses without re-running the whole program |
| JSON export | Data can be exported to JSON format for use in other tools |
| K-Means from scratch | We implemented K-Means manually with NumPy instead of using a library — more educational |
| Trend visualization | Running percentage line charts show how habits evolve over 25 rounds (graph 5) |

---

## Key Findings

- **Dana** heavily favors rock (52%) and is in her own cluster — the "rock player"
- **Jarin** is the most predictable (75% accuracy) — after rock he ALWAYS picks paper
- **Krew** is also very predictable (70.8%) — after scissors he ALWAYS picks paper
- **Eleanor** is the most random (entropy 1.554/1.585) but still 50% predictable
- **Eleanor** is the only player with a significant trend: more rock, less paper over time
- **Overall:** 61.5% prediction accuracy — nearly 2x better than random guessing

---

## Libraries Used

| Library | What It Does | Where We Use It |
|---------|-------------|-----------------|
| **pandas** | Loads and organizes data in tables (DataFrames) | `data_loader.py` for CSV / JSON / TXT handling; `analysis.py` for filtering |
| **numpy** | Fast math on arrays and matrices | `analysis.py` for transition matrices, K-Means, regression |
| **matplotlib** | Creates graphs and charts | `visualizations.py` for all 5 graphs |
| **json** | Reads and writes JSON files | `data_loader.py` for JSON import / export |
| **html** (stdlib) | Safe HTML escaping for the dashboard | `build_dashboard.py` |
