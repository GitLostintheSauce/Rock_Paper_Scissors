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

```bash
cd Rock_Paper_Scissors
python3 rps_analysis.py
```

An interactive menu will appear. Type a number (1-9) to view different analyses, or press 9 to run everything at once.

---

## Project Structure

```
Rock_Paper_Scissors/
├── rps_analysis.py          # Main file — run this one (interactive menu)
├── data_loader.py           # Data loading, cleaning, validation, export
├── analysis.py              # All analysis algorithms and statistics
├── visualizations.py        # All matplotlib graphing functions
├── rps_data.csv             # Raw data (100 rows, 4 users x 25 rounds)
├── README.md                # This file
├── graph1_choice_frequencies.png
├── graph2_transition_heatmaps.png
├── graph3_prediction_accuracy.png
├── graph4_entropy.png
├── graph5_trends.png
├── cs-project Spring 2026.md.pdf    # Assignment description
└── Rock Paper Scissors data - Sheet2.pdf  # Original data sheet
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

Below is every requirement from the ENGR 010 project description and exactly where it is met in our code.

### 1. Data Management

| Requirement | Where It's Met |
|-------------|---------------|
| Imports data from CSV | `data_loader.py` -> `load_csv()` (line 18) |
| Imports data from JSON | `data_loader.py` -> `load_json()` (line 60) |
| Data cleaning and preprocessing | `data_loader.py` -> `load_csv()` strips whitespace, lowercases, drops nulls (lines 34-50) |
| Stores and organizes datasets | Data stored in pandas DataFrames; results in dictionaries and numpy arrays |
| Handles missing values and outliers | `data_loader.py` -> `load_csv()` uses `dropna()` (line 42) and filters invalid choices (lines 46-50) |

### 2. Analysis Features — Algorithms (3 required)

| Algorithm | Where It's Met |
|-----------|---------------|
| **Classification** | `analysis.py` -> `predict_next_move()` (line 96) classifies the next move into rock/paper/scissors based on transition probabilities |
| **Clustering (K-Means)** | `analysis.py` -> `cluster_users_kmeans()` (line 134) groups players by play style using manual K-Means with numpy |
| **Regression** | `analysis.py` -> `regression_analysis()` (line 196) performs least-squares linear regression to detect trends over rounds |

| Requirement | Where It's Met |
|-------------|---------------|
| Calculates statistical metrics | `analysis.py` -> `compute_stats()` (line 237) calculates entropy, streaks, most/least common |
| Identifies patterns and correlations | `analysis.py` -> `build_transition_matrix()` (line 52) finds move-to-move patterns |
| Performs comparative analysis | `rps_analysis.py` -> `display_accuracy()` and `display_clusters()` compare all 4 users |

### 3. Visualization Features

| Requirement | Where It's Met |
|-------------|---------------|
| Static plots of data distributions | `visualizations.py` -> `plot_choice_frequencies()` (line 18) — bar chart of choice distribution |
| Correlation matrices | `visualizations.py` -> `plot_transition_heatmaps()` (line 55) — 3x3 heatmaps showing move correlations |
| Trend analysis charts | `visualizations.py` -> `plot_trends()` (line 173) — line chart of running percentages over time |
| Interactive dashboard | `rps_analysis.py` -> `show_menu()` and `run_interactive()` (lines 157-210) — text-based interactive menu for exploring all data |

### 4. Technical Requirements

#### Variables and Data Types

| Requirement | Where It's Met |
|-------------|---------------|
| Numerical data types | numpy arrays for matrices, floats for probabilities and slopes throughout `analysis.py` |
| Clear variable naming | All variables use descriptive names: `transition_matrices`, `accuracy_dict`, `choices_list`, etc. |
| Constants | `analysis.py` lines 10-11: `VALID_CHOICES` and `USERS` defined in ALL_CAPS |

#### Control Structures

| Requirement | Where It's Met |
|-------------|---------------|
| if/elif/else | `rps_analysis.py` -> `run_interactive()` (lines 174-210) uses if/elif/else for menu choices; `display_accuracy()` uses if/elif/else for verdicts |
| for loop | Used throughout — e.g., `analysis.py` -> `compute_choice_frequencies()` loops through users and choices |
| while loop | `analysis.py` -> `build_transition_matrix()` (line 67) uses while loop for consecutive pairs; `cluster_users_kmeans()` (line 165) uses while loop for K-Means iterations; `rps_analysis.py` -> `run_interactive()` uses while loop for menu |
| try/except | `data_loader.py` -> `load_csv()` (line 24) handles FileNotFoundError; `rps_analysis.py` -> `run_interactive()` handles EOFError |

#### Functions and Organization

| Requirement | Where It's Met |
|-------------|---------------|
| Minimum 5 custom functions | We have 20+ functions across 4 files |
| Parameters and return values | Every function uses parameters and returns results — see docstrings |
| Docstrings | Every function has a docstring explaining what it does, its parameters, and return values |
| Multiple Python files | 4 files: `rps_analysis.py`, `data_loader.py`, `analysis.py`, `visualizations.py` |

#### Data Structures

| Requirement | Where It's Met |
|-------------|---------------|
| Lists | `analysis.py` -> `choices_list` stores time-series of moves (e.g., line 60); `probs` list in `compute_stats()` |
| Dictionaries | `analysis.py` -> `frequency_dict`, `accuracy_dict`, `stats` are all nested dictionaries storing per-user results |
| NumPy arrays | `analysis.py` -> `build_transition_matrix()` uses `np.zeros((3, 3))` (line 63); `cluster_users_kmeans()` uses numpy arrays for data points and centers |

#### Scientific Computing

| Requirement | Where It's Met |
|-------------|---------------|
| NumPy | Used for transition matrices, K-Means distances, regression calculations, and `argmax` predictions |
| Pandas | `data_loader.py` loads and cleans data with pandas; `analysis.py` uses DataFrame filtering throughout |
| 3+ Matplotlib plot types | Bar chart (graph 1, 3, 4), Heatmap (graph 2), Line chart (graph 5) — 3 distinct types |
| Statistical analysis | Entropy calculation, prediction accuracy, regression slopes, frequency distributions |

### 5. Creativity

| Feature | Description |
|---------|-------------|
| Entropy analysis | Measures how "random" each player actually is using information theory (graph 4) |
| Interactive menu | Users can explore individual analyses without re-running the whole program |
| JSON export | Data can be exported to JSON format for use in other tools |
| K-Means from scratch | We implemented K-Means manually with numpy instead of using a library — more educational |
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
| **pandas** | Loads and organizes data in tables (DataFrames) | `data_loader.py` for CSV/JSON handling |
| **numpy** | Fast math on arrays and matrices | `analysis.py` for transition matrices, K-Means, regression |
| **matplotlib** | Creates graphs and charts | `visualizations.py` for all 5 graphs |
| **json** | Reads and writes JSON files | `data_loader.py` for JSON import/export |
