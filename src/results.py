"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment
you scroll past it. Two things are kept:
  - results/run_<id>.txt : the full report of one run (CV fold table, out-of-fold
    classification report, fairness audit, tuning report) with the settings that produced it;
  - results/runs.csv     : the scoreboard, one row per run, so models compare side by side.
"""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """
    Appends one run (a dict) to the scoreboard CSV (created if missing; columns that a new
    run adds -- e.g. more folds -- are added, not lost) and returns the whole board.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    board = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
    board.to_csv(path, index=False)
    return board


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to `run_<run>.txt` inside `results_dir` (created if it
    doesn't exist yet) and returns the path. `run` is the id main.py also puts in the
    scoreboard, so a row of runs.csv and its report can always be matched.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    model_cfg = config["model"]
    cv_cfg = config["cv"]
    test_cfg = config["test_set"]

    header = "\n".join([
        f"Run: {run}",
        f"Model: {model_cfg['type']}  params: {model_cfg.get('params', {})}",
        f"CV: {cv_cfg['n_splits']}-fold stratified "
        f"(shuffle={cv_cfg.get('shuffle', True)}, random_state={cv_cfg.get('random_state')})",
        f"Metrics: {cv_cfg['metrics']}  |  selection metric: {cv_cfg['selection_metric']}",
        f"Locked test set: size={test_cfg['size']}, random_state={test_cfg['random_state']} "
        f"-- never touched by this run",
        "=" * 60,
        "",
    ]) + "\n"

    with open(path, "w") as f:
        f.write(header + report_text)
    return path
