"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's
gone the moment you scroll past it. This module writes the full report
(CV fold table, out-of-fold classification report, fairness audit) to a
timestamped file in `results/` instead, along with the settings that
produced it -- model, CV design, and the locked test set's size/seed (so
you can always tell, from the file alone, that the test set was never
touched by this run).
"""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    model_cfg = config["model"]
    cv_cfg = config["cv"]
    test_cfg = config["test_set"]

    header = "\n".join([
        f"Run: {timestamp}",
        f"Model: {model_cfg['type']}  params: {model_cfg.get('params', {})}",
        f"CV: {cv_cfg['n_splits']}-fold stratified "
        f"(shuffle={cv_cfg.get('shuffle', True)}, random_state={cv_cfg.get('random_state')}, "
        f"scoring={cv_cfg.get('scoring', 'accuracy')})",
        f"Locked test set: size={test_cfg['size']}, random_state={test_cfg['random_state']} "
        f"-- never touched by this run",
        "=" * 60,
        "",
    ]) + "\n"

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
