"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's
gone the moment you scroll past it or close the window. This module writes
the full report (accuracy, classification report, fairness table, and --
since week 3 -- the before/after comparison across every dataset variant
and model that was run) to a timestamped file in `results/` instead, so
you can open it again later, or compare two runs side by side after
changing something in config.yaml.
"""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str, run_label: str = "") -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.

    `run_label` is free text describing what this run covers (e.g. the
    list of dataset variants x models compared) -- since week 3 a single
    run can cover more than one model, so the header no longer assumes a
    single `config["model"]`.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    header_lines = [f"Run: {timestamp}"]
    if run_label:
        header_lines.append(run_label)
    header_lines.append(
        f"Test size: {config['split']['test_size']}  "
        f"random_state: {config['split']['random_state']}"
    )
    header_lines.append("=" * 60)
    header_lines.append("")
    header = "\n".join(header_lines) + "\n"

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
