import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Union


def log_run(csv_path: Union[str, Path], row: Dict) -> None:
    """
    Appends one row of experiment results to a CSV log file, creating
    the file (and its header) the first time it's called.

    Every call adds a "timestamp" column automatically, so the log
    keeps a full history of every training run (one row per
    architecture per execution of main.py), not just the latest one.

    Parameters
    ----------
    csv_path : path to the CSV log file (created if it doesn't exist).
    row      : flat dict of column_name -> value for this run
               (e.g. {"model_variant": "resnet", "val_f1": 0.71, ...}).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}

    file_exists = csv_path.exists()
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
