#!/usr/bin/env python

import subprocess
from pathlib import Path


def run_scripts(files_to_run: str):
    """
    Execute specific scripts based on their paths.

    Args:
        files_to_run (str): List of scripts to execute.
                            Each line should specify:
                            - A single script file (absolute or relative path)
                            - Comments (lines starting with # are ignored)
    """
    print("Starting execution of specified scripts...")

    for line in files_to_run.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):  # Ignore empty lines and comments
            continue

        script_path = Path(line)
        if script_path.exists():
            print(f"Running script: {script_path}")
            try:
                # Execute the script using Python subprocess
                subprocess.run(["python", str(script_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing {script_path}: {e}")
                print("Stopping execution due to an error.")
                raise
        else:
            print(f"Script not found: {script_path}")
            print("Stopping execution due to missing script.")
            raise FileNotFoundError(f"Script not found: {script_path}")


# Specify the full or relative paths of the files you want to run
files_to_run = """
/home/iyad/V1_DIR/V1/different_rebalancing_dates/makeBondsData.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates/makeStocksData.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates/combineStocksAndBondsForBacktest.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates/saveDiffRebalDatesStksAndRets.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates_V/makeBondsData.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates_V/makeStocksData.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates_V/combineStocksAndBondsForBacktest.py
/home/iyad/V1_DIR/V1/different_rebalancing_dates_V/saveDiffRebalDatesStksAndRets.py
# /home/iyad/V1_DIR/V1/different_rebalancing_dates/withPicassoDaily.py
"""

# Run the specified scripts
if __name__ == "__main__":
    try:
        run_scripts(files_to_run)
    except Exception as e:
        print(f"Execution halted: {e}")
