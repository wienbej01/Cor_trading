#!/usr/bin/env python3
"""
Wrapper script to run the backtest with proper Python path.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now run the actual backtest script
from src.run_backtest import cli

if __name__ == "__main__":
    cli()