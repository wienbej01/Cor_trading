# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the "FX-Commodity Correlation Arbitrage Strategy" project.

## Project Overview

This project is a production-grade Python implementation of a mean-reversion trading strategy based on the correlation and cointegration between foreign exchange (FX) and commodity pairs. The primary focus is on `USD/CAD` vs. `WTI` crude oil and `USD/NOK` vs. `Brent` crude oil.

The core of the strategy is to identify temporary deviations in the historical price relationships of these pairs and take positions that profit from their expected reversion to the mean.

### Key Technologies

*   **Programming Language:** Python 3.11+
*   **Core Libraries:**
    *   `pandas` and `numpy` for data manipulation and analysis.
    *   `statsmodels` and `scikit-learn` for statistical modeling and machine learning.
    *   `yfinance` for downloading historical market data.
    *   `PyYAML` for configuration management.
    *   `click` for building the command-line interface (CLI).
    *   `loguru` for logging.
*   **Development Environment:** The project uses `venv` for virtual environment management and `pip` for package installation.

### Architecture

The project follows a modular architecture, with distinct components for different functionalities:

*   **`src/core`:** Core components like configuration management.
*   **`src/data`:** Data loading and processing from sources like Yahoo Finance.
*   **`src/features`:** Feature engineering, including indicators, cointegration analysis, and regime detection.
*   **`src/strategy`:** The core trading strategy logic, including signal generation and position sizing.
*   **`src/ml`:** Machine learning models for signal filtering and enhancement (currently includes stubs and an ensemble model).
*   **`src/backtest`:** The backtesting engine, which simulates the strategy's performance on historical data and calculates performance metrics.
*   **`src/exec`:** Stubs for broker integration (e.g., Interactive Brokers, OANDA).
*   **`configs`:** YAML files for configuring trading pairs and strategy parameters.
*   **`src/run_backtest.py`:** The main entry point for running backtests via the CLI.

## Building and Running

### Setup

1.  **Clone the repository.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running Backtests

The primary way to interact with the project is through the `src/run_backtest.py` script.

**Basic Usage:**

```bash
python src/run_backtest.py run --pair <pair_name> --start <YYYY-MM-DD> --end <YYYY-MM-DD>
```

**Example:**

```bash
python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15
```

### CLI Commands

*   `run`: Run a backtest for a specified pair and date range.
*   `list-pairs`: List all available trading pairs defined in `configs/pairs.yaml`.
*   `show-config`: Display the configuration for a specific pair.

## Development Conventions

*   **Code Style:** The project follows the PEP 8 style guide.
*   **Typing:** Type hints are used for all public functions.
*   **Documentation:** The code includes comprehensive docstrings.
*   **Configuration:** The strategy is highly configurable through the `configs/pairs.yaml` file. This allows for easy tuning of parameters without modifying the code.
*   **Testing:** The `tests` directory contains unit and integration tests. `pytest` is the testing framework.
*   **Modularity:** The code is organized into logical modules, promoting code reuse and maintainability.
