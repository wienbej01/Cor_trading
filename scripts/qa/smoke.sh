#!/bin/bash
set -e  # Exit on any error

echo "=== Cor Trading System Smoke Test ==="
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"

# 1. Environment check
echo "1. Checking Python environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
python --version
pip list | grep -E "(pandas|numpy|scikit-learn|pytest)"

# 2. Run quick unit test
echo "2. Running architecture test..."
cd /home/jacobwienberg/Cor_trading
PYTHONPATH=. pytest tests/test_architecture.py -v --tb=short || { echo "Architecture test failed"; exit 1; }

# 3. Run short backtest (1 month, single pair, includes costs/slippage via config)
echo "3. Running short backtest (USDCAD-WTI, Jan-Jun 2025)..."
rm -f backtest_results/usdcad_wti_backtest_*.csv  # Clean previous
PYTHONPATH=. python src/run_backtest.py run --pair usdcad_wti --start 2025-01-01 --end 2025-06-30 --save-data

# 4. Verify output
if ls backtest_results/usdcad_wti_backtest_*.csv >/dev/null 2>&1; then
    echo "Backtest output verified: CSV file created."
    echo "Sample equity curve:"
    tail -5 backtest_results/usdcad_wti_backtest_*.csv
else
    echo "Backtest failed: No CSV output."
    exit 1
fi

# 5. Basic metrics check (quick Python snippet for PnL)
echo "4. Quick PnL check..."
python -c "
import pandas as pd
import glob
files = glob.glob('backtest_results/usdcad_wti_backtest_*.csv')
if files:
    df = pd.read_csv(files[0])
    if 'trade_pnl' in df.columns:
        total_pnl = df['trade_pnl'].sum()
        num_trades = len(df[df['trade_pnl'] != 0])
        print(f'Total PnL: {total_pnl:.2f}, Trades: {num_trades}')
        if total_pnl > -1000:  # Basic threshold (short period)
            print('PnL check: PASS (no catastrophic loss)')
        else:
            print('PnL check: FAIL')
            exit(1)
    else:
        print('No trade_pnl column; assuming no trades (PASS for smoke)')
else:
    print('No backtest file; FAIL')
    exit(1)
"

echo "=== Smoke Test PASS ==="