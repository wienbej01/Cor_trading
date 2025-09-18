# Label Specification - FX-Commodity Mean Reversion

This document defines the exact label construction window, ensures no future leakage, and provides a code sketch for implementation. Labels are used for supervised learning research and must be strictly embargoed.

Code references
- Label core: [backtest.engine.calculate_trade_pnl()](src/backtest/engine.py:173)
- Signal delay: [backtest.engine.backtest_pair()](src/backtest/engine.py:109)
- Feature embargo: [interfaces.feature_preparation.embargoed_split()](src/interfaces/feature_preparation.py:201)

1) Label construction window
- Label time: t (decision bar)
- Label value time: t+1 (execution bar)
- Max holding period: min(time_stop.max_days, 3×OU_half_life) capped at [2, time_stop.max_days]
- Label type: Trade-level return (from entry to exit)
- Label sign: Same as signal_t (1 for long, -1 for short)

2) No leakage enforcement
- Features at t must only use data from t and earlier
- Labels are computed from t+1 onward (embargo period starts at t+1)
- Trade exits are determined by rules at t (no peaking at t+1 prices)
- Time-stop exit uses effective_max_days computed at entry time t

3) Code sketch for label creation
```python
def create_labels(df: pd.DataFrame, pair_config: dict) -> pd.DataFrame:
    \"\"\"
    Create trade-level labels from backtest results.
    
    Parameters:
    - df: DataFrame with columns ['signal', 'fx_position', 'comd_position', 
         'fx_price', 'comd_price', 'time_stop_exit']
    - pair_config: Configuration dict with keys like 'time_stop.max_days'
    
    Returns:
    - DataFrame with added columns ['label', 'label_window_start', 'label_window_end']
    \"\"\"
    # Initialize label columns
    df['label'] = 0.0
    df['label_window_start'] = pd.NaT
    df['label_window_end'] = pd.NaT
    
    # Track active positions
    in_position = False
    position_type = 0  # 1 for long, -1 for short
    entry_time = None
    
    for i in range(1, len(df)):  # Start from t=1 to allow t-1 access
        t = df.index[i]
        t_minus_1 = df.index[i-1]
        
        # Position opened at t-1
        if not in_position and df.loc[t_minus_1, 'signal'] != 0:
            in_position = True
            position_type = np.sign(df.loc[t_minus_1, 'signal'])
            entry_time = t_minus_1
            df.loc[t, 'label_window_start'] = entry_time
            
        # Position closed at t (exit rule triggered)
        if in_position and (df.loc[t, 'signal'] == 0 or df.loc[t, 'time_stop_exit']):
            # Calculate trade PnL from entry to exit
            fx_return = df.loc[entry_time:t, 'fx_price'].pct_change().shift(-1)
            comd_return = df.loc[entry_time:t, 'comd_price'].pct_change().shift(-1)
            
            # Apply one-bar delay (execution at t+1 close)
            fx_pnl = fx_return * df.loc[entry_time:t_minus_1, 'fx_position'].values
            comd_pnl = comd_return * df.loc[entry_time:t_minus_1, 'comd_position'].values
            
            total_pnl = (fx_pnl + comd_pnl).sum()
            
            # Assign label
            df.loc[t, 'label'] = total_pnl * position_type
            df.loc[t, 'label_window_end'] = t
            
            # Reset position tracking
            in_position = False
            position_type = 0
            entry_time = None
    
    return df
```

4) Label usage in supervised research
- Features and labels must be purged/embargoed using [interfaces.feature_preparation.embargoed_split()](src/interfaces/feature_preparation.py:201)
- Embargo period = max holding window per trade (computed dynamically)
- Labels are real-valued returns, suitable for regression tasks
- Class labels can be derived by binning (e.g., top/bottom 30% = long/short signals)

5) Testability checklist
- [ ] Labels reference only past/future data consistent with one-bar execution delay
- [ ] Trade PnL calculation matches [backtest.engine.calculate_trade_pnl()](src/backtest/engine.py:173)
- [ ] Embargo periods align with dynamic holding windows
- [ ] Label sign matches initiating signal

6) Economic rationale
- Labels capture realized mean-reversion trades in regime-filtered environments
- Dynamic time-stop reflects economic half-life of dislocations
- One-bar delay ensures realistic execution in live trading