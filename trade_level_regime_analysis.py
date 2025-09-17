import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def analyze_actual_trades():
    """Analyze actual trade-level performance with proper regime classification"""
    
    print("=== TRADE-LEVEL REGIME ANALYSIS ===")
    print("Focusing on actual trades, not daily positions...")
    
    # Define backtest files
    backtest_files = {
        "USDCAD-WTI-Kalman": "backtest_results/usdcad_wti_backtest_20250821_112149.csv",
        "USDCAD-WTI-OLS": "backtest_results/usdcad_wti_backtest_20250821_112216.csv",
        "USDNOK-Brent-Kalman": "backtest_results/usdnok_brent_backtest_20250821_112249.csv",
        "USDNOK-Brent-OLS": "backtest_results/usdnok_brent_backtest_20250821_112327.csv"
    }
    
    all_results = {}
    
    for strategy_name, file_path in backtest_files.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {strategy_name}")
        print(f"{'='*60}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Calculate market indicators
            df['fx_return'] = df['fx_price'].pct_change()
            df['comd_return'] = df['comd_price'].pct_change()
            
            # Identify ACTUAL TRADES (where trade_pnl != 0)
            actual_trades = df[df['trade_pnl'] != 0].copy()
            
            print(f"Total data points: {len(df):,}")
            print(f"Actual trades: {len(actual_trades):,}")
            print(f"Trade frequency: {len(actual_trades)/len(df)*100:.1f}% of days")
            
            if len(actual_trades) == 0:
                print("⚠️  NO TRADES FOUND - STRATEGY NEVER ENTERED POSITIONS")
                continue
            
            # Calculate regime indicators for trade dates
            # Volatility (20-day rolling)
            df['fx_vol_20'] = df['fx_return'].rolling(20).std() * np.sqrt(252)
            df['comd_vol_20'] = df['comd_return'].rolling(20).std() * np.sqrt(252)
            df['avg_vol'] = (df['fx_vol_20'] + df['comd_vol_20']) / 2
            
            # Trend (20-day cumulative returns)
            df['fx_trend_20'] = df['fx_return'].rolling(20).sum()
            df['comd_trend_20'] = df['comd_return'].rolling(20).sum()
            df['avg_trend'] = (df['fx_trend_20'] + df['comd_trend_20']) / 2
            
            # Z-score momentum
            df['zscore_momentum_10'] = df['spread_z'].diff(10)
            df['zscore_momentum_20'] = df['spread_z'].diff(20)
            
            # Correlation regime
            df['fx_comd_corr_20'] = df['fx_return'].rolling(20).corr(df['comd_return'])
            
            # Add regime classifications to actual trades
            actual_trades = actual_trades.merge(
                df[['Date', 'avg_vol', 'avg_trend', 'zscore_momentum_20', 'fx_comd_corr_20', 'spread_vol_20']], 
                on='Date', how='left'
            )
            
            # Classify regimes for actual trades
            def classify_volatility(vol):
                if pd.isna(vol):
                    return 'Unknown'
                elif vol < 0.05:  # 5% annualized
                    return 'Low Vol (<5%)'
                elif vol > 0.15:  # 15% annualized
                    return 'High Vol (>15%)'
                else:
                    return 'Medium Vol (5-15%)'
            
            def classify_market(trend):
                if pd.isna(trend):
                    return 'Unknown'
                elif trend > 0.03:  # 3% over 20 days
                    return 'Strong Up Trend'
                elif trend > 0.01:  # 1% over 20 days
                    return 'Weak Up Trend'
                elif trend < -0.03:  # -3% over 20 days
                    return 'Strong Down Trend'
                elif trend < -0.01:  # -1% over 20 days
                    return 'Weak Down Trend'
                else:
                    return 'Ranging'
            
            def classify_momentum(zscore_mom):
                if pd.isna(zscore_mom):
                    return 'Unknown'
                elif zscore_mom > 1.0:
                    return 'Strong Up Momentum'
                elif zscore_mom > 0.3:
                    return 'Weak Up Momentum'
                elif zscore_mom < -1.0:
                    return 'Strong Down Momentum'
                elif zscore_mom < -0.3:
                    return 'Weak Down Momentum'
                else:
                    return 'Neutral Momentum'
            
            def classify_correlation(corr):
                if pd.isna(corr):
                    return 'Unknown'
                elif corr > 0.7:
                    return 'High Correlation (>0.7)'
                elif corr > 0.3:
                    return 'Medium Correlation (0.3-0.7)'
                elif corr > 0.0:
                    return 'Low Correlation (0-0.3)'
                else:
                    return 'Negative Correlation (<0)'
            
            actual_trades['volatility_regime'] = actual_trades['avg_vol'].apply(classify_volatility)
            actual_trades['market_regime'] = actual_trades['avg_trend'].apply(classify_market)
            actual_trades['momentum_regime'] = actual_trades['zscore_momentum_20'].apply(classify_momentum)
            actual_trades['correlation_regime'] = actual_trades['fx_comd_corr_20'].apply(classify_correlation)
            
            # Time-based analysis
            actual_trades['year'] = actual_trades['Date'].dt.year
            actual_trades['month'] = actual_trades['Date'].dt.month
            actual_trades['quarter'] = actual_trades['Date'].dt.quarter
            
            # Performance analysis by regime
            print(f"\n📊 TRADE PERFORMANCE BY REGIME:")
            
            # 1. Time-based analysis
            print(f"\n1. TIME-BASED PERFORMANCE:")
            yearly_perf = actual_trades.groupby('year').agg({
                'trade_pnl': ['count', 'sum', 'mean', 'std'],
                'trade_return': ['mean', 'std']
            }).round(4)
            
            for year in sorted(actual_trades['year'].unique()):
                if year >= 2015:  # Only show complete years
                    trades = yearly_perf.loc[year, ('trade_pnl', 'count')]
                    total_pnl = yearly_perf.loc[year, ('trade_pnl', 'sum')]
                    avg_pnl = yearly_perf.loc[year, ('trade_pnl', 'mean')]
                    win_rate = (actual_trades[actual_trades['year'] == year]['trade_pnl'] > 0).mean() * 100
                    print(f"  {year}: {trades} trades, {total_pnl:.3f} total PnL, {avg_pnl:.3f} avg PnL, {win_rate:.1f}% win rate")
            
            # 2. Volatility regime performance
            print(f"\n2. VOLATILITY REGIME PERFORMANCE:")
            vol_perf = actual_trades.groupby('volatility_regime').agg({
                'trade_pnl': ['count', 'sum', 'mean', 'std'],
                'trade_return': ['mean', 'std']
            }).round(4)
            
            for regime in ['Low Vol (<5%)', 'Medium Vol (5-15%)', 'High Vol (>15%)']:
                if regime in vol_perf.index:
                    trades = vol_perf.loc[regime, ('trade_pnl', 'count')]
                    total_pnl = vol_perf.loc[regime, ('trade_pnl', 'sum')]
                    avg_pnl = vol_perf.loc[regime, ('trade_pnl', 'mean')]
                    win_rate = (actual_trades[actual_trades['volatility_regime'] == regime]['trade_pnl'] > 0).mean() * 100
                    print(f"  {regime}: {trades} trades, {total_pnl:.3f} total PnL, {avg_pnl:.3f} avg PnL, {win_rate:.1f}% win rate")
            
            # 3. Market condition performance
            print(f"\n3. MARKET CONDITION PERFORMANCE:")
            market_perf = actual_trades.groupby('market_regime').agg({
                'trade_pnl': ['count', 'sum', 'mean', 'std'],
                'trade_return': ['mean', 'std']
            }).round(4)
            
            market_conditions = ['Strong Up Trend', 'Weak Up Trend', 'Ranging', 'Weak Down Trend', 'Strong Down Trend']
            for condition in market_conditions:
                if condition in market_perf.index:
                    trades = market_perf.loc[condition, ('trade_pnl', 'count')]
                    total_pnl = market_perf.loc[condition, ('trade_pnl', 'sum')]
                    avg_pnl = market_perf.loc[condition, ('trade_pnl', 'mean')]
                    win_rate = (actual_trades[actual_trades['market_regime'] == condition]['trade_pnl'] > 0).mean() * 100
                    print(f"  {condition}: {trades} trades, {total_pnl:.3f} total PnL, {avg_pnl:.3f} avg PnL, {win_rate:.1f}% win rate")
            
            # 4. Momentum regime performance
            print(f"\n4. MOMENTUM REGIME PERFORMANCE:")
            momentum_perf = actual_trades.groupby('momentum_regime').agg({
                'trade_pnl': ['count', 'sum', 'mean', 'std'],
                'trade_return': ['mean', 'std']
            }).round(4)
            
            momentum_conditions = ['Strong Up Momentum', 'Weak Up Momentum', 'Neutral Momentum', 'Weak Down Momentum', 'Strong Down Momentum']
            for condition in momentum_conditions:
                if condition in momentum_perf.index:
                    trades = momentum_perf.loc[condition, ('trade_pnl', 'count')]
                    total_pnl = momentum_perf.loc[condition, ('trade_pnl', 'sum')]
                    avg_pnl = momentum_perf.loc[condition, ('trade_pnl', 'mean')]
                    win_rate = (actual_trades[actual_trades['momentum_regime'] == condition]['trade_pnl'] > 0).mean() * 100
                    print(f"  {condition}: {trades} trades, {total_pnl:.3f} total PnL, {avg_pnl:.3f} avg PnL, {win_rate:.1f}% win rate")
            
            # 5. Correlation regime performance
            print(f"\n5. CORRELATION REGIME PERFORMANCE:")
            corr_perf = actual_trades.groupby('correlation_regime').agg({
                'trade_pnl': ['count', 'sum', 'mean', 'std'],
                'trade_return': ['mean', 'std']
            }).round(4)
            
            corr_conditions = ['High Correlation (>0.7)', 'Medium Correlation (0.3-0.7)', 'Low Correlation (0-0.3)', 'Negative Correlation (<0)']
            for condition in corr_conditions:
                if condition in corr_perf.index:
                    trades = corr_perf.loc[condition, ('trade_pnl', 'count')]
                    total_pnl = corr_perf.loc[condition, ('trade_pnl', 'sum')]
                    avg_pnl = corr_perf.loc[condition, ('trade_pnl', 'mean')]
                    win_rate = (actual_trades[actual_trades['correlation_regime'] == condition]['trade_pnl'] > 0).mean() * 100
                    print(f"  {condition}: {trades} trades, {total_pnl:.3f} total PnL, {avg_pnl:.3f} avg PnL, {win_rate:.1f}% win rate")
            
            # Key insights
            print(f"\n🔍 KEY INSIGHTS:")
            
            # Best and worst performing regimes
            if len(actual_trades) > 0:
                best_vol = actual_trades.groupby('volatility_regime')['trade_pnl'].mean().idxmax()
                best_vol_pnl = actual_trades.groupby('volatility_regime')['trade_pnl'].mean().max()
                worst_vol = actual_trades.groupby('volatility_regime')['trade_pnl'].mean().idxmin()
                worst_vol_pnl = actual_trades.groupby('volatility_regime')['trade_pnl'].mean().min()
                
                best_market = actual_trades.groupby('market_regime')['trade_pnl'].mean().idxmax()
                best_market_pnl = actual_trades.groupby('market_regime')['trade_pnl'].mean().max()
                worst_market = actual_trades.groupby('market_regime')['trade_pnl'].mean().idxmin()
                worst_market_pnl = actual_trades.groupby('market_regime')['trade_pnl'].mean().min()
                
                print(f"  Best Volatility Regime: {best_vol} (avg PnL: {best_vol_pnl:.3f})")
                print(f"  Worst Volatility Regime: {worst_vol} (avg PnL: {worst_vol_pnl:.3f})")
                print(f"  Best Market Condition: {best_market} (avg PnL: {best_market_pnl:.3f})")
                print(f"  Worst Market Condition: {worst_market} (avg PnL: {worst_market_pnl:.3f})")
            
            # Store results
            all_results[strategy_name] = {
                'total_trades': len(actual_trades),
                'trade_data': actual_trades,
                'yearly_perf': yearly_perf if 'yearly_perf' in locals() else None,
                'vol_perf': vol_perf if 'vol_perf' in locals() else None,
                'market_perf': market_perf if 'market_perf' in locals() else None,
                'momentum_perf': momentum_perf if 'momentum_perf' in locals() else None,
                'corr_perf': corr_perf if 'corr_perf' in locals() else None
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {strategy_name}: {e}")
            continue
    
    # Summary comparison across strategies
    print(f"\n{'='*80}")
    print("CROSS-STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    if all_results:
        print(f"\n📈 STRATEGY COMPARISON SUMMARY:")
        for strategy, results in all_results.items():
            if 'total_trades' in results:
                trades = results['total_trades']
                print(f"  {strategy}: {trades} total trades")
        
        # Find best performing regimes across all strategies
        print(f"\n🏆 BEST PERFORMING REGIMES ACROSS ALL STRATEGIES:")
        
        all_vol_results = []
        all_market_results = []
        
        for strategy, results in all_results.items():
            if 'trade_data' in results and results['trade_data'] is not None:
                trade_data = results['trade_data']
                
                # Volatility regime performance
                vol_perf = trade_data.groupby('volatility_regime')['trade_pnl'].agg(['count', 'mean']).round(4)
                vol_perf['strategy'] = strategy
                all_vol_results.append(vol_perf)
                
                # Market regime performance
                market_perf = trade_data.groupby('market_regime')['trade_pnl'].agg(['count', 'mean']).round(4)
                market_perf['strategy'] = strategy
                all_market_results.append(market_perf)
        
        if all_vol_results:
            combined_vol = pd.concat(all_vol_results)
            best_vol_overall = combined_vol.groupby('volatility_regime')['mean'].mean().idxmax()
            best_vol_pnl_overall = combined_vol.groupby('volatility_regime')['mean'].mean().max()
            
            print(f"  Best Volatility Regime Overall: {best_vol_overall} (avg PnL: {best_vol_pnl_overall:.3f})")
        
        if all_market_results:
            combined_market = pd.concat(all_market_results)
            best_market_overall = combined_market.groupby('market_regime')['mean'].mean().idxmax()
            best_market_pnl_overall = combined_market.groupby('market_regime')['mean'].mean().max()
            
            print(f"  Best Market Condition Overall: {best_market_overall} (avg PnL: {best_market_pnl_overall:.3f})")
    
    print(f"\n✅ Trade-level regime analysis complete!")
    return all_results


if __name__ == "__main__":
    results = analyze_actual_trades()