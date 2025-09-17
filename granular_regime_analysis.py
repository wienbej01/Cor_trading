import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

class GranularRegimeAnalyzer:
    def __init__(self):
        self.regime_definitions = {
            'time_buckets': {
                '2015-2016': ('2015-01-01', '2016-12-31'),
                '2017-2018': ('2017-01-01', '2018-12-31'),
                '2019-2020': ('2019-01-01', '2020-12-31'),
                '2021-2022': ('2021-01-01', '2022-12-31'),
                '2023-2024': ('2023-01-01', '2024-12-31'),
                '2025': ('2025-01-01', '2025-12-31')
            },
            'market_conditions': {
                'trending_up': {'min_return': 0.02, 'window': 20},
                'trending_down': {'max_return': -0.02, 'window': 20},
                'ranging': {'min_return': -0.01, 'max_return': 0.01, 'window': 20}
            },
            'volatility_regimes': {
                'low_vol': {'threshold': 0.05, 'window': 20},
                'medium_vol': {'min_threshold': 0.05, 'max_threshold': 0.10, 'window': 20},
                'high_vol': {'threshold': 0.10, 'window': 20}
            },
            'momentum_regimes': {
                'strong_up': {'threshold': 0.8, 'window': 10},
                'weak_up': {'min_threshold': 0.2, 'max_threshold': 0.8, 'window': 10},
                'neutral': {'min_threshold': -0.2, 'max_threshold': 0.2, 'window': 10},
                'weak_down': {'min_threshold': -0.8, 'max_threshold': -0.2, 'window': 10},
                'strong_down': {'threshold': -0.8, 'window': 10}
            }
        }

    def load_backtest_data(self, file_path):
        """Load backtest data with proper date parsing"""
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_regime_indicators(self, df):
        """Calculate various regime indicators"""
        # Market trend indicators
        df['fx_return'] = df['fx_price'].pct_change()
        df['comd_return'] = df['comd_price'].pct_change()
        
        # Rolling returns for trend detection
        for window in [5, 10, 20, 40]:
            df[f'fx_trend_{window}'] = df['fx_return'].rolling(window).sum()
            df[f'comd_trend_{window}'] = df['comd_return'].rolling(window).sum()
            df[f'fx_vol_{window}'] = df['fx_return'].rolling(window).std() * np.sqrt(252)
            df[f'comd_vol_{window}'] = df['comd_return'].rolling(window).std() * np.sqrt(252)
        
        # Spread volatility
        df['spread_return'] = df['spread'].pct_change()
        df['spread_vol_20'] = df['spread_return'].rolling(20).std() * np.sqrt(252)
        
        # Correlation regime
        df['fx_comd_corr_20'] = df['fx_return'].rolling(20).corr(df['comd_return'])
        df['fx_comd_corr_60'] = df['fx_return'].rolling(60).corr(df['comd_return'])
        
        # Z-score momentum
        df['zscore_momentum_10'] = df['spread_z'].diff(10)
        df['zscore_momentum_20'] = df['spread_z'].diff(20)
        
        # Equity curve analysis
        df['equity_return'] = df['equity'].pct_change()
        df['equity_vol_20'] = df['equity_return'].rolling(20).std() * np.sqrt(252)
        
        return df

    def classify_time_buckets(self, df):
        """Classify data into 2-year time buckets"""
        df['time_bucket'] = 'Unknown'
        
        for bucket, (start, end) in self.regime_definitions['time_buckets'].items():
            mask = (df['Date'] >= start) & (df['Date'] <= end)
            df.loc[mask, 'time_bucket'] = bucket
        
        return df

    def classify_market_conditions(self, df):
        """Classify market into trending up/down/ranging"""
        df['market_condition'] = 'Unknown'
        
        # Use 20-day rolling returns for classification
        for i in range(len(df)):
            if i < 20:
                continue
                
            fx_trend = df.loc[i, 'fx_trend_20']
            comd_trend = df.loc[i, 'comd_trend_20']
            avg_trend = (fx_trend + comd_trend) / 2
            
            if avg_trend > 0.02:  # 2% over 20 days
                df.loc[i, 'market_condition'] = 'trending_up'
            elif avg_trend < -0.02:  # -2% over 20 days
                df.loc[i, 'market_condition'] = 'trending_down'
            else:
                df.loc[i, 'market_condition'] = 'ranging'
        
        return df

    def classify_volatility_regimes(self, df):
        """Classify volatility regimes"""
        df['volatility_regime'] = 'Unknown'
        
        for i in range(len(df)):
            if i < 20:
                continue
                
            avg_vol = (df.loc[i, 'fx_vol_20'] + df.loc[i, 'comd_vol_20']) / 2
            
            if avg_vol < 0.05:  # 5% annualized volatility
                df.loc[i, 'volatility_regime'] = 'low_vol'
            elif avg_vol > 0.10:  # 10% annualized volatility
                df.loc[i, 'volatility_regime'] = 'high_vol'
            else:
                df.loc[i, 'volatility_regime'] = 'medium_vol'
        
        return df

    def classify_momentum_regimes(self, df):
        """Classify momentum regimes based on z-score momentum"""
        df['momentum_regime'] = 'Unknown'
        
        for i in range(len(df)):
            if i < 20:
                continue
                
            zscore_mom = df.loc[i, 'zscore_momentum_20']
            
            if zscore_mom > 0.8:
                df.loc[i, 'momentum_regime'] = 'strong_up'
            elif zscore_mom > 0.2:
                df.loc[i, 'momentum_regime'] = 'weak_up'
            elif zscore_mom < -0.8:
                df.loc[i, 'momentum_regime'] = 'strong_down'
            elif zscore_mom < -0.2:
                df.loc[i, 'momentum_regime'] = 'weak_down'
            else:
                df.loc[i, 'momentum_regime'] = 'neutral'
        
        return df

    def calculate_performance_metrics(self, df, regime_col, regime_value):
        """Calculate performance metrics for a specific regime"""
        regime_data = df[df[regime_col] == regime_value].copy()
        
        if len(regime_data) == 0:
            return {
                'regime': regime_value,
                'days': 0,
                'trades': 0,
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'trade_frequency': 0
            }
        
        # Calculate metrics
        start_equity = regime_data['equity'].iloc[0]
        end_equity = regime_data['equity'].iloc[-1]
        total_return = (end_equity / start_equity - 1) * 100
        
        days = len(regime_data)
        years = days / 252
        annual_return = total_return / years if years > 0 else 0
        
        # Volatility
        equity_returns = regime_data['equity'].pct_change().dropna()
        volatility = equity_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0
        
        # Max drawdown
        running_max = regime_data['equity'].expanding().max()
        drawdown = (regime_data['equity'] / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Trading metrics
        trades = regime_data[regime_data['trade_pnl'] != 0]
        if len(trades) > 0:
            win_rate = (trades['trade_pnl'] > 0).mean() * 100
            profit_factor = trades[trades['trade_pnl'] > 0]['trade_pnl'].sum() / abs(trades[trades['trade_pnl'] < 0]['trade_pnl'].sum()) if (trades['trade_pnl'] < 0).any() else 0
            avg_trade_return = trades['trade_pnl'].mean()
            trade_frequency = len(trades) / days * 252  # trades per year
        else:
            win_rate = profit_factor = avg_trade_return = trade_frequency = 0
        
        return {
            'regime': regime_value,
            'days': days,
            'trades': len(trades),
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'trade_frequency': trade_frequency
        }

    def analyze_all_regimes(self, df, strategy_name):
        """Analyze performance across all regime classifications"""
        results = {}
        
        # Time bucket analysis
        time_results = []
        for bucket in self.regime_definitions['time_buckets'].keys():
            metrics = self.calculate_performance_metrics(df, 'time_bucket', bucket)
            time_results.append(metrics)
        results['time_buckets'] = pd.DataFrame(time_results)
        
        # Market condition analysis
        market_results = []
        for condition in ['trending_up', 'trending_down', 'ranging']:
            metrics = self.calculate_performance_metrics(df, 'market_condition', condition)
            market_results.append(metrics)
        results['market_conditions'] = pd.DataFrame(market_results)
        
        # Volatility regime analysis
        vol_results = []
        for regime in ['low_vol', 'medium_vol', 'high_vol']:
            metrics = self.calculate_performance_metrics(df, 'volatility_regime', regime)
            vol_results.append(metrics)
        results['volatility_regimes'] = pd.DataFrame(vol_results)
        
        # Momentum regime analysis
        momentum_results = []
        for regime in ['strong_up', 'weak_up', 'neutral', 'weak_down', 'strong_down']:
            metrics = self.calculate_performance_metrics(df, 'momentum_regime', regime)
            momentum_results.append(metrics)
        results['momentum_regimes'] = pd.DataFrame(momentum_results)
        
        return results

    def create_comprehensive_report(self, all_results):
        """Create a comprehensive granular performance report"""
        report = []
        
        report.append("=" * 80)
        report.append("GRANULAR REGIME-BASED PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        
        for strategy, results in all_results.items():
            report.append(f"\n{'='*60}")
            report.append(f"STRATEGY: {strategy}")
            report.append(f"{'='*60}")
            
            # Time Bucket Analysis
            report.append("\n1. TIME BUCKET ANALYSIS (2-Year Periods)")
            report.append("-" * 50)
            time_df = results['time_buckets']
            report.append(time_df.to_string(index=False, float_format="%.2f"))
            
            # Market Condition Analysis
            report.append("\n2. MARKET CONDITION ANALYSIS")
            report.append("-" * 50)
            market_df = results['market_conditions']
            report.append(market_df.to_string(index=False, float_format="%.2f"))
            
            # Volatility Regime Analysis
            report.append("\n3. VOLATILITY REGIME ANALYSIS")
            report.append("-" * 50)
            vol_df = results['volatility_regimes']
            report.append(vol_df.to_string(index=False, float_format="%.2f"))
            
            # Momentum Regime Analysis
            report.append("\n4. MOMENTUM REGIME ANALYSIS")
            report.append("-" * 50)
            momentum_df = results['momentum_regimes']
            report.append(momentum_df.to_string(index=False, float_format="%.2f"))
            
            # Key Insights
            report.append("\n5. KEY REGIME INSIGHTS")
            report.append("-" * 50)
            
            # Best performing time bucket
            best_time = time_df.loc[time_df['total_return'].idxmax()]
            worst_time = time_df.loc[time_df['total_return'].idxmin()]
            report.append(f"Best Time Period: {best_time['regime']} (Return: {best_time['total_return']:.2f}%)")
            report.append(f"Worst Time Period: {worst_time['regime']} (Return: {worst_time['total_return']:.2f}%)")
            
            # Best performing market condition
            if len(market_df) > 0:
                best_market = market_df.loc[market_df['total_return'].idxmax()]
                worst_market = market_df.loc[market_df['total_return'].idxmin()]
                report.append(f"Best Market Condition: {best_market['regime']} (Return: {best_market['total_return']:.2f}%)")
                report.append(f"Worst Market Condition: {worst_market['regime']} (Return: {worst_market['total_return']:.2f}%)")
            
            # Best performing volatility regime
            if len(vol_df) > 0:
                best_vol = vol_df.loc[vol_df['total_return'].idxmax()]
                worst_vol = vol_df.loc[vol_df['total_return'].idxmin()]
                report.append(f"Best Volatility Regime: {best_vol['regime']} (Return: {best_vol['total_return']:.2f}%)")
                report.append(f"Worst Volatility Regime: {worst_vol['regime']} (Return: {worst_vol['total_return']:.2f}%)")
            
            # Best performing momentum regime
            if len(momentum_df) > 0:
                best_momentum = momentum_df.loc[momentum_df['total_return'].idxmax()]
                worst_momentum = momentum_df.loc[momentum_df['total_return'].idxmin()]
                report.append(f"Best Momentum Regime: {best_momentum['regime']} (Return: {best_momentum['total_return']:.2f}%)")
                report.append(f"Worst Momentum Regime: {worst_momentum['regime']} (Return: {worst_momentum['total_return']:.2f}%)")
        
        return "\n".join(report)

    def run_granular_analysis(self, backtest_files):
        """Run comprehensive granular analysis on all backtest files"""
        all_results = {}
        
        for strategy_name, file_path in backtest_files.items():
            print(f"Analyzing {strategy_name}...")
            
            # Load data
            df = self.load_backtest_data(file_path)
            if df is None:
                continue
            
            # Calculate regime indicators
            df = self.calculate_regime_indicators(df)
            
            # Classify regimes
            df = self.classify_time_buckets(df)
            df = self.classify_market_conditions(df)
            df = self.classify_volatility_regimes(df)
            df = self.classify_momentum_regimes(df)
            
            # Analyze all regimes
            results = self.analyze_all_regimes(df, strategy_name)
            all_results[strategy_name] = results
        
        # Create comprehensive report
        report = self.create_comprehensive_report(all_results)
        
        # Save report
        with open("granular_regime_analysis_report.txt", "w") as f:
            f.write(report)
        
        print("✅ Granular regime analysis complete!")
        print("📊 Report saved to: granular_regime_analysis_report.txt")
        
        return all_results


def main():
    """Main function to run granular regime analysis"""
    
    # Define backtest files to analyze
    backtest_files = {
        "USDCAD-WTI-Kalman": "backtest_results/usdcad_wti_backtest_20250821_112149.csv",
        "USDCAD-WTI-OLS": "backtest_results/usdcad_wti_backtest_20250821_112216.csv",
        "USDNOK-Brent-Kalman": "backtest_results/usdnok_brent_backtest_20250821_112249.csv",
        "USDNOK-Brent-OLS": "backtest_results/usdnok_brent_backtest_20250821_112327.csv"
    }
    
    # Create analyzer and run analysis
    analyzer = GranularRegimeAnalyzer()
    results = analyzer.run_granular_analysis(backtest_files)
    
    return results


if __name__ == "__main__":
    main()