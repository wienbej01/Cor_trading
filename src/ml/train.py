import click
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

from src.ml.models import create_baseline_model
from src.ml.schema import DEFAULT_FEATURES

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """Plots a calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, 's-', label='Model')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')
    ax.set_title('Calibration plot')
    return fig

@click.command()
@click.option('--pair', required=True, type=str)
@click.option('--cv', type=int, default=5, help='Number of cross-validation folds.')
@click.option('--seed', type=int, default=42, help='Random seed.')
@click.option('--input-dir', default='data/ml', help='Directory to load data from.')
@click.option('--out', required=True, type=str, help='Output directory for artifacts.')
def main(pair, cv, seed, input_dir, out):
    """Trains the ML trade filter model."""
    click.echo(f"Starting model training for {pair}...")
    
    # 1. Load data
    input_path = Path(input_dir) / pair / 'train.parquet'
    if not input_path.exists():
        click.echo(f"Error: Training data not found at {input_path}", err=True)
        return
        
    train_df = pd.read_parquet(input_path)
    
    # Drop features that might not be available at inference time or are identifiers
    features_to_use = [f for f in DEFAULT_FEATURES if f in train_df.columns and f not in ['spread', 'alpha', 'beta']]
    X = train_df[features_to_use]
    y = train_df['label']

    # 2. Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    metrics = {
        'roc_auc': [],
        'pr_auc': [],
        'brier_score': []
    }
    
    click.echo(f"Performing {cv}-fold time-series cross-validation...")
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = create_baseline_model(random_state=seed)
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
        metrics['pr_auc'].append(average_precision_score(y_test, y_prob))
        metrics['brier_score'].append(brier_score_loss(y_test, y_prob))
        
        click.echo(f"  Fold {i+1}/{cv} | ROC-AUC: {metrics['roc_auc'][-1]:.4f}")

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    click.echo(f"CV Average ROC-AUC: {avg_metrics['roc_auc']:.4f}")

    # 3. Retrain on full dataset and save artifacts
    click.echo("Retraining model on the full training dataset...")
    final_model = create_baseline_model(random_state=seed)
    final_model.fit(X, y)
    
    # Create output directory
    output_path = Path(out)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_artifact_path = output_path / ts
    run_artifact_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = run_artifact_path / 'model.joblib'
    joblib.dump(final_model, model_path)
    click.echo(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = run_artifact_path / 'metrics.json'
    full_metrics = {'cv_metrics': metrics, 'avg_cv_metrics': avg_metrics}
    with open(metrics_path, 'w') as f:
        json.dump(full_metrics, f, indent=4)
    click.echo(f"Metrics saved to {metrics_path}")
    
    # Save calibration plot
    y_prob_full = final_model.predict_proba(X)[:, 1]
    cal_fig = plot_calibration_curve(y, y_prob_full)
    plot_path = run_artifact_path / 'calibration_plot.png'
    cal_fig.savefig(plot_path)
    click.echo(f"Calibration plot saved to {plot_path}")

if __name__ == '__main__':
    main()
