from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

def create_baseline_model(calibration_method: str = 'isotonic', random_state: int = 42):
    """
    Creates the baseline ML model pipeline.

    The model is a HistGradientBoostingClassifier, which is efficient for
    tabular data. It is wrapped in a CalibratedClassifierCV to ensure
    that the predicted probabilities are well-calibrated.

    Args:
        calibration_method (str): The method to use for calibration.
            'isotonic' or 'sigmoid'.
        random_state (int): The random state for reproducibility.

    Returns:
        A scikit-learn Pipeline object representing the model.
    """
    # Define the base classifier
    hgb_clf = HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=100,  # Keep it fast for a baseline
        learning_rate=0.1,
        max_depth=5
    )

    # Create the calibrated classifier
    # The base estimator is cloned and not used directly.
    calibrated_clf = CalibratedClassifierCV(
        hgb_clf,
        method=calibration_method,
        cv=3 # Use 3-fold CV to find the best calibration
    )
    
    pipeline = Pipeline([
        ('classifier', calibrated_clf)
    ])

    return pipeline
