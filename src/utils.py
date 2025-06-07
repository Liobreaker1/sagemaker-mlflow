import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Union, List, Dict


def eval_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty input arrays passed to eval_metrics.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }
