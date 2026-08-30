from projection_core import build_production_model, train_projection_model
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _weekly_data() -> pd.DataFrame:
    rows = []
    for variety, offset in [("001RED", 0.0), ("002RED", 2.0)]:
        for week in range(1, 17):
            stems = 10.0 + week * 0.2 + offset
            rows.append({
                "Anio": 2025,
                "Semana": week,
                "Bloque&Varid": variety,
                "Tallos/m2": stems,
                "Produccion": 900.0 + week * 25.0 + offset * 10.0,
                "m2Variedad": 100.0,
            })
    return pd.DataFrame(rows)


def test_build_production_model_uses_shared_parameters():
    model = build_production_model()

    assert model.n_estimators == 100
    assert model.max_depth == 16
    assert model.random_state == 42
    assert model.max_features == "sqrt"


def test_train_projection_model_predicts_production_with_reference_pattern():
    result = train_projection_model(_weekly_data(), "001RED")

    assert result["selected_var"] == "001RED"
    assert result["reference_var"] == "002RED"
    assert result["using_reference_pattern"] is True
    assert result["rows_used"] == 12
    assert np.isfinite(result["rmse"])
    assert result["factor_origin"] in {"variedad", "global"}
    assert not result["chart_df"].empty
    assert {
        "anio",
        "semana",
        "produccion_real",
        "produccion_patron",
        "estimado_modelo",
    } == set(result["chart_df"].columns)
