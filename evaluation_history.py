import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


RECENT_EVALUATION_WEEKS = 5


def normalize_recent_case(value):
    text = "" if value is None or pd.isna(value) else str(value)
    text = "".join(text.split())
    prefix_length = len(text) - len(text.lstrip("0123456789"))
    if 0 < prefix_length < 3 and prefix_length < len(text):
        text = text[:prefix_length].zfill(3) + text[prefix_length:]
    return text.casefold()


def normalize_block(value):
    text = "" if value is None or pd.isna(value) else str(value).strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(numeric) and float(numeric).is_integer():
        return f"{int(numeric):03d}"
    return text


def case_from_fields(block, variety, case):
    if pd.notna(block) and str(block).strip() and pd.notna(variety) and str(variety).strip():
        return normalize_recent_case(normalize_block(block) + str(variety).strip())
    return normalize_recent_case(case)


def _history_keys(frame):
    if not {"Anio", "Semana"}.issubset(frame.columns):
        raise ValueError("La evaluacion debe incluir Anio y Semana.")
    if "Bloque&Varid" not in frame.columns and not {
        "Bloque", "Variedad"
    }.issubset(frame.columns):
        raise ValueError("La evaluacion debe identificar cada Bloque y Variedad.")

    years = pd.to_numeric(frame["Anio"], errors="coerce")
    weeks = pd.to_numeric(frame["Semana"], errors="coerce")
    errors = pd.to_numeric(frame["%dif"], errors="coerce")
    cases = []
    for _, row in frame.iterrows():
        cases.append(case_from_fields(
            row.get("Bloque"), row.get("Variedad"), row.get("Bloque&Varid")
        ))
    keys = pd.DataFrame({
        "case": cases, "year": years, "week": weeks, "error": errors
    }, index=frame.index)
    valid = (
        keys["case"].ne("")
        & np.isfinite(keys["year"])
        & np.isfinite(keys["week"])
        & np.isfinite(keys["error"])
        & keys["year"].eq(np.floor(keys["year"]))
        & keys["week"].eq(np.floor(keys["week"]))
        & keys["week"].between(1, 53)
    )
    if not valid.all():
        raise ValueError(
            "La evaluacion contiene identificadores, periodos o valores %Dif invalidos."
        )
    return keys


def save_recent_evaluation_history(report, destination):
    """Reemplaza periodos recibidos y conserva cinco semanas por caso."""
    destination = Path(destination)
    incoming = report.copy().reset_index(drop=True)
    new_keys = _history_keys(incoming)
    if incoming.empty:
        raise ValueError("La evaluacion no contiene valores validos.")
    if destination.exists():
        previous = pd.read_csv(destination, encoding="utf-8-sig")
        if not previous.empty:
            previous = previous.reset_index(drop=True)
            old_keys = _history_keys(previous)
            replaced = set(zip(new_keys["case"], new_keys["year"], new_keys["week"]))
            keep = [
                (case, year, week) not in replaced
                for case, year, week in zip(
                    old_keys["case"], old_keys["year"], old_keys["week"]
                )
            ]
            incoming = pd.concat([previous.loc[keep], incoming], ignore_index=True)

    keys = _history_keys(incoming)
    periods = keys[["case", "year", "week"]].drop_duplicates()
    periods = periods.sort_values(["case", "year", "week"])
    periods = periods.groupby("case", sort=False).tail(RECENT_EVALUATION_WEEKS)
    retained = set(map(tuple, periods.to_numpy()))
    mask = [
        (case, year, week) in retained
        for case, year, week in zip(keys["case"], keys["year"], keys["week"])
    ]
    history = incoming.loc[mask].reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="",
            dir=destination.parent, suffix=".tmp", delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            history.to_csv(temp, index=False, lineterminator="\n")
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return history
