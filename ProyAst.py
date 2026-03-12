import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import requests
import os
from projection_core import (
    find_reference_pattern,
    get_fincas,
    get_varieties,
    load_excel_bytes,
    scale_reference_projection,
    train_projection_model,
)


def _get_api_defaults() -> tuple[str, str]:
    api_url_default = "https://proyecciones-api.onrender.com/predict"
    api_key_default = ""
    try:
        api_url_default = st.secrets.get("API_URL", api_url_default)
        api_key_default = st.secrets.get("API_KEY", api_key_default)
    except Exception:
        api_url_default = os.getenv("API_URL", api_url_default)
        api_key_default = os.getenv("API_KEY", api_key_default)
    return api_url_default, api_key_default


def _render_api_section(uploaded_file, selected_var: str) -> None:
    api_url_default, api_key_default = _get_api_defaults()
    api_url = st.text_input("URL API Render", value=api_url_default)
    api_key = st.text_input("API Key", value=api_key_default, type="password")

    if not st.button("Ejecutar por API"):
        return

    try:
        uploaded_bytes = uploaded_file.getvalue()
        response = requests.post(
            api_url,
            headers={"X-API-Key": api_key},
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            data={"selected_var": selected_var},
            timeout=120,
        )
    except Exception as exc:
        st.error(f"No fue posible llamar la API: {exc}")
        return

    if not response.ok:
        detail_text = response.text
        try:
            error_payload = response.json()
            detail_text = error_payload.get("detail", detail_text)
            st.error(f"Error API: {response.status_code} - {detail_text}")
            st.json(error_payload)
        except ValueError:
            st.error(f"Error API: {response.status_code} - {detail_text}")
            st.code(response.text)
        return

    result_json = response.json()
    st.success("Respuesta recibida desde API Render")
    st.json(result_json)

    preview = result_json.get("result", {}).get("preview", [])
    if preview:
        st.dataframe(pd.DataFrame(preview), width="stretch")


def _render_local_projection(df: pd.DataFrame, selected_var: str) -> None:
    # 1. Buscar patron de referencia (menor MSE en Tallos/m2)
    reference_pattern = find_reference_pattern(df, selected_var)
    ref_series = reference_pattern["series"] if reference_pattern else None

    # 2. Entrenar modelo con Tallos/m2 del patron como features (logica original)
    try:
        model_result = train_projection_model(
            df, selected_var, reference_series=ref_series
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    chart_df = model_result["chart_df"].copy()

    # 3. Proyeccion patron escalada a la variedad objetivo
    if ref_series is not None:
        scaled_projection = scale_reference_projection(
            ref_series,
            chart_df["tallos_m2_real"],
        )
        if not scaled_projection.empty:
            chart_df = chart_df.iloc[: len(scaled_projection)].copy()
            chart_df["proyeccion_patron"] = scaled_projection.values

    ref_label = reference_pattern["reference_var"] if reference_pattern else "No encontrado"
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Filas usadas", model_result["rows_used"])
    metric_col_2.metric("RMSE", f"{model_result['rmse']:.2f}")
    metric_col_3.metric("Patrón referencia", ref_label)
    tallos_m2_avg = model_result.get("tallos_m2_avg")
    metric_col_4.metric(
        "Promedio Tallos/m2",
        f"{tallos_m2_avg:.2f}" if tallos_m2_avg is not None else "N/D",
    )

    if model_result.get("using_reference_pattern"):
        st.caption(f"Modelo entrenado con Tallos/m2 del patrón: {ref_label}")
    else:
        st.caption(
            "Modelo entrenado con Tallos/m2 propios de la variedad (patron no disponible).")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(chart_df["tallos_m2_real"].reset_index(drop=True),
            label="Tallos/m2 real", color="#b42318", linestyle="--")
    ax.plot(chart_df["prediccion_modelo"].reset_index(drop=True),
            label="Predicción Tallos/m2 (RF)", color="#0b6e4f", linewidth=2)
    if "proyeccion_patron" in chart_df:
        ax.plot(chart_df["proyeccion_patron"].reset_index(drop=True),
                label="Patrón Tallos/m2 (escalado)", color="#1d4ed8", linewidth=2)
    ax.set_title(f"Proyección para {selected_var}")
    ax.set_xlabel("Observación")
    ax.set_ylabel("Tallos/m2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.dataframe(chart_df.tail(20), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Estimados Productivos", layout="wide")
    st.markdown(
        "<h1 style='font-size: 22px; text-align: center;'>ESTIMADOS DE UNIDADES PRODUCTIVAS</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] > div {
            min-height: 10px !important;
            font-size: 12px !important;
        }
        div[data-baseweb="select"] span {
            font-size: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Sube archivo Excel", type=["xlsx"])
    if uploaded_file is None:
        st.write("Por favor, sube un archivo Excel para continuar.")
        return

    try:
        df = load_excel_bytes(uploaded_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    fincas = get_fincas(df)
    selected_finca = None
    working_df = df
    if fincas:
        selected_finca = st.selectbox("Finca", fincas)
        working_df = df[df["Finca"].astype(str) == str(selected_finca)].copy()

    varieties = get_varieties(working_df)
    if not varieties:
        st.error("No hay variedades disponibles para la finca seleccionada.")
        return

    selected_var = st.selectbox("Variedad a proyectar", varieties)

    st.write(f"Cantidad total de registros: {len(working_df)}")
    if selected_finca:
        st.write(f"Finca seleccionada: {selected_finca}")

    st.subheader("Consumo de API")
    _render_api_section(uploaded_file, selected_var)

    st.subheader("Proyección local")
    _render_local_projection(working_df, selected_var)


if __name__ == "__main__":
    main()
