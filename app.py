import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="San Miguel",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# ESTILOS
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.92rem;
        color: #6c757d;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #212529;
    }
    .stock-card {
        background-color: #fff4ee;
        border: 1px solid #ffd3bd;
        padding: 14px 16px;
        border-radius: 14px;
    }
    .stock-title {
        font-size: 0.82rem;
        color: #e85d1c;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .stock-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #212529;
        line-height: 1;
        margin-top: 4px;
    }
    .stock-sub {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 6px;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #e85d1c;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    div[data-testid="stTabs"] button {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def fmt_num(x, dec=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{dec}f}"

def safe_upper(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()

# Columnas nativas del modelo V1 (orden de visualización)
COLS_MODELO_V1 = [
    "CODIGO", "DESCRIPCION", "MATERIAL", "CEDRO",
    "STOCK_ULT", "FECHA_ULT", "N_MESES_HIST",
    "MAC", "DAC", "SDDC", "VDC", "YC", "UPD",
    "EPSO", "NPMY", "ALTD", "LTV", "RULT", "VCRT", "SDC",
    "SS", "RP", "MAX", "NLPA",
    "ACCION", "CANTIDAD_SUGERIDA",
]

COLS_MODELO_V2 = [
    "CODIGO", "DESCRIPCION", "MATERIAL", "CEDRO", "CATEGORIA",
    "STOCK_ULT", "FECHA_ULT",
    "N_MESES_HIST", "N_MESES_CON_CONSUMO", "MESES_DESDE_ULTIMO_CONSUMO",
    "TASA_ACTIVIDAD", "CONSUMO_PROMEDIO_ACTIVO",
    "MAC", "DAC", "SS", "RP", "MAX", "EPSO",
    "ACCION", "CANTIDAD_SUGERIDA",
]

LEAD_TIME_V2_DIAS = 30

# =========================================================
# CONEXIÓN A SUPABASE
# =========================================================
@st.cache_resource
def get_engine():
    return create_engine(
        st.secrets["database"]["url"],
        poolclass=NullPool,
        pool_pre_ping=True
    )

# =========================================================
# CARGA DESDE POSTGRES
# =========================================================
@st.cache_data(show_spinner="Cargando histórico del SKU...", ttl=600)
def load_historico_codigo(codigo: str) -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT *
        FROM public.historico
        WHERE codigo = %(codigo)s
        ORDER BY fecha
    """

    df = pd.read_sql(query, engine, params={"codigo": str(codigo)})
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

    for c in ["CONSUMO", "STOCK", "STOCK_ANT", "GTQ"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"STOCK", "STOCK_ANT", "CONSUMO"}.issubset(df.columns):
        entrada = df["STOCK"].fillna(0) - df["STOCK_ANT"].fillna(0) + df["CONSUMO"].fillna(0)
        df["ENTRADA"] = entrada.clip(lower=0)
    else:
        df["ENTRADA"] = np.nan

    return df


@st.cache_data(show_spinner="Cargando modelo V1...", ttl=600)
def load_modelo_v1() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql('SELECT * FROM niveles_inventario', engine)
    df.columns = [str(c).strip().upper() for c in df.columns]

    num_cols = [
        "STOCK_ULT", "N_MESES_HIST", "MAC", "DAC", "SDDC", "VDC", "YC",
        "UPD", "EPSO", "NPMY", "ALTD", "LTV", "RULT", "VCRT", "SDC",
        "SS", "RP", "MAX", "NLPA", "CANTIDAD_SUGERIDA",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "FECHA_ULT" in df.columns:
        df["FECHA_ULT"] = pd.to_datetime(df["FECHA_ULT"], errors="coerce")

    for c in ["CODIGO", "MATERIAL", "DESCRIPCION", "CEDRO", "ACCION"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    cols_final = [c for c in COLS_MODELO_V1 if c in df.columns]
    return df[cols_final].copy()


@st.cache_data(show_spinner="Cargando modelo V2...", ttl=600)
def load_modelo_v2() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql('SELECT * FROM niveles_inventario_v2', engine)
    df.columns = [str(c).strip().upper() for c in df.columns]

    num_cols = [
        "STOCK_ULT", "N_MESES_HIST", "N_MESES_CON_CONSUMO",
        "MESES_DESDE_ULTIMO_CONSUMO", "TASA_ACTIVIDAD", "CONSUMO_PROMEDIO_ACTIVO",
        "MAC", "DAC", "SS", "RP", "MAX", "EPSO", "CANTIDAD_SUGERIDA",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "FECHA_ULT" in df.columns:
        df["FECHA_ULT"] = pd.to_datetime(df["FECHA_ULT"], errors="coerce")

    for c in ["CODIGO", "MATERIAL", "DESCRIPCION", "CEDRO", "CATEGORIA", "ACCION"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    cols_final = [c for c in COLS_MODELO_V2 if c in df.columns]
    return df[cols_final].copy()


@st.cache_data(show_spinner="Cargando matrices...", ttl=600)
def load_matrices():
    engine = get_engine()
    m_cant = pd.read_sql('SELECT * FROM matriz_cantidad', engine)
    m_monto = pd.read_sql('SELECT * FROM matriz_monto', engine)

    # Reconstruir el formato matriz: poner ABC como índice
    if "ABC" in m_cant.columns:
        m_cant = m_cant.set_index("ABC")
    if "ABC" in m_monto.columns:
        m_monto = m_monto.set_index("ABC")

    m_cant.columns = [str(c) for c in m_cant.columns]
    m_monto.columns = [str(c) for c in m_monto.columns]

    return m_cant, m_monto


# =========================================================
# CARGA INICIAL
# =========================================================
try:
    df_hist = pd.DataFrame()
    df_modelo_v1 = load_modelo_v1()
    df_modelo_v2 = load_modelo_v2()
    matriz_cantidad, matriz_monto = load_matrices()
except Exception as e:
    st.error(f"Error al conectar con la base de datos: {e}")
    st.stop()

for _df in (df_hist, df_modelo_v1, df_modelo_v2):
    if "CODIGO" in _df.columns:
        _df["CODIGO"] = _df["CODIGO"].astype(str)

# =========================================================
# RENDER: BASE COMPLETA (genérico por modelo)
# =========================================================
def render_base_completa(df_modelo: pd.DataFrame, version_key: str, version_label: str):
    st.markdown(f"#### Base completa — modelo {version_label}")

    search_base = st.text_input(
        "Buscar por código o descripción",
        value="",
        key=f"search_base_{version_key}",
    ).strip()

    base_total = df_modelo.copy()
    if search_base:
        q = search_base.upper()
        mask = safe_upper(base_total["CODIGO"]).str.contains(q, regex=False)
        if "DESCRIPCION" in base_total.columns:
            mask = mask | safe_upper(base_total["DESCRIPCION"]).str.contains(q, regex=False)
        base_total = base_total[mask]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card("Total códigos", f"{len(base_total):,}")
        st.caption("Materiales en el modelo.")

    with k2:
        if "ACCION" in base_total.columns:
            acciones = base_total["ACCION"].fillna("").astype(str).str.strip()
            con_accion = (acciones != "").sum()
            metric_card("Con acción asignada", f"{con_accion:,}")
            st.caption("Materiales con recomendación.")

    with k3:
        if "CANTIDAD_SUGERIDA" in base_total.columns:
            metric_card(
                "Compra sugerida total",
                fmt_num(base_total["CANTIDAD_SUGERIDA"].sum(), 0),
            )
            st.caption("Unidades totales sugeridas.")

    with k4:
        if {"UPD", "STOCK_ULT"}.issubset(base_total.columns):
            valor_inv = (base_total["UPD"] * base_total["STOCK_ULT"]).sum()
            metric_card("Valor inventario GTQ", f"Q {fmt_num(valor_inv, 0)}")
            st.caption("Precio unitario × stock último.")
        elif "CATEGORIA" in base_total.columns:
            n_cat = base_total["CATEGORIA"].nunique()
            metric_card("Categorías de rotación", f"{n_cat}")
            st.caption("Segmentación del modelo v2.")

    tiene_categoria = "CATEGORIA" in base_total.columns

    if tiene_categoria:
        col_dist1, col_dist2 = st.columns(2)
    else:
        col_dist1 = st.container()
        col_dist2 = None
    with col_dist1:
        if "ACCION" in base_total.columns and len(base_total):
            st.markdown("##### Distribución por acción")
            dist_accion = (
                base_total["ACCION"].fillna("(sin acción)").value_counts().reset_index()
            )
            dist_accion.columns = ["ACCION", "Conteo"]
            fig_accion = px.bar(dist_accion, x="ACCION", y="Conteo", text="Conteo", color="ACCION")
            fig_accion.update_traces(textposition="outside", showlegend=False)
            fig_accion.update_layout(
                xaxis_title=None, yaxis_title="Códigos",
                margin=dict(l=10, r=10, t=20, b=10), height=260, showlegend=False,
            )
            st.plotly_chart(fig_accion, use_container_width=True, key=f"fig_accion_{version_key}")
    if col_dist2 is not None:
        with col_dist2:
            if "CATEGORIA" in base_total.columns and len(base_total):
                st.markdown("##### Distribución por categoría")
                dist_cat = (
                    base_total["CATEGORIA"].fillna("(sin cat.)").value_counts().reset_index()
                )
                dist_cat.columns = ["CATEGORIA", "Conteo"]
                fig_cat = px.bar(dist_cat, x="CATEGORIA", y="Conteo", text="Conteo", color="CATEGORIA")
                fig_cat.update_traces(textposition="outside", showlegend=False)
                fig_cat.update_layout(
                    xaxis_title=None, yaxis_title="Códigos",
                    margin=dict(l=10, r=10, t=20, b=10), height=260, showlegend=False,
                )
                st.plotly_chart(fig_cat, use_container_width=True, key=f"fig_cat_{version_key}")

    st.markdown("---")

    st.markdown("##### Tabla completa")
    st.dataframe(base_total, use_container_width=True, height=600)

    csv_download = base_total.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV",
        data=csv_download,
        file_name=f"niveles_inventario_{version_key}.csv",
        mime="text/csv",
        key=f"dl_{version_key}",
    )


# =========================================================
# RENDER: ANÁLISIS POR CÓDIGO
# =========================================================
def render_analisis_codigo(df_modelo: pd.DataFrame, df_hist: pd.DataFrame,
                           version_key: str, version_label: str):
    orden_acciones = [
        "COMPRA URGENTE", "COMPRAR", "OK", "SOBRE-STOCK", "SIN MOVIMIENTO 2024+",
    ]
    acciones_presentes = df_modelo["ACCION"].dropna().astype(str).unique().tolist()
    acciones_ordenadas = (
        [a for a in orden_acciones if a in acciones_presentes]
        + sorted([a for a in acciones_presentes if a not in orden_acciones])
    )

    fcol1, _ = st.columns([1.5, 4.5])
    with fcol1:
        st.markdown('<div class="kpi-label">FILTRAR POR ACCIÓN</div>', unsafe_allow_html=True)
        accion_sel = st.selectbox(
            label="Acción",
            options=["Todos"] + acciones_ordenadas,
            index=0,
            key=f"accion_filter_{version_key}",
            label_visibility="collapsed",
        )

    if accion_sel != "Todos":
        df_modelo_f = df_modelo[df_modelo["ACCION"].astype(str) == accion_sel].copy()
    else:
        df_modelo_f = df_modelo.copy()

    if df_modelo_f.empty:
        st.warning(f"No hay SKUs con acción '{accion_sel}'.")
        return

    st.caption(f"Mostrando **{len(df_modelo_f):,}** SKUs con acción **{accion_sel}**.")

    top1, top2, top3, top4 = st.columns([2.2, 1.2, 1.2, 1.2])

    with top1:
        st.markdown('<div class="kpi-label">PRODUCTO (SKU)</div>', unsafe_allow_html=True)
        df_sel = df_modelo_f.copy()
        if "DESCRIPCION" in df_sel.columns:
            df_sel["LABEL"] = df_sel["CODIGO"] + " - " + df_sel["DESCRIPCION"].fillna("")
        else:
            df_sel["LABEL"] = df_sel["CODIGO"]

        label_sel = st.selectbox(
            label="Producto",
            options=df_sel["LABEL"].tolist(),
            key=f"sku_{version_key}",
            label_visibility="collapsed",
        )
        row_sel = df_sel[df_sel["LABEL"] == label_sel].iloc[0]
        codigo_sel = row_sel["CODIGO"]

    df_hist_sel = load_historico_codigo(codigo_sel)
    if "FECHA" in df_hist_sel.columns:
        df_hist_sel = df_hist_sel.sort_values("FECHA")

    if not df_hist_sel.empty and df_hist_sel["FECHA"].notna().any():
        fecha_min = df_hist_sel["FECHA"].min().date()
        fecha_max = df_hist_sel["FECHA"].max().date()
    else:
        fecha_min = pd.Timestamp("2024-01-01").date()
        fecha_max = pd.Timestamp("2025-12-31").date()

    with top2:
        st.markdown('<div class="kpi-label">FECHA INICIO</div>', unsafe_allow_html=True)
        f_ini = st.date_input(
            "Inicio", value=fecha_min, min_value=fecha_min, max_value=fecha_max,
            key=f"f_ini_{version_key}_{codigo_sel}", label_visibility="collapsed",
        )
    with top3:
        st.markdown('<div class="kpi-label">FECHA FIN</div>', unsafe_allow_html=True)
        f_fin = st.date_input(
            "Fin", value=fecha_max, min_value=fecha_min, max_value=fecha_max,
            key=f"f_fin_{version_key}_{codigo_sel}", label_visibility="collapsed",
        )

    stock_actual = row_sel.get("STOCK_ULT", np.nan)
    if "ALTD" in df_modelo.columns and pd.notna(row_sel.get("ALTD", np.nan)):
        lead_dias = int(row_sel["ALTD"])
        lead_txt = f"Lead Time: {lead_dias} días"
    else:
        lead_dias = LEAD_TIME_V2_DIAS
        lead_txt = f"Lead Time: {lead_dias} días (supuesto)"

    with top4:
        st.markdown(
            f"""
            <div class="stock-card">
                <div class="stock-title">STOCK ACTUAL</div>
                <div class="stock-value">{fmt_num(stock_actual, 0)}</div>
                <div class="stock-sub">{lead_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if str(row_sel.get("ACCION", "")).strip().upper() == "SIN MOVIMIENTO 2024+":
        st.warning(
            "Este SKU no registra consumo desde 2024. "
            "El modelo no calcula niveles MAX / RP / SS para códigos sin movimiento "
            "de consumo. Se muestra el histórico completo "
            "(Stock, Consumo, Tránsito) sólo para referencia."
        )

    if "FECHA" in df_hist_sel.columns:
        mask = (df_hist_sel["FECHA"].dt.date >= f_ini) & (df_hist_sel["FECHA"].dt.date <= f_fin)
        df_hist_plot = df_hist_sel[mask].copy()
    else:
        df_hist_plot = df_hist_sel.copy()

    st.markdown("---")

    desc = row_sel.get("DESCRIPCION", "")
    titulo = f"### Historial de Consumo — {codigo_sel}"
    if desc:
        titulo += f"  ·  {desc}"
    if "CATEGORIA" in df_modelo.columns:
        cat = row_sel.get("CATEGORIA", "")
        if cat:
            titulo += f"  ·  *{cat}*"
    st.markdown(titulo)

    ref_max = row_sel.get("MAX", np.nan)
    ref_rp = row_sel.get("RP", np.nan)
    ref_ss = row_sel.get("SS", np.nan)

    fig = go.Figure()

    if not df_hist_plot.empty:
        if "CONSUMO" in df_hist_plot.columns:
            fig.add_trace(
                go.Bar(
                    x=df_hist_plot["FECHA"],
                    y=df_hist_plot["CONSUMO"].fillna(0),
                    name="Consumo",
                    marker_color="#1f2d3d",
                    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Consumo: <b>%{y:,.0f}</b><extra></extra>",
                )
            )
        if "ENTRADA" in df_hist_plot.columns:
            fig.add_trace(
                go.Bar(
                    x=df_hist_plot["FECHA"],
                    y=df_hist_plot["ENTRADA"].fillna(0),
                    name="Tránsito (entradas)",
                    marker_color="#f5a623",
                    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Entrada: <b>%{y:,.0f}</b><extra></extra>",
                )
            )
        if "STOCK" in df_hist_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_hist_plot["FECHA"], y=df_hist_plot["STOCK"],
                    name="Stock", mode="lines+markers",
                    line=dict(color="#e85d1c", width=3, shape="hv"),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Stock: <b>%{y:,.0f}</b><extra></extra>",
                )
            )
        x_ref_min = df_hist_plot["FECHA"].min()
        x_ref_max = df_hist_plot["FECHA"].max()
        if pd.notna(ref_max):
            fig.add_trace(go.Scatter(
                x=[x_ref_min, x_ref_max], y=[ref_max, ref_max],
                name="Nivel Máximo", mode="lines",
                line=dict(color="#2ecc71", width=2, dash="dash"),
                hovertemplate="MAX: <b>%{y:,.0f}</b><extra></extra>",
            ))
        if pd.notna(ref_rp):
            fig.add_trace(go.Scatter(
                x=[x_ref_min, x_ref_max], y=[ref_rp, ref_rp],
                name="Punto de Reorden", mode="lines",
                line=dict(color="#1e88e5", width=2, dash="dash"),
                hovertemplate="RP: <b>%{y:,.0f}</b><extra></extra>",
            ))
        if pd.notna(ref_ss):
            fig.add_trace(go.Scatter(
                x=[x_ref_min, x_ref_max], y=[ref_ss, ref_ss],
                name="Stock de Seguridad", mode="lines",
                line=dict(color="#8e44ad", width=2, dash="dash"),
                hovertemplate="SS: <b>%{y:,.0f}</b><extra></extra>",
            ))

    fig.update_layout(
        barmode="group",
        hovermode="x unified",
        xaxis=dict(title=""),
        yaxis=dict(title="Unidades"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"fig_{version_key}")

    if df_hist_plot.empty:
        st.info("No hay movimientos para este SKU en el rango seleccionado.")

    tcol1, tcol2 = st.columns(2)

    with tcol1:
        if not df_hist_plot.empty and "FECHA" in df_hist_plot.columns:
            fechas_disp = sorted(df_hist_plot["FECHA"].dt.date.unique().tolist())
            idx_default = len(fechas_disp) - 1
            fecha_detalle = st.selectbox(
                "Fecha del detalle",
                options=fechas_disp,
                index=idx_default,
                format_func=lambda d: d.strftime("%Y-%m-%d"),
                key=f"fecha_detalle_{version_key}",
            )
            fila = df_hist_plot[df_hist_plot["FECHA"].dt.date == fecha_detalle]
            if not fila.empty:
                fila = fila.iloc[0]
                datos = [
                    ("Fecha", fecha_detalle.strftime("%Y-%m-%d")),
                    ("SKU", codigo_sel),
                    ("Consumo", fmt_num(fila.get("CONSUMO", np.nan), 0)),
                    ("Stock", fmt_num(fila.get("STOCK", np.nan), 0)),
                    ("Tránsito (entrada)", fmt_num(fila.get("ENTRADA", np.nan), 0)),
                    ("RP", fmt_num(ref_rp, 0)),
                    ("SS", fmt_num(ref_ss, 0)),
                    ("MAX", fmt_num(ref_max, 0)),
                ]
                tabla_datos = pd.DataFrame(datos, columns=["Atributo", "Valor"])
                st.markdown(
                    f'<div class="kpi-label">TABLA DE DATOS — {fecha_detalle.strftime("%Y-%m-%d")}</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(tabla_datos, use_container_width=True, hide_index=True, height=330)
        else:
            st.info("Sin fechas disponibles en el rango.")

    with tcol2:
        st.markdown(
            f'<div class="kpi-label">CÁLCULO DE NIVELES — {codigo_sel} ({version_label})</div>',
            unsafe_allow_html=True,
        )
        niveles = [
            ("SKU", codigo_sel),
            ("Descripción", str(row_sel.get("DESCRIPCION", "-"))),
        ]
        if "CATEGORIA" in df_modelo.columns:
            niveles.append(("Categoría", str(row_sel.get("CATEGORIA", "-"))))

        niveles += [
            ("MAC (prom. mensual)", fmt_num(row_sel.get("MAC", np.nan), 2)),
            ("DAC (prom. diario)", fmt_num(row_sel.get("DAC", np.nan), 4)),
        ]

        for label_, col in [
            ("SDDC (desv. diaria)", "SDDC"),
            ("VDC (varianza diaria)", "VDC"),
            ("YC (demanda anual)", "YC"),
        ]:
            if col in df_modelo.columns:
                niveles.append((label_, fmt_num(row_sel.get(col, np.nan), 4 if col != "YC" else 2)))

        for label_, col, dec in [
            ("Meses con consumo", "N_MESES_CON_CONSUMO", 0),
            ("Meses desde último consumo", "MESES_DESDE_ULTIMO_CONSUMO", 0),
            ("Tasa de actividad", "TASA_ACTIVIDAD", 2),
            ("Consumo promedio activo", "CONSUMO_PROMEDIO_ACTIVO", 2),
        ]:
            if col in df_modelo.columns:
                niveles.append((label_, fmt_num(row_sel.get(col, np.nan), dec)))

        niveles += [
            ("RP (punto de reorden)", fmt_num(ref_rp, 0)),
            ("SS (stock seguridad)", fmt_num(ref_ss, 0)),
            ("MAX (nivel máximo)", fmt_num(ref_max, 0)),
            ("EPSO (lote económico)", fmt_num(row_sel.get("EPSO", np.nan), 0)),
            ("Lead Time", f"{lead_dias} días"),
            ("Acción sugerida", str(row_sel.get("ACCION", "-"))),
            ("Cantidad sugerida", fmt_num(row_sel.get("CANTIDAD_SUGERIDA", np.nan), 0)),
        ]
        tabla_niveles = pd.DataFrame(niveles, columns=["Atributo", "Valor"])
        st.dataframe(tabla_niveles, use_container_width=True, hide_index=True, height=560)

# =========================================================
# HEADER
# =========================================================
st.title("Inventario - San Miguel")
st.caption("Modelos v1 (estocástico, desde 2024) y v2 (segmentado por rotación)")

# Mostrar usuario logueado (solo aparece en Streamlit Cloud)
if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
    st.sidebar.caption(f"Conectada como: {st.user.email}")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Ir a sección",
    ["Matrices", "Base completa", "Análisis por código"],
)

# =========================================================
# MATRICES
# =========================================================
if menu == "Matrices":
    st.subheader("Último Movimiento")

    matriz_cantidad_view = matriz_cantidad.drop(columns=["<=2020"], errors="ignore").copy()
    matriz_monto_view = matriz_monto.drop(columns=["<=2020"], errors="ignore").copy()

    cmat1, cmat2 = st.columns(2)

    with cmat1:
        st.markdown("### Matriz de cantidad")
        st.dataframe(matriz_cantidad_view, use_container_width=True)
        fig_mat_cant = px.imshow(
            matriz_cantidad_view, text_auto=True, aspect="auto",
            title="Cantidad de códigos por ABC y año",
        )
        st.plotly_chart(fig_mat_cant, use_container_width=True)

    with cmat2:
        st.markdown("### Matriz de monto (GTQ)")
        matriz_monto_display = matriz_monto_view.copy()
        for c in matriz_monto_display.columns:
            matriz_monto_display[c] = matriz_monto_display[c].apply(
                lambda x: f"Q {x:,.2f}" if pd.notna(x) else ""
            )
        st.dataframe(matriz_monto_display, use_container_width=True)
        fig_mat_monto = px.imshow(
            matriz_monto_view, text_auto=True, aspect="auto",
            title="Monto por ABC y año",
        )
        st.plotly_chart(fig_mat_monto, use_container_width=True)

# =========================================================
# BASE COMPLETA — con subtabs V1 / V2
# =========================================================
elif menu == "Base completa":
    st.subheader("Base Completa")
    sub_v1, sub_v2 = st.tabs(["V1 — estocástico", "V2 — segmentado"])

    with sub_v1:
        render_base_completa(df_modelo_v1, version_key="v1", version_label="V1")

    with sub_v2:
        render_base_completa(df_modelo_v2, version_key="v2", version_label="V2 segmentado")

# =========================================================
# ANÁLISIS POR CÓDIGO — con subtabs V1 / V2
# =========================================================
elif menu == "Análisis por código":
    st.subheader("Análisis por código")
    sub_v1, sub_v2 = st.tabs(["V1 — estocástico", "V2 — segmentado"])

    with sub_v1:
        render_analisis_codigo(df_modelo_v1, df_hist,
                               version_key="v1", version_label="V1")

    with sub_v2:
        render_analisis_codigo(df_modelo_v2, df_hist,
                               version_key="v2", version_label="V2 segmentado")