import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

st.set_page_config(page_title="Dashboard Empresa", layout="wide")
st.title("Dashboard Interno")

@st.cache_resource
def get_engine():
    return create_engine(
        st.secrets["database"]["url"],
        connect_args={"sslmode": "require"},
    )

@st.cache_data(ttl=600)
def cargar_tabla(nombre_tabla):
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {nombre_tabla}", engine)

# Usuario logueado (solo aparece en Streamlit Cloud, en local no)
if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
    st.caption(f"Conectada como: {st.user.email}")

# Selector de tabla
tabla = st.selectbox(
    "Selecciona una tabla:",
    ["matrices_abc", "niveles_inventario_v2", "niveles_inventario", "resultado_combinado"],
)

try:
    df = cargar_tabla(tabla)
    st.success(f"Cargados {len(df):,} registros de '{tabla}'")
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"Error al cargar la tabla: {e}")