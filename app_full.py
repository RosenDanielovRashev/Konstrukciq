
import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO
import plotly.graph_objs as go

@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={
        "E1_over_E2": "Ed_over_Ei",
        "Eeq_over_E2": "Ee_over_Ei"
    })
    return df

data = load_data()

def compute_Ed(h, D, Ee, Ei):
    hD = h / D
    EeEi = Ee / Ei
    tol = 1e-4
    iso_levels = sorted(data['Ee_over_Ei'].unique())

    for low, high in zip(iso_levels, iso_levels[1:]):
        if not (low - tol <= EeEi <= high + tol):
            continue

        grp_low = data[data['Ee_over_Ei'] == low].sort_values('h_over_D')
        grp_high = data[data['Ee_over_Ei'] == high].sort_values('h_over_D')

        h_min = max(grp_low['h_over_D'].min(), grp_high['h_over_D'].min())
        h_max = min(grp_low['h_over_D'].max(), grp_high['h_over_D'].max())
        if not (h_min - tol <= hD <= h_max + tol):
            continue

        y_low = np.interp(hD, grp_low['h_over_D'], grp_low['Ed_over_Ei'])
        y_high = np.interp(hD, grp_high['h_over_D'], grp_high['Ed_over_Ei'])

        frac = 0 if np.isclose(high, low) else (EeEi - low) / (high - low)
        ed_over_ei = y_low + frac * (y_high - y_low)

        return ed_over_ei * Ei, hD, y_low, y_high, low, high

    return None, None, None, None, None, None

st.set_page_config(layout="wide")
st.title("📐 Оразмеряване пласт по пласт с номограма")

NUM_LAYERS = 4

if "current_layer" not in st.session_state:
    st.session_state.current_layer = 1
if "results" not in st.session_state:
    st.session_state.results = []
if "layer_done" not in st.session_state:
    st.session_state.layer_done = False
if "completed" not in st.session_state:
    st.session_state.completed = False

D = st.selectbox("Диаметър на отпечатъка D (cm)", options=[34.0, 32.04], index=1)

if not st.session_state.completed:
    st.header(f"Пласт {st.session_state.current_layer} от {NUM_LAYERS}")

    Ee = st.number_input("Ee (MPa)", value=160.0 + 100 * (st.session_state.current_layer - 1), key="Ee")
    Ei = st.number_input("Ei (MPa)", value=1000.0 - 100 * (st.session_state.current_layer - 1), key="Ei")
    h = st.number_input("h (cm)", value=4.0 + 2 * (st.session_state.current_layer - 1), key="h")

    EeEi = Ee / Ei

    st.write("### Въведени параметри:")
    st.write(pd.DataFrame({
        "Параметър": ["Ee", "Ei", "h", "D", "Ee / Ei", "h / D"],
        "Стойност": [Ee, Ei, h, D, round(EeEi, 3), round(h / D, 3)]
    }))

    if not st.session_state.layer_done:
        if st.button("Изчисли Ed"):
            result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, D, Ee, Ei)

            if result is None:
                st.warning("❗ Точката е извън обхвата.")
            else:
                EdEi_point = result / Ei
                st.success(f"✅ Ed / Ei = {EdEi_point:.3f}, Ed = {result:.2f} MPa")
                st.info(f"Интерполация: Ee / Ei = {low_iso:.3f} → {high_iso:.3f}")

                st.session_state.results.append({
                    "Пласт": st.session_state.current_layer,
                    "Ee": Ee,
                    "Ei": Ei,
                    "h": h,
                    "h/D": round(h / D, 3),
                    "Ee/Ei": round(EeEi, 3),
                    "Ed": round(result, 2),
                    "Ed/Ei": round(EdEi_point, 3)
                })
                st.session_state.layer_done = True

    if st.session_state.layer_done and st.session_state.current_layer < NUM_LAYERS:
        if st.button("Напред към следващ пласт"):
            st.session_state.layer_done = False
            st.session_state.current_layer += 1

    if st.session_state.layer_done and st.session_state.current_layer == NUM_LAYERS:
        if st.button("Приключи и покажи резултати"):
            st.session_state.completed = True

if st.session_state.completed:
    st.success("✅ Всички пластове са въведени.")
    st.subheader("📊 Обобщена таблица")
    df = pd.DataFrame(st.session_state.results)
    st.table(df)

    def generate_pdf(df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", size=12)
        pdf.cell(200, 10, txt="Оразмеряване на пътна конструкция", ln=True)
        pdf.ln(5)

        headers = ["Пласт", "Ee", "Ei", "h", "h/D", "Ee/Ei", "Ed", "Ed/Ei"]
        col_widths = [20, 20, 20, 15, 20, 20, 20, 20]

        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, border=1)
        pdf.ln()

        for _, row in df.iterrows():
            for col, w in zip(headers, col_widths):
                pdf.cell(w, 8, str(row[col]), border=1)
            pdf.ln()

        buffer = BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer.read()

    pdf_bytes = generate_pdf(df)
    st.download_button("⬇️ Изтегли PDF", data=pdf_bytes, file_name="results.pdf")
