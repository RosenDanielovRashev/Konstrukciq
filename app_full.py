
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from fpdf import FPDF
from io import BytesIO

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

def update_preview(title, D, p, En, layers, results):
    lines = [f"{title}", "", f"Осов товар: 100 kN", f"Диаметър D: {D:.2f} см", f"Налягане p: {p:.3f} MPa", f"Необходим En: {En:.1f} MPa", ""]
    for idx, (layer, res) in enumerate(zip(layers, results), 1):
        lines.append(f"Пласт {idx}: Ee = {layer['Ee']} MPa | Ei = {layer['Ei']} MPa | h = {layer['h']} см | Ed = {res['Ed (MPa)']}")
    return "\n".join(lines)

def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=12)
    for line in text.split("\n"):
        for wrapped_line in textwrap.wrap(line, width=90):
            pdf.cell(200, 8, txt=wrapped_line, ln=True)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer.read()

def draw_layers(layers):
    fig = go.Figure()
    y = 0
    colors = ['#444444', '#888888', '#BBBBBB', '#DDDDDD']

    for i, layer in enumerate(layers):
        h = layer['h']
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[y, y, y - h, y - h, y],
            fill='toself',
            name=f"Пласт {i+1}",
            fillcolor=colors[i % len(colors)],
            line=dict(color='black'),
            hoverinfo='text',
            text=f"h = {h} cm"
        ))
        y -= h

    fig.update_layout(
        height=500,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Дълбочина [cm]", autorange='reversed'),
        showlegend=True,
        margin=dict(t=30)
    )
    return fig

st.set_page_config(layout="wide")
st.title("📐 Оразмеряване на пътна конструкция")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("🧾 Въвеждане на данни")

    title = st.text_input("Заглавие", value="Оразмеряване на пътната конструкция за 10 т/ос")
    D = st.number_input("Диаметър на отпечатъка D (cm)", value=32.04)
    p = st.number_input("Налягане на гумите p (MPa)", value=0.62)
    En = st.number_input("Необходим модул на еластичност En (MPa)", value=160.0)

    num_layers = st.number_input("Брой пластове", min_value=1, max_value=10, value=3, step=1)

    layers = []
    for i in range(int(num_layers)):
        st.markdown(f"### Пласт {i + 1}")
        Ee = st.number_input(f"Ee{i+1} (MPa)", key=f"ee{i}", value=160.0 + i * 100)
        Ei = st.number_input(f"E{i+1} (MPa)", key=f"ei{i}", value=1000.0 - i * 200)
        h = st.number_input(f"h{i+1} (cm)", key=f"h{i}", value=4.0 + i * 2)
        layers.append({'Ee': Ee, 'Ei': Ei, 'h': h})

    st.subheader("📊 Резултати от изчисленията:")

    results = []
    for i, layer in enumerate(layers, 1):
        Ee, Ei, h = layer['Ee'], layer['Ei'], layer['h']
        Ed, hD, y_low, y_high, iso_low, iso_high = compute_Ed(h, D, Ee, Ei)
        results.append({
            "Ee (MPa)": Ee,
            "Ei (MPa)": Ei,
            "Ee/Ei": round(Ee / Ei, 3),
            "h (cm)": h,
            "h/D": round(h / D, 3),
            "Ed (MPa)": round(Ed, 2) if Ed else "❌",
            "Ed/Ei": round(Ed / Ei, 3) if Ed else "❌"
        })

    st.dataframe(pd.DataFrame(results))

    if st.button("Обнови визуализация"):
        st.session_state['doc'] = update_preview(title, D, p, En, layers, results)

    if st.button("Свали PDF"):
        pdf_bytes = generate_pdf(st.session_state.get('doc', ""))
        st.download_button("⬇️ Изтегли PDF", data=pdf_bytes, file_name="orazmerqvane.pdf")

with col2:
    st.header("📄 Преглед на документа")
    preview = st.session_state.get('doc', "Попълни данни отляво и натисни 'Обнови визуализация'")
    st.text_area("Преглед", preview, height=500)

    st.subheader("📐 Схема на конструкцията")
    fig = draw_layers(layers)
    st.plotly_chart(fig, use_container_width=True)
