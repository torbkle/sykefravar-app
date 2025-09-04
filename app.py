import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import Image

# Layout: logo og tittel
col1, col2 = st.columns([1, 4])
with col1:
    logo = Image.open("logo.png")  # Sørg for at logo.png ligger i samme mappe
    st.image(logo, width=100)
with col2:
    st.title("📊 Sykefraværsprediksjon")

# Last opp CSV-fil
uploaded_file = st.file_uploader("Last opp CSV-fil", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Dataoversikt", df.head())

    # Sjekk at nødvendige kolonner finnes
    required_cols = ["alder", "kjønn", "stillingsprosent", "antall_barn", "arbeidsmiljø_score", "jobbstress_score", "sykefravær"]
    if all(col in df.columns for col in required_cols):
        X = df[required_cols[:-1]]  # Alle unntatt 'sykefravær'
        y = df["sykefravær"]

        # Konverter kjønn til tall
        X = pd.get_dummies(X, drop_first=True)

        # Tren modell
        model = LinearRegression()
        model.fit(X, y)

        # Prediksjoner
        predictions = model.predict(X)
        df["Predikert sykefravær"] = predictions

        st.subheader("📈 Prediksjoner")
        st.write(df[["sykefravær", "Predikert sykefravær"]])

        # Evaluer modellen
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)

        st.markdown(f"**R²-score:** {r2:.2f}")
        st.markdown(f"**MAE:** {mae:.2f}")
    else:
        st.error("CSV-filen mangler nødvendige kolonner.")

# Om appen
st.markdown("---")
st.markdown("### 👨‍💻 Om denne appen")
st.markdown("""
Denne appen er utviklet av **Torbjørn Kleiven**.

Den bruker en lineær regresjonsmodell for å predikere sykefravær basert på faktorer som alder, kjønn, arbeidsmiljø og jobbstress.

📬 Kontakt: torbjoernkleiven@gmail.com
""")
