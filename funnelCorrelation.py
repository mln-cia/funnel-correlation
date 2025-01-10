import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
from sklearn.linear_model import LinearRegression
import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Funzione per generare l'Excel campione
def generate_sample_excel():
    data = {
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Awareness": [100, 120, 110],
        "Consideration": [80, 85, 88],
        "Purchase Intent": [50, 55, 52],
        "Current Customer": [30, 32, 31],
        "Recommend": [70, 72, 75],
        "Other YouGov Metrics": [None,None,None],
    }
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sample")
    return output.getvalue()

def calculate_r2_matrix(df):
    numeric_columns = df.select_dtypes(include=["number"])
    columns = numeric_columns.columns
    r2_matrix = pd.DataFrame(index=columns, columns=columns)
    
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                r2_matrix.loc[col1, col2] = 1.0  # Diagonale con R^2 = 1
            else:
                # Fitting del modello lineare
                x = numeric_columns[[col1]].values
                y = numeric_columns[col2].values
                model = LinearRegression()
                model.fit(x, y)
                r2 = model.score(x, y)  # Calcolo di R^2
                n = x.shape[0]  # Numero di osservazioni
                k = x.shape[1] - 1  # Numero di variabili indipendenti (escludendo la dipendente)
                r2_matrix.loc[col1, col2] =  1 - (1 - r2) * (n - 1) / (n - k - 1)

    return r2_matrix

def plot_heatmap(df, vmin, vmax):
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2%",
        cmap="coolwarm",
        cbar=True,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        mask=np.triu(np.ones_like(df, dtype=bool))                
        )
    st.pyplot(fig)


# Titolo dell'app
st.title("YouGov Data Funnel Correlation")

# Upload del file
st.markdown("### Load Excel YouGov Data")
sample_file = generate_sample_excel()
st.download_button(
    label="Download Sample File",
    data=sample_file,
    file_name="sample_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
uploaded_file = st.file_uploader("Carica un file Excel", type=["xlsx"])

# Se un file viene caricato
if uploaded_file is not None:
    try:
        # Legge i dati
        df = pd.read_excel(uploaded_file)
        st.write("Data Overview")
        st.dataframe(df)

        # Rimuove eventuali colonne non numeriche
        numeric_columns = df.select_dtypes(include=["number"])
        analysis_choice = st.radio("Select An option:", ["Correlation", "R^2"])

        if analysis_choice == "Correlation":
            if not numeric_columns.empty:
                st.write("Correlation between variables:")
                correlation_matrix = numeric_columns.corr()
                st.dataframe(correlation_matrix)

                vmin, vmax = st.slider('Select Range Scale of Legend', min_value=-1., max_value=1., value=(-1.,1.), step=.05)
                st.subheader("Heatmap")
                plot_heatmap(correlation_matrix, vmin, vmax)
                
                
            else:
                st.warning("No numeric columns, check input data.")
        elif analysis_choice == "R^2":
            if not numeric_columns.empty:
                st.write("R^2 Calculation:")
                r2_matrix = calculate_r2_matrix(df.fillna(0))
                r2_matrix = r2_matrix.astype(float).round(3)  # Formattazione della matrice
                st.dataframe(r2_matrix)

                st.subheader("Heatmap")
                plot_heatmap(r2_matrix)

            else:
                st.warning("No numeric columns, check input data.")


    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")

