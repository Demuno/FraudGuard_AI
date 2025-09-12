import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Detector de Fraudes", layout="wide")
st.title("Dashboard de Detecção de Fraudes em Transações")

MODEL_PATH = os.path.join(".", "models", "anomaly_detection_model.joblib")
SCALER_PATH = os.path.join(".", "models", "scaler.joblib")

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")

model, scaler = load_model()

if model is None or scaler is None:
    st.error("Model or scaler could not be loaded. Please check the model files.")
else:
    st.success("Model and scaler loaded successfully.")

    st.header("Análise de Novas Transações")
    st.write("Faça o upload de um arquivo CSV com transações para que o modelo as analise.")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        df_completo = pd.read_csv(uploaded_file)

        if len(df_completo) > 15000:
            df = df_completo.sample(n=15000, random_state=42)
        else:
            df = df_completo

        st.info(f"Analisando uma amostra de {len(df):,} transações (de um total de {len(df_completo):,}).")
        
        st.write("Amostra dos dados carregados:")
        st.dataframe(df.head())
        required_features = [f'V{i}' for i in range(1, 29)] + ['Amount']
        if all(feature in df.columns for feature in required_features):
            with st.spinner("Analisando transações... Isso pode levar um momento..."):
                x = df[required_features]
                x_scaled = scaler.transform(x)

                df['is_anomaly'] = model.predict(x_scaled)
                df['status'] = df['is_anomaly'].apply(lambda x: "Suspeita de Fraude" if x == -1 else "Transação Normal")

                suspect_transactions = df[df['status'] == "Suspeita de Fraude"]
            
            st.success("Análise concluída! Foram encontradas **{len(suspect_transactions)}** transações suspeitas.")

            if not suspect_transactions.empty:
                st.subheader("Transações Suspeitas de Fraude")
                st.dataframe(suspect_transactions)
            
            st.subheader("Visualização Gráfica das Anomalias")
            st.write("Gráfico de dispersão do Valor (`Amount`) por uma das features (`V4`). Pontos em vermelho são as anomalias detectadas.")

            st.scatter_chart(df, x='V4', y='Amount', color='status', size='Amount')
        else:
            st.error(f"O arquivo CSV deve conter as colunas necessárias para a análise. Exemplo: {required_features[:3]}...")
