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
            color_map = {
                'Suspeita de Fraude': '#FF4B4B',
                'Transação Normal': '#0068C9'    
            }
    
            df['color'] = df['status'].map(color_map)
    
            st.scatter_chart(
                data=df,
                x='V4',
                y='Amount',
                color='color',
                size='Amount'
            )
            st.markdown("""
                <style>
                .legend-container { display: flex; flex-direction: column; align-items: flex-start; font-size: 14px; }
                .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
                .legend-color-box { width: 15px; height: 15px; margin-right: 10px; border: 1px solid #444; }
                </style>
                <div class="legend-container">
                    <b>Legenda:</b>
                    <div class="legend-item">
                        <div class="legend-color-box" style="background-color:#FF4B4B;"></div>
                        <span>Transação Suspeita</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color-box" style="background-color:#0068C9;"></div>
                        <span>Transação Normal</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"O arquivo CSV deve conter as colunas necessárias para a análise. Exemplo: {required_features[:3]}...")
