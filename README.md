# FraudGuard: Detector de Anomalias em Transações Financeiras com IA

## Descrição do Projeto

**FraudGuard** O FraudGuard é uma Prova de Conceito (PoC) que valida a arquitetura e a tecnologia para um futuro sistema de Inteligência Artificial focado na detecção de anomalias em extratos bancários reais.

Este projeto serve como um estudo técnico para o desenvolvimento de um MVP (Produto Mínimo Viável). O objetivo aqui foi construir e testar a solução de ponta a ponta, desde o treinamento do modelo até sua disponibilização via API e dashboard interativo, utilizando um dataset público de transações financeiras como base.

O sistema utiliza um modelo de aprendizado não supervisionado (`IsolationForest`) para identificar padrões incomuns em um grande volume de dados, simulando a detecção de fraudes em tempo real.

## Features Principais

-   **Modelo de Detecção de Anomalias:** Treinamento de um modelo com `Scikit-learn` a partir de dados históricos em formato CSV.
-   **API RESTful:** Uma API de alta performance com `FastAPI` para fazer previsões em tempo real para transações individuais.
-   **Dashboard Interativo:** Uma interface visual com `Streamlit` para análise em lote, permitindo o upload de arquivos CSV e a visualização gráfica das transações suspeitas.
-   **Arquitetura Desacoplada:** Separação clara entre o processo de treinamento (offline) e o de inferência (online), uma prática recomendada para sistemas de MLOps.

## Arquitetura do Sistema

O projeto é dividido em dois processos principais:

1.  **Processo Offline (Treinamento):**
    -   Um script (`train_model.py`) lê um dataset histórico em CSV.
    -   Ele treina um modelo de detecção de anomalias.
    -   Salva os artefatos do modelo treinado (arquivos .joblib).

2.  **Processo Online (Previsão):**
    -   **API com FastAPI:** Carrega o modelo treinado e o expõe através de um endpoint. Recebe uma nova transação (em JSON) e retorna uma classificação (Normal/Suspeita).
    -   **Dashboard com Streamlit:** Carrega o modelo treinado e fornece uma interface para upload de um arquivo CSV. Processa o arquivo em lote e exibe as anomalias detectadas em tabelas e gráficos.

## Tech Stack

-   **Linguagem:** Python 3.9+
-   **Análise de Dados:** Pandas
-   **Machine Learning:** Scikit-learn
-   **API:** FastAPI & Uvicorn
-   **Dashboard:** Streamlit
-   **Serialização de Modelos:** Joblib

## Setup e Instalação

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

1.  **Clonar o Repositório:**
    `git clone https://github.com/Demuno/Fraud_Guard_AI.git`
    `cd FraudGuard`

2.  **Criar um Ambiente Virtual (Recomendado):**
    `python -m venv venv`
    `venv\Scripts\activate` (Para Windows)
    `source venv/bin/activate` (Para macOS/Linux)

3.  **Instalar as Dependências:**
    `pip install -r requirements.txt`

## Como Usar

O fluxo de trabalho do projeto consiste em treinar o modelo primeiro e depois utilizar a API ou o Dashboard.

### Passo 1: Obtenção dos Dados

1.  Baixe o dataset "Credit Card Fraud Detection" do Kaggle.
2.  Extraia o arquivo `creditcard.csv`.
3.  Mova-o para a pasta `data/` do projeto.
4.  Renomeie o arquivo para `transactions.csv`.

### Passo 2: Treinamento do Modelo

Execute o script de treinamento. Este processo lerá os dados, treinará o modelo e salvará os artefatos na pasta `models/`.

`python src/train_model.py`

### Passo 3: Utilização (Escolha uma das opções)

#### Opção A: Executando a API para Previsões em Tempo Real

Inicie o servidor da API com Uvicorn.

`uvicorn src.api:app --reload`

-   Acesse `http://127.0.0.1:8000/docs` para ver a documentação interativa e testar o endpoint `/predict`.

#### Opção B: Executando o Dashboard Interativo

Inicie a aplicação Streamlit.

`streamlit run src/dashboard.py`

-   O dashboard abrirá automaticamente no seu navegador.
-   Utilize o botão "Browse files" para fazer o upload do arquivo `data/transactions.csv` e visualizar a análise.

## Estrutura do Projeto

- FraudGuard/

        data/

            transactions.csv

        models/

            anomaly_model.joblib

            scaler.joblib

        src/

            train_model.py

            api.py

            dashboard.py

        requirements.txt

        README.md

## Próximos Passos e Melhorias

-   **Otimização de Performance:** Implementar pipelines de pré-processamento em batch para o dashboard.
-   **Adaptação para Novos Dados:** Desenvolver um fluxo de "Engenharia de Features" para adaptar o modelo a diferentes fontes de dados (ex: extratos bancários).
-   **Experimentação de Modelos:** Testar outros algoritmos de detecção de anomalias.
-   **Monitoramento:** Adicionar logging e monitoramento na API.
-   **Testes Automatizados:** Implementar testes unitários e de integração para a API.
-   **Validação com dados reais:**  Realizar a análise a partir de um extrato bancário para testar a aplicação em um cenário prático.
