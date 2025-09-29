import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Dicionário SAP Inteligente", page_icon="🤖")
st.title("🤖 Dicionário de Transações SAP (IA local)")
st.write("Pesquise em linguagem natural e veja a transação SAP correspondente.")

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
arquivo_base = "transacoes_sap.xlsx"
modelo = SentenceTransformer("all-MiniLM-L6-v2")  # modelo leve e rápido
threshold = 0.60 # nota de corte minima de similaridade

# -----------------------------
# CARREGAR PLANILHA
# -----------------------------
@st.cache_data
def carregar_excel():
    df = pd.read_excel(arquivo_base)
    df.columns = df.columns.str.strip().str.lower()

    if "descrição" not in df.columns or "código" not in df.columns:
        st.error("❌ A planilha deve conter as colunas 'Descrição' e 'Código'.") # testa pra verificar se a planilha está ok
        return None

    df = df.dropna(subset=["descrição", "código"])
    df["descrição"] = df["descrição"].astype(str).str.strip()
    df["código"] = df["código"].astype(str).str.strip()
    return df

df = carregar_excel()

# -----------------------------
# EXPANDIR DESCRIÇÕES (vírgulas viram várias instruções)
# -----------------------------
def expandir_descricoes(df):
    descricoes, codigos = [], []
    for _, row in df.iterrows():
        partes = [d.strip() for d in str(row["descrição"]).split(",")]
        for desc in partes:
            if desc:  # ignora vazio
                descricoes.append(desc.lower())
                codigos.append(row["código"])
    return descricoes, codigos

# -----------------------------
# EMBEDDINGS
# -----------------------------
@st.cache_resource
def preparar_embeddings(df):
    descricoes, codigos = expandir_descricoes(df)
    embeddings = modelo.encode(descricoes, convert_to_tensor=True)
    return descricoes, codigos, embeddings

if df is not None:
    descricoes, codigos, embeddings = preparar_embeddings(df)

    # -----------------------------
    # CAMPO DE BUSCA
    # -----------------------------
    consulta = st.text_input("O que você deseja fazer?")

    if consulta:
        consulta_emb = modelo.encode(consulta, convert_to_tensor=True)

        # Calcular similaridade
        scores = util.cos_sim(consulta_emb, embeddings)[0]
        top_k = min(5, len(descricoes))  # mostrar até 5 melhores
        resultados = sorted(
            zip(descricoes, codigos, scores),
            key=lambda x: x[2],
            reverse=True
        )[:top_k]

    melhor_score = float(resultados[0][2].item()) if len(resultados) > 0 else 0

    if melhor_score < threshold: 
        st.error(" Nenhuma transação correspondente encontrada.")
        st.warning (
                f"""
                Base utilizada : **{"transacoes_sap.xlsx"}
                Para adicionar uma nova transação :
                1. Edite a planilha que está salva aqui : 'https://github.com/larissafeitosa24/meu-dicionario-sap'
                2. Inclua uma nova linha com :
                - **Descrição**( palavras-chave separada por virgula)
                - **Código SAP**
                3. Salve e recarregue o app
                """
            )

    else :
        st.info(f"🔎 Resultados para: **{consulta}**")
        for desc, cod, score in resultados:
            st.write(f"- {desc} → **{cod}**  (confiança: {score:.2f})")
