import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(page_title="Dicionário SAP Inteligente", page_icon="🤖")
st.title("🤖 Dicionário de Transações SAP (IA local)")
st.write("Pesquise em linguagem natural e veja a transação SAP correspondente.")

# -----------------------------
# PARÂMETROS
# -----------------------------
arquivo_base = "transacoes_sap.xlsx"  # nome do Excel no mesmo diretório do script
modelo = SentenceTransformer("all-MiniLM-L6-v2")  # modelo de embeddings
threshold = 0.40  # nota de corte mínima de similaridade

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
@st.cache_data
def carregar_excel():
    """Lê o Excel com as transações SAP"""
    df = pd.read_excel(arquivo_base)
    df.columns = df.columns.str.strip().str.lower()

    if "descrição" not in df.columns or "código" not in df.columns:
        st.error("❌ A planilha deve conter as colunas 'Descrição' e 'Código'.")
        return None

    df = df.dropna(subset=["descrição", "código"])
    df["descrição"] = df["descrição"].astype(str).str.strip()
    df["código"] = df["código"].astype(str).str.strip()
    return df

def expandir_descricoes(df):
    """Expande descrições separadas por vírgula em múltiplas instruções"""
    descricoes, codigos = [], []
    for _, row in df.iterrows():
        partes = [d.strip() for d in str(row["descrição"]).split(",")]
        for desc in partes:
            if desc:
                descricoes.append(desc.lower())
                codigos.append(row["código"])
    return descricoes, codigos

@st.cache_resource
def preparar_embeddings(df):
    """Cria embeddings a partir das descrições"""
    descricoes, codigos = expandir_descricoes(df)
    embeddings = modelo.encode(descricoes, convert_to_tensor=True)
    return descricoes, codigos, embeddings

# -----------------------------
# EXECUÇÃO PRINCIPAL
# -----------------------------
df = carregar_excel()

if df is not None:
    descricoes, codigos, embeddings = preparar_embeddings(df)

    consulta = st.text_input("O que você deseja fazer?")

    if consulta:
        consulta_emb = modelo.encode(consulta, convert_to_tensor=True)

        # Similaridade entre a consulta e todas as descrições
        scores = util.cos_sim(consulta_emb, embeddings)[0]
        top_k = min(5, len(descricoes))

        resultados = sorted(
            zip(descricoes, codigos, scores),
            key=lambda x: x[2],
            reverse=True
        )[:top_k]

        # Verifica se há resultados válidos
        melhor_score = float(resultados[0][2].item()) if len(resultados) > 0 else 0

        if melhor_score < threshold:
            st.error("❌ Nenhuma transação correspondente encontrada.")
            st.warning(
                f"""
                ➡️ Base utilizada: **{"transacoes_sap.xlsx"}**  

                Para adicionar uma nova transação:  
                1. Edite a planilha em 'https://github.com/larissafeitosa24/meu-dicionario-sap/tree/main'.  
                2. Inclua uma nova linha com:  
                   - **Descrição** (palavras-chave, separadas por vírgula)  
                   - **Código SAP**  
                3. Salve e recarregue o app.  
                """
            )
        else:
            st.info(f"🔎 Resultados para: **{consulta}**")
            for desc, cod, score in resultados:
                if float(score) >= threshold:
                    st.write(f"- {desc} → **{cod}**  (confiança: {float(score):.2f})")

