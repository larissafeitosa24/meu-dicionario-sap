
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

# -----------------------------
# CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(page_title="Dicionário SAP Inteligente", page_icon="🤖")
st.title("🤖 Dicionário de Transações SAP")
st.write("Pesquise em linguagem natural e veja a transação SAP correspondente.")

# -----------------------------
# PARÂMETROS
# -----------------------------
ARQUIVO_BASE = "transacoes_sap.xlsx"  # seu arquivo atual (com espaço no nome)
ABA = "Planilha1"           # aba detectada no arquivo
MODELO = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.50
TOP_K = 5

# Variações aceitas de nomes de colunas no Excel
COL_VARIANTS = {
    "descricao": {"descrição", "descricao", "description", "desc"},
    "codigo": {"transação", "transacao", "código", "codigo", "tcode"},
    "modulo": {"módulo", "modulo", "module"},
    "sap_system": {"sap", "sistema", "sap_system", "sap alvo", "target_sap"},
}

def _normaliza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for padrao, variantes in COL_VARIANTS.items():
        for v in variantes:
            if v in base_cols:
                ren[base_cols[v]] = padrao
                break
    df = df.rename(columns=ren)

    for c in ["descricao", "codigo", "modulo", "sap_system"]:
        if c not in df.columns:
            df[c] = pd.NA

    for c in ["descricao", "codigo", "modulo", "sap_system"]:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["None", "nan", "NaN"]), c] = ""
    df["codigo"] = df["codigo"].str.upper()
    return df

@st.cache_data
def carregar_excel(caminho: str, aba: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(caminho, sheet_name=aba)
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo '{caminho}' (aba '{aba}'): {e}")
        return None

    df.columns = df.columns.str.strip().str.lower()
    df = _normaliza_colunas(df)
    df = df.dropna(subset=["descricao", "codigo"])
    df = df[(df["descricao"].str.len() > 0) & (df["codigo"].str.len() > 0)]
    return df

def expandir_descricoes(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    descricoes, codigos, modulos, saps = [], [], [], []
    for _, row in df.iterrows():
        partes = [d.strip() for d in str(row["descricao"]).split(",")]
        if not partes:
            partes = [str(row["descricao"]).strip()]
        for desc in partes:
            if desc:
                descricoes.append(desc.lower())
                codigos.append(row["codigo"])
                modulos.append(row.get("modulo", "") or "")
                saps.append(row.get("sap_system", "") or "")
    return descricoes, codigos, modulos, saps

@st.cache_resource
def preparar_embeddings(df: pd.DataFrame):
    descricoes, codigos, modulos, saps = expandir_descricoes(df)
    embeddings = MODELO.encode(descricoes, convert_to_tensor=True)
    return descricoes, codigos, modulos, saps, embeddings

df = carregar_excel(ARQUIVO_BASE, ABA)

if df is not None and len(df) > 0:
    descricoes, codigos, modulos, saps, embeddings = preparar_embeddings(df)
    consulta = st.text_input("O que você deseja fazer?")
    if consulta:
        consulta_emb = MODELO.encode(consulta, convert_to_tensor=True)
        scores = util.cos_sim(consulta_emb, embeddings)[0]
        k = min(TOP_K, len(descricoes))
        resultados = sorted(
            zip(descricoes, codigos, modulos, saps, scores),
            key=lambda x: float(x[4]),
            reverse=True
        )[:k]
        melhor_score = float(resultados[0][4]) if len(resultados) > 0 else 0.0
        if melhor_score < THRESHOLD:
            st.error("❌ Nenhuma transação correspondente encontrada.")
        else:
            st.info(f"🔎 Resultados para: **{consulta}** (threshold={THRESHOLD:.2f})")
            dados_tabela = []
            for desc, cod, mod, sap, score in resultados:
                if float(score) >= THRESHOLD:
                    dados_tabela.append({
                        "Descrição (match)": desc,
                        "Transação": cod,
                        "Módulo": (mod if mod else "—"),
                        "SAP": (sap if sap else "—"),
                        "Confiança": round(float(score), 2)
                    })
            if dados_tabela:
                st.dataframe(pd.DataFrame(dados_tabela), use_container_width=True)
            else:
                st.warning("Nenhum resultado acima do threshold.")


