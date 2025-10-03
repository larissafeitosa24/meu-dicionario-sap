

import streamlit as st
import pandas as pd
import re
import unicodedata
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

# -----------------------------
# CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(page_title="Localizador de Transações SAP – Neoenergia", page_icon="⚡")

# CSS para reduzir espaço do topo
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;   /* padrão é ~5rem, aqui reduzimos */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo + título alinhados
col1, col2 = st.columns([1, 4])  # proporção 1:4 (logo / título)
with col1:
    st.image("neo_logo.png", width=100)
with col2:
    st.title("⚡ Localizador de Transações SAP – Neoenergia")

# Subtítulo
st.write(
    "Este aplicativo foi desenvolvido para apoiar os auditores da Neoenergia na execução de suas atividades, "
    "facilitando a localização da transação SAP mais adequada para cada necessidade. "
    "Digite abaixo o que deseja encontrar."
)

# -----------------------------
# PARÂMETROS
# -----------------------------
ARQUIVO_BASE = "transacoes_sap.xlsx"
ABA = "Planilha1"
MODELO = SentenceTransformer("all-MiniLM-L6-v2")

COL_VARIANTS = {
    "descricao": {"descrição", "descricao", "description", "desc"},
    "codigo": {"transação", "transacao", "código", "codigo", "tcode"},
    "modulo": {"módulo", "modulo", "module"},
    "sap_system": {"sap", "sistema", "sap_system", "sap alvo", "target_sap"},
}

# -----------------------------
# FUNÇÕES DE APOIO
# -----------------------------
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

def destacar_termos(texto: str, consulta: str) -> str:
    termos = consulta.lower().split()
    for termo in termos:
        texto = re.sub(rf"({termo})", r"**\1**", texto, flags=re.IGNORECASE)
    return texto

# -----------------------------
# EXECUÇÃO DO APP
# -----------------------------
df = carregar_excel(ARQUIVO_BASE, ABA)

if df is not None and len(df) > 0:
    descricoes, codigos, modulos, saps, embeddings = preparar_embeddings(df)
    consulta = st.text_input("O que você deseja fazer?")

    # Sliders para calibrar thresholds
    threshold_exato = st.slider("Limite para Exato Expandido", 0.70, 0.95, 0.85, 0.01)
    threshold_semantica = st.slider("Limite para Busca Semântica", 0.0, 1.0, 0.35, 0.01)

    if consulta:
        consulta_emb = MODELO.encode(consulta, convert_to_tensor=True)
        scores = util.cos_sim(consulta_emb, embeddings)[0]

        resultados = sorted(
            zip(descricoes, codigos, modulos, saps, scores),
            key=lambda x: float(x[4]),
            reverse=True
        )

        if not resultados:
            st.error("❌ Nenhuma transação encontrada.")
        else:
            melhor_score = float(resultados[0][4])

            # 1) Exato expandido
            if melhor_score >= threshold_exato:
                desc, cod, mod, sap, score = resultados[0]
                dados_tabela = [{
                    "Descrição": desc,
                    "Transação": cod,
                    "Módulo": (mod if mod else "—"),
                    "SAP": (sap if sap else "—")
                }]
                st.caption("🔎 Modo de busca: **Exato expandido**")
                st.dataframe(pd.DataFrame(dados_tabela), use_container_width=True)

            # 2) Busca semântica
            else:
                dados_tabela = []
                for desc, cod, mod, sap, score in resultados:
                    if float(score) >= threshold_semantica:
                        desc_destacada = destacar_termos(desc, consulta)
                        dados_tabela.append({
                            "Descrição": desc_destacada,
                            "Transação": cod,
                            "Módulo": (mod if mod else "—"),
                            "SAP": (sap if sap else "—")
                        })
                if dados_tabela:
                    st.caption("🔎 Modo de busca: **Semântica**")
                    st.markdown(pd.DataFrame(dados_tabela).to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.warning("Nenhum resultado acima do threshold definido.")
