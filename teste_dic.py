import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(page_title="Localizador de Transações SAP – Neoenergia", page_icon="⚡")
st.image("neo_logo.png", width=180)
st.title("⚡ Localizador de Transações SAP – Neoenergia")
st.write(
    "Este aplicativo foi desenvolvido para apoiar os auditores da Neoenergia na execução de suas atividades, "
    "facilitando a localização da transação SAP mais adequada para cada necessidade. "
    "Digite abaixo o que deseja encontrar."
)

# -----------------------------
# PARÂMETROS
# -----------------------------
ARQUIVO_BASE = "transacoes_sap.xlsx"
ABA = "Sheet1"
MODELO = SentenceTransformer("all-MiniLM-L6-v2")

THRESHOLD_SEMANTICA = 0.45  # limite fixo para semântica

COL_VARIANTS = {
    "descricao": {"descrição", "descricao", "description", "desc"},
    "codigo": {"transação", "transacao", "código", "codigo", "tcode"},
    "modulo": {"módulo", "modulo", "module"},
    "sap_system": {"sap", "sistema", "sap_system", "sap alvo", "target_sap"},
    "frases_alternativas": {"frases_alternativas", "variacoes", "sinonimos"}
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

    for c in ["descricao", "codigo", "modulo", "sap_system", "frases_alternativas"]:
        if c not in df.columns:
            df[c] = ""

    df["codigo"] = df["codigo"].astype(str).str.upper()
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
    return df

def expandir_descricoes(df: pd.DataFrame):
    descricoes, codigos, modulos, saps = [], [], [], []
    for _, row in df.iterrows():
        base_textos = [str(row["descricao"])]
        if row.get("frases_alternativas"):
            base_textos += str(row["frases_alternativas"]).split(";")

        for desc in base_textos:
            desc = desc.strip().lower()
            if desc:
                descricoes.append(desc)
                codigos.append(row["codigo"])
                modulos.append(row.get("modulo", "") or "")
                saps.append(row.get("sap_system", "") or "")
    return descricoes, codigos, modulos, saps

@st.cache_resource
def preparar_embeddings(df: pd.DataFrame):
    descricoes, codigos, modulos, saps = expandir_descricoes(df)
    embeddings = MODELO.encode(descricoes, convert_to_tensor=True)
    return descricoes, codigos, modulos, saps, embeddings

# -----------------------------
# EXECUÇÃO DO APP
# -----------------------------
df = carregar_excel(ARQUIVO_BASE, ABA)

if df is not None and len(df) > 0:
    descricoes, codigos, modulos, saps, embeddings = preparar_embeddings(df)
    consulta = st.text_input("O que você deseja fazer?")

    if consulta:
        consulta_lower = consulta.lower().strip()

        # 🔹 1. Verificação de correspondências exatas / alternativas
        matches_expandido = df[
            (df["descricao"].str.lower().str.contains(consulta_lower, na=False)) |
            (df["frases_alternativas"].str.lower().str.contains(consulta_lower, na=False))
        ]

        if len(matches_expandido) >= 1:
            if (
                len(matches_expandido) > 1 or 
                consulta_lower.startswith("transação para") or
                consulta_lower.startswith("transação que")
            ):
                # 🔹 Retorna expandido (1 ou mais matches fortes)
                st.dataframe(
                    matches_expandido[["descricao", "codigo", "modulo", "sap_system"]],
                    use_container_width=True
                )
            else:
                # Apenas 1 correspondência forte → expandido também
                st.dataframe(
                    matches_expandido[["descricao", "codigo", "modulo", "sap_system"]],
                    use_container_width=True
                )

        else:
            # 🔹 2. Caso não encontre nada → vai para semântica
            consulta_emb = MODELO.encode(consulta, convert_to_tensor=True)
            scores = util.cos_sim(consulta_emb, embeddings)[0]

            resultados = sorted(
                zip(descricoes, codigos, modulos, saps, scores),
                key=lambda x: float(x[4]),
                reverse=True
            )

            dados_tabela = []
            for desc, cod, mod, sap, score in resultados:
                if float(score) >= THRESHOLD_SEMANTICA:
                    dados_tabela.append({
                        "Descrição": desc,
                        "Transação": cod,
                        "Módulo": (mod if mod else "—"),
                        "SAP": (sap if sap else "—")
                    })

            if dados_tabela:
                st.dataframe(pd.DataFrame(dados_tabela), use_container_width=True)
            else:
                st.warning("Nenhum resultado encontrado.")



