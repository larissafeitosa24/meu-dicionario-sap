import streamlit as st
import pandas as pd
import re, unicodedata, difflib
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
ARQUIVO_BASE = "transacoes_sap_expandido_prefixo.xlsx"
ABA = "Planilha1"
MODELO = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def normalize(txt: str) -> str:
    if not isinstance(txt, str):
        txt = "" if pd.isna(txt) else str(txt)
    t = txt.strip().lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"\s+", " ", t)
    return t

def tokenize_set(txt: str) -> set:
    return set(re.findall(r"\w+", normalize(txt)))

def destacar_termos(texto: str, consulta: str) -> str:
    termos = consulta.lower().split()
    for termo in termos:
        texto = re.sub(rf"({re.escape(termo)})", r"**\1**", texto, flags=re.IGNORECASE)
    return texto

# -----------------------------
# CARREGAMENTO
# -----------------------------
@st.cache_data
def carregar_excel(caminho: str, aba: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(caminho, sheet_name=aba)
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo '{caminho}' (aba '{aba}'): {e}")
        return None
    return df

df = carregar_excel(ARQUIVO_BASE, ABA)

if df is not None and len(df) > 0:
    # Garante colunas
    if "frases_alternativas" not in df.columns:
        df["frases_alternativas"] = ""

    df["frases_alternativas"] = df["frases_alternativas"].fillna("").astype(str)

    # Normalizações
    df["_desc_norm"] = df["descricao"].apply(normalize)
    df["_alt_list_norm"] = df["frases_alternativas"].apply(
        lambda s: [normalize(p) for p in s.split(";") if p.strip()]
    )
    df["_token_set"] = df["_desc_norm"].apply(tokenize_set)

    # Mapa para descrição oficial por código
    code_to_desc = dict(zip(df["codigo"], df["descricao"]))

    # Preparação embeddings
    descricoes = [normalize(d) for d in df["descricao"].tolist()]
    codigos = df["codigo"].tolist()
    modulos = df.get("modulo", [""] * len(df)).tolist()
    saps = df.get("sap_system", [""] * len(df)).tolist()
    embeddings = MODELO.encode(descricoes, convert_to_tensor=True)

    # -----------------------------
    # ENTRADA DO USUÁRIO
    # -----------------------------
    consulta = st.text_input("O que você deseja fazer?")
    threshold_exato = st.slider("Limite para Expandido", 0.70, 0.95, 0.85, 0.01)
    threshold_semantica = st.slider("Limite para Semântico", 0.0, 1.0, 0.35, 0.01)

    if consulta:
        consulta_raw = consulta.strip()
        qn = normalize(consulta_raw)
        qtokens = tokenize_set(consulta_raw)

        # -------- 1) EXPANDIDO --------
        mask_equal_desc = (df["_desc_norm"] == qn)
        mask_equal_alt = df["_alt_list_norm"].apply(lambda lst: qn in lst)
        equal_hits = df[mask_equal_desc | mask_equal_alt]

        def strip_prefix(q: str) -> str:
            for pref in ("transacao para ", "transacao que "):
                if q.startswith(pref):
                    return q[len(pref):]
            return q

        qn_no_pref = strip_prefix(qn)
        mask_pref_desc = (df["_desc_norm"] == qn_no_pref)
        mask_pref_alt = df["_alt_list_norm"].apply(lambda lst: qn_no_pref in lst)
        pref_hits = df[mask_pref_desc | mask_pref_alt] if qn != qn_no_pref else df.iloc[0:0]

        df["__overlap__"] = df["_token_set"].apply(lambda s: len(s & qtokens))
        overlap_hits = df[df["__overlap__"] >= 2].copy()
        if not overlap_hits.empty:
            overlap_hits["__sim__"] = overlap_hits["descricao"].apply(
                lambda d: difflib.SequenceMatcher(None, normalize(d), qn).ratio()
            )
            overlap_hits.sort_values("__sim__", ascending=False, inplace=True)

        if not equal_hits.empty:
            out = equal_hits[["descricao", "codigo", "modulo", "sap_system"]].drop_duplicates("codigo")
            st.success(f"{len(out)} resultado(s) encontrados (Expandido)")
            st.dataframe(out, use_container_width=True)

        elif not pref_hits.empty:
            out = pref_hits[["descricao", "codigo", "modulo", "sap_system"]].drop_duplicates("codigo")
            st.success(f"{len(out)} resultado(s) encontrados (Expandido com prefixo)")
            st.dataframe(out, use_container_width=True)

        elif not overlap_hits.empty:
            best = overlap_hits.iloc[[0]][["descricao", "codigo", "modulo", "sap_system"]]
            st.success("1 resultado encontrado (Expandido por overlap)")
            st.dataframe(best, use_container_width=True)

        else:
            # -------- 2) SEMÂNTICO --------
            consulta_emb = MODELO.encode(consulta_raw, convert_to_tensor=True)
            scores = util.cos_sim(consulta_emb, embeddings)[0]

            resultados = sorted(
                zip(descricoes, codigos, modulos, saps, scores),
                key=lambda x: float(x[4]),
                reverse=True
            )

            best_per_code = {}
            for desc_phrase, cod, mod, sap, score in resultados:
                s = float(score)
                if s >= threshold_semantica:
                    if cod not in best_per_code or s > best_per_code[cod]["score"]:
                        best_per_code[cod] = {"score": s, "mod": mod, "sap": sap}

            if best_per_code:
                rows = []
                for cod, info in sorted(best_per_code.items(), key=lambda it: it[1]["score"], reverse=True):
                    desc_oficial = code_to_desc.get(cod, "")
                    rows.append({
                        "Descrição": destacar_termos(desc_oficial, consulta_raw),
                        "Transação": cod,
                        "Módulo": (info["mod"] if info["mod"] else "—"),
                        "SAP": (info["sap"] if info["sap"] else "—"),
                    })
                df_out = pd.DataFrame(rows)
                st.success(f"{len(df_out)} transações encontradas (Semântico, deduplicado)")
                st.markdown(df_out.to_markdown(index=False), unsafe_allow_html=True)
            else:
                st.warning("Nenhum resultado encontrado.")
