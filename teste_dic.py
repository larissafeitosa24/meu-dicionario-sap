import nltk
nltk.download('rslp', quiet=True)
import streamlit as st
import pandas as pd
import re, unicodedata, difflib
from sentence_transformers import SentenceTransformer, util
from nltk.stem import RSLPStemmer


# -----------------------------
# CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(page_title="Localizador de Transações SAP – Neoenergia", page_icon="⚡")
st.markdown(
"""
<style>
/* Remove padding padrão do Streamlit */
.block-container {
padding-top: 1rem;
padding-bottom: 1rem;
}

/* Centraliza menos verticalmente e encurta espaço entre seções */
div[data-testid="stVerticalBlock"] {
gap: 0.6rem;
}

/* Deixa os inputs mais próximos do topo */
.stTextInput, .stMultiSelect {
margin-top: -0.3rem;
margin-bottom: 0.5rem;
}

/* Reduz o espaçamento abaixo da imagem/logo */
img {
margin-bottom: -0.5rem;
}
</style>
""",
unsafe_allow_html=True,
)
st.image("neo_logo.png", width=180)
st.title("⚡ Localizador de Transações SAP – Neoenergia")
st.write(
    "Este aplicativo foi desenvolvido para apoiar os auditores da Neoenergia na execução de suas atividades, "
    "facilitando a localização da transação SAP mais adequada para cada necessidade. "
    "Você pode realizar uma busca livre e, se desejar, usar os filtros de palavras-chave para refinar os resultados."
)

# -----------------------------
# PARÂMETROS
# -----------------------------
ARQUIVO_BASE = "transacoes_sap.xlsx"
ABA = "Sheet1"
MODELO = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
stemmer = RSLPStemmer()

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def normalize(txt: str) -> str:
    if not isinstance(txt, str):
        txt = "" if pd.isna(txt) else str(txt)
    t = txt.strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def tokenize_set(txt: str) -> set:
    return set(re.findall(r"\w+", normalize(txt)))

def destacar_termos(texto: str, consulta: str) -> str:
    termos = consulta.lower().split()
    for termo in termos:
        texto = re.sub(rf"({re.escape(termo)})", r"**\1**", texto, flags=re.IGNORECASE)
    return texto

def calcular_threshold(pergunta: str) -> float:
    n = len(pergunta.split())
    if n <= 1:
        return 0.60
    elif n <= 3:
        return 0.55
    else:
        return 0.50

def stem(texto):
    try:
        return stemmer.stem(texto.lower())
    except:
        return texto.lower()

def aplicar_filtro(df, selecionadas, livres):
    """Filtra por palavras do multiselect e campo livre"""
    palavras = []
    palavras += [normalize(p) for p in selecionadas]
    palavras += [normalize(p) for p in livres.split(";") if livres.strip()]
    if not palavras:
        return df
    mask = df["descricao"].apply(lambda d: all(p in normalize(d) for p in palavras))
    return df[mask]

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
    if "frases_alternativas" not in df.columns:
        df["frases_alternativas"] = ""

    df["frases_alternativas"] = df["frases_alternativas"].fillna("").astype(str)

    df["_desc_norm"] = df["descricao"].apply(normalize)
    df["_alt_list_norm"] = df["frases_alternativas"].apply(
        lambda s: [normalize(p) for p in s.split(";") if p.strip()]
    )
    df["_token_set"] = df["_desc_norm"].apply(tokenize_set)

    code_to_desc = dict(zip(df["codigo"], df["descricao"]))

    descricoes = [normalize(d) for d in df["descricao"].tolist()]
    codigos = df["codigo"].tolist()
    modulos = df.get("modulo", [""] * len(df)).tolist()
    saps = df.get("sap_system", [""] * len(df)).tolist()
    embeddings = MODELO.encode(descricoes, convert_to_tensor=True, normalize_embeddings=True)

    # -----------------------------
    # ENTRADAS DO USUÁRIO
    # -----------------------------
    consulta = st.text_input("🧠 O que você deseja fazer?")

    # 🔹 Opções pré-definidas de filtro
    opcoes_filtro = ["Auditoria", "Contrato", "Projeto", "Pagamento", "Orçamento", "Risco", "Controle", "Financeiro", "Compras", "Manutenção","Programa","Contábil"]
    filtro_multiselect = st.multiselect("🔍 Filtro por palavra-chave (opcional)", opcoes_filtro)
   
    threshold_exato = 0.85

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
            out = aplicar_filtro(out, filtro_multiselect, filtro_livre)
            st.success(f"{len(out)} resultado(s) encontrados (Expandido)")
            st.dataframe(out, use_container_width=True)

        elif not pref_hits.empty:
            out = pref_hits[["descricao", "codigo", "modulo", "sap_system"]].drop_duplicates("codigo")
            out = aplicar_filtro(out, filtro_multiselect, filtro_livre)
            st.success(f"{len(out)} resultado(s) encontrados (Expandido com prefixo)")
            st.dataframe(out, use_container_width=True)

        elif not overlap_hits.empty:
            best = overlap_hits.iloc[[0]][["descricao", "codigo", "modulo", "sap_system"]]
            best = aplicar_filtro(best, filtro_multiselect, filtro_livre)
            st.success("1 resultado encontrado (Expandido por overlap)")
            st.dataframe(best, use_container_width=True)

        else:
            # -------- 2) SEMÂNTICO APRIMORADO --------
            consulta_emb = MODELO.encode(qn, convert_to_tensor=True, normalize_embeddings=True)
            scores = util.cos_sim(consulta_emb, embeddings)[0].cpu().numpy()

            resultados = sorted(
                zip(descricoes, codigos, modulos, saps, scores),
                key=lambda x: float(x[4]),
                reverse=True
            )

            threshold_semantica = calcular_threshold(consulta_raw)
            best_per_code = {}

            for desc_phrase, cod, mod, sap, score in resultados:
                s = float(score)
                bonus_literal = 0.07 if qn in desc_phrase else 0.0
                if stem(qn) in [stem(w) for w in desc_phrase.split()]:
                    bonus_stem = 0.05
                else:
                    bonus_stem = 0.0
                s_final = s + bonus_literal + bonus_stem

                if s_final >= threshold_semantica or bonus_literal > 0:
                    if cod not in best_per_code or s_final > best_per_code[cod]["score"]:
                        best_per_code[cod] = {"score": s_final, "mod": mod, "sap": sap}

            if best_per_code:
                rows = []
                for cod, info in sorted(best_per_code.items(), key=lambda it: it[1]["score"], reverse=True):
                    desc_oficial = code_to_desc.get(cod, "")
                    rows.append({
                        "descricao": desc_oficial,
                        "Transação": cod,
                        "Módulo": (info["mod"] if info["mod"] else "—"),
                        "SAP": (info["sap"] if info["sap"] else "—"),
                    })
                df_out = pd.DataFrame(rows)
                df_out = aplicar_filtro(df_out, filtro_multiselect, filtro_livre)
                if not df_out.empty:
                    st.success(f"{len(df_out)} transações encontradas (Semântico aprimorado)")
                    df_out["Descrição"] = df_out["descricao"].apply(lambda d: destacar_termos(d, consulta_raw))
                    st.markdown(df_out[["Descrição","Transação","Módulo","SAP"]].to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.warning("Nenhum resultado após aplicar o filtro.")
            else:
                st.warning("Nenhum resultado encontrado.")




