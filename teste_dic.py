import nltk
nltk.download('rslp', quiet=True)
import streamlit as st
import pandas as pd
import re, unicodedata, difflib, numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.stem import RSLPStemmer

# -----------------------------
# AJUSTE DE LAYOUT (CSS CUSTOM)
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }
    .stTextInput, .stMultiSelect {
        margin-top: -0.3rem;
        margin-bottom: 0.5rem;
    }
    img {
        margin-bottom: -0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# CONFIGURAÇÃO DO APP
# -----------------------------
st.set_page_config(page_title="Localizador de Transações SAP – Neoenergia", page_icon="⚡")
st.image("neo_logo.png", width=180)
st.title("⚡ Localizador de Transações SAP – Neoenergia")
st.write(
    "Este aplicativo apoia os auditores da Neoenergia na localização da transação SAP mais adequada. "
    "Selecione um filtro de palavra-chave para começar ou digite o que deseja encontrar."
)

# -----------------------------
# PARÂMETROS / MODELO
# -----------------------------
ARQUIVO_BASE = "transacoes_sap.xlsx"
ABA = "Sheet1"
MODELO = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
stemmer = RSLPStemmer()

# -----------------------------
# SINÔNIMOS DE ALTO IMPACTO
# -----------------------------
SINONIMOS = {
    "exibir": ["mostrar", "visualizar", "consultar","Abrir","Acessar","Visualizar","Mostrar","Ver"],
    "mostrar": ["exibir", "visualizar", "consultar"],
    "pedido": ["ordem", "requisição", "compra"],
    "compras": ["pedido", "requisição", "ordem"],
    "contrato": ["acordo", "fornecedor", "negociação"],
    "criar": ["Gerar","Produzir","Montar","Construir","Inserir","criar"],
    "modificar": ["Editar","Alterar","Atualizar","Ajustar","Revisar","modificar"],
    "analisar": ["Analisar","Auditar","Verificar","Monitorar","Avaliar"],
    "listar": ["relação de","catalogo de","tabela de","coleção de","registro de"]
}

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
        return 0.50
    elif n <= 3:
        return 0.47
    else:
        return 0.45

def stem(texto):
    try:
        return stemmer.stem(texto.lower())
    except:
        return texto.lower()

def expandir_termos(q: str) -> str:
    tokens = q.split()
    expandido = []
    for t in tokens:
        expandido.append(t)
        if t in SINONIMOS:
            expandido.extend(SINONIMOS[t])
    return " ".join(expandido)

def aplicar_filtro(df, selecionadas):
    if not selecionadas:
        return df
    
    if "Grupo" not in df.columns:
        return df
    return df[df["Grupo"].isin(selecionadas)]

# -----------------------------
# CARREGAMENTO E EMBEDDINGS (CACHE)
# -----------------------------
@st.cache_data
def carregar_excel(caminho: str, aba: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(caminho, sheet_name=aba)
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo '{caminho}' (aba '{aba}'): {e}")
        return None
    return df

@st.cache_data
def gerar_embeddings(textos):
    return MODELO.encode(textos, convert_to_tensor=True, normalize_embeddings=True)

df = carregar_excel(ARQUIVO_BASE, ABA)

if df is not None and len(df) > 0:
    if "frases_alternativas" not in df.columns:
        df["frases_alternativas"] = ""

    df["frases_alternativas"] = df["frases_alternativas"].fillna("").astype(str)

    if "modulo" not in df.columns:
        df["modulo"] = ""
    mostrar_modulo = not df["modulo"].fillna("").str.strip().eq("").all()

    df["_desc_norm"] = df["descricao"].apply(normalize)
    df["_alt_list_norm"] = df["frases_alternativas"].apply(
        lambda s: [normalize(p) for p in s.split(";") if p.strip()]
    )
    df["_token_set"] = df["_desc_norm"].apply(tokenize_set)

    code_to_desc = dict(zip(df["codigo"], df["descricao"]))
    code_to_group = dict(zip(df["codigo"],df["Grupo"]))

    descricoes = [stem(normalize(d)) for d in df["descricao"].tolist()]
    codigos = df["codigo"].tolist()
    modulos = df.get("modulo", [""] * len(df)).tolist()
    saps = df.get("sap_system", [""] * len(df)).tolist()
    embeddings = gerar_embeddings(descricoes)

    # -----------------------------
    # ENTRADAS DO USUÁRIO
    # -----------------------------
    opcoes_filtro = [
        "Auditoria/Segurança", "Programa", "Compras", "Contratos",
        "Orçamento(FM)", "Planejamento (CO/PS)", "Projetos(PS)", "Materiais & Estoque", "Contábil","Outros",
        "Relatórios Z-Corporativos","Tesouraria /Cobrança(FI-CA)","Manutenção (PM)","Contábil(FI)","IS-U (Comercial/Medidores)",
        "Vendas & Faturamento (SD)","Documentos & Saída (DMS/Spool)","Custos/Controladoria (CO)",
        "Basis/Tecnico","Cadastros Mestre (BP/MM/SD)","Ativo Fixo (FI-AA)","Qualidade (QM)","Documentos (DMS/Spool)"
    ]
    filtro_multiselect = st.multiselect("🔍 Filtro por palavra-chave", opcoes_filtro)
    consulta = st.text_input("🧠 Busca livre (opcional)")

    if filtro_multiselect or consulta.strip():
        consulta_raw = consulta.strip()
        qn = normalize(consulta_raw)
        qn_expandido = expandir_termos(qn)
        qtokens = tokenize_set(consulta_raw)

        if not consulta.strip():
            st.info("🔎 Exibindo resultados com base apenas nos filtros aplicados.")
            df_filtrado = aplicar_filtro(df, filtro_multiselect)
            if not df_filtrado.empty:
                st.success(f"{len(df_filtrado)} transações encontradas (Filtro direto)")
                st.dataframe(df_filtrado, use_container_width=True)
            else:
                st.warning("Nenhuma transação encontrada com esses filtros.")
            st.stop()

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

        cols_base = ["Grupo","descricao", "codigo", "sap_system"]
        cols_show = cols_base + ["modulo"] if mostrar_modulo else cols_base

        if not equal_hits.empty:
            out = equal_hits[cols_show].drop_duplicates("codigo")
            out = aplicar_filtro(out, filtro_multiselect)
            st.success(f"{len(out)} resultado(s) encontrados (Expandido)")
            st.dataframe(out, use_container_width=True)

        elif not pref_hits.empty:
            out = pref_hits[cols_show].drop_duplicates("codigo")
            out = aplicar_filtro(out, filtro_multiselect)
            st.success(f"{len(out)} resultado(s) encontrados (Expandido com prefixo)")
            st.dataframe(out, use_container_width=True)

        elif not overlap_hits.empty:
            best = overlap_hits.iloc[[0]][cols_show]
            best = aplicar_filtro(best, filtro_multiselect)
            st.success("1 resultado encontrado (Expandido por overlap)")
            st.dataframe(best, use_container_width=True)

        else:
            # -------- 2) SEMÂNTICO --------
            consulta_emb = MODELO.encode(qn_expandido, convert_to_tensor=True, normalize_embeddings=True)
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
                bonus_literal = 0.1 if qn in desc_phrase else 0.0
                bonus_stem = 0.07 if stem(qn) in [stem(w) for w in desc_phrase.split()] else 0.0
                s_final = s + bonus_literal + bonus_stem

                if s_final >= threshold_semantica or bonus_literal > 0:
                    if cod not in best_per_code or s_final > best_per_code[cod]["score"]:
                        best_per_code[cod] = {"score": s_final, "mod": mod, "sap": sap}

            if best_per_code:
                rows = []
                for cod, info in sorted(best_per_code.items(), key=lambda it: it[1]["score"], reverse=True):
                    desc_oficial = code_to_desc.get(cod, "")
                    base_row = {
                        "Grupo": code_to_group.get(cod,""),
                        "descricao": desc_oficial,
                        "Transação": cod,
                        "SAP": (info["sap"] if info["sap"] else "—"),
                    }
                    if mostrar_modulo:
                        base_row["Módulo"] = (info["mod"] if info["mod"] else "—")
                    rows.append(base_row)

                df_out = pd.DataFrame(rows)
                df_out = aplicar_filtro(df_out, filtro_multiselect)
                if not df_out.empty:
                    st.success(f"{len(df_out)} transações encontradas (Semântico aprimorado)")
                    df_out["Descrição"] = df_out["descricao"].apply(lambda d: destacar_termos(d, consulta_raw))
                    colunas_final = ["Descrição", "Transação", "SAP"]
                    if mostrar_modulo:
                        colunas_final.insert(2, "Módulo")
                    st.markdown(df_out[colunas_final].to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.warning("Nenhum resultado após aplicar o filtro.")
            else:
                st.warning("Nenhum resultado encontrado.")


