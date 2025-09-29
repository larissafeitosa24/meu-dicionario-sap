import streamlit as st
import pandas as pd
from rapidfuzz import process

st.set_page_config(page_title="Dicionário SAP", page_icon="💻")
st.title("💻 Dicionário de Transações SAP")
st.write("Pesquise em linguagem natural e veja a transação SAP correspondente.")

# -----------------------------
# CARREGAR PLANILHA LOCAL (do repositório)
# -----------------------------
@st.cache_data
def carregar_excel():
 df = pd.read_excel("transacoes_sap.xlsx")
 return df

df = carregar_excel()

# -----------------------------
# PREPARAR DICIONÁRIO EXPANDIDO
# -----------------------------
transacoes = {}
for _, row in df.iterrows():
 descricoes = [d.strip().lower() for d in str(row["Descrição"]).split(",")]
for desc in descricoes:
 transacoes[desc] = row["Código"]

st.success(f"✅ {len(transacoes)} instruções carregadas com sucesso!")

# -----------------------------
# CAMPO DE BUSCA
# -----------------------------
acao = st.text_input("O que você deseja fazer?")

if acao:
        acao_proc = acao.lower()

        # 🔹 Mostrar todas relacionadas à palavra
        relacionados = {desc: cod for desc, cod in transacoes.items() if acao_proc in desc}
        if relacionados:
            st.info(f"📌 Transações relacionadas a '{acao}':")
            for d, c in relacionados.items():
                st.write(f"- {d} → **{c}**")

        else:
            # 🔹 Fuzzy matching
            melhor_match, score, _ = process.extractOne(acao_proc, transacoes.keys())
            if score and score > 75:
                resultado = transacoes[melhor_match]
                st.success(f"👉 Transação SAP: **{resultado}**  \n(interpretado como: *{melhor_match}*)")
            else:
                st.error("❌ Não encontrei nenhuma transação correspondente.")
                st.warning(
                    f"""
                    ➡️ Base utilizada: **{"transacoes_sap.xlsx"}**  
                    🔗 [Abrir planilha no GitHub]({'https://github.com/larissafeitosa24/meu-dicionario-sap/blob/main/'})  

                    Para adicionar uma nova transação:  
                    1. Abra o arquivo no GitHub.  
                    2. Clique em **Edit** (se tiver permissão) ou **Download** para editar localmente.  
                    3. Adicione uma nova linha com:  
                       - **Descrição** (palavras-chave, separadas por vírgula se quiser várias)  
                       - **Código SAP** correspondente  
                    4. Salve/commite a mudança.  
                    5. Recarregue o app.  
                    """
                )


