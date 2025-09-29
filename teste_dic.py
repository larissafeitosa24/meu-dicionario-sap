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

# 🔹 Caso especial: mostrar todas relacionadas a uma palavra
relacionados = {desc: cod for desc, cod in transacoes.items() if acao_proc in desc}
if relacionados:
 st.info(f"📌 Transações relacionadas a '{acao}':")
for d, c in relacionados.items():
 st.write(f"- {d} → **{c}**")

# 🔹 Caso normal: fuzzy matching para achar o mais parecido
else:
 melhor_match, score, _ = process.extractOne(acao_proc, transacoes.keys())
if score > 75:
 resultado = transacoes[melhor_match]
st.success(f"👉 Transação SAP: **{resultado}** \n(interpretado como: *{melhor_match}*)")
else:
st.error("❌ Não encontrei nenhuma transação correspondente. Tente reformular a frase.")
st.warning( 
  f"""
  Base Utilizada  : **{transacoes_sap.xlsx}**
  [Abrir planilha no Github]({})

  Para adicionar uma nova transação :
  1. Abra o arquivo no github
  2. Clique em **Edit** 
  3. Adicione uma nova linha com :
   - **Descrição** (palavras-chave, separado por virgula )
   - **Código SAP** ( digite o código da transação SAP)
  4. Salve a mudança
  5. Recarregue a página  
 """
 )
# Caso erro : nao encontrar nenhuma transação relacionada com a palavra digitada
