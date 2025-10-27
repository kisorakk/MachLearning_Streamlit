import streamlit as st
import pandas as pd
import joblib
import os

# Configuração da página
st.set_page_config(layout="centered", page_title="Previsão de Desempenho Acadêmico")

# Definição das features
FEATURES_NAMES = [
    'ï»¿Nota_P1',
    'Nota_P2',
    'Media_Trabalhos',
    'Frequencia',
    'Reprovacoes_Anteriores',
    'Acessos_Plataforma_Mes'
]

COLUNAS_HISTORICO = FEATURES_NAMES + ["Previsao_Resultado", 'Prob_Aprovacao', 'Prob_Reprovacao']

# Inicialização do histórico
if 'historico_previsoes' not in st.session_state:
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)

@st.cache_resource
def carregar_modelo(caminho_modelo="modelo_desempenho_academico.joblib"):
    try:
        if os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo
        else:
            st.error("O arquivo do modelo não foi encontrado.")
            st.warning("Por favor, execute o script de treinamento para gerar o modelo.")               
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

# Carrega o modelo
pipeline_modelo = carregar_modelo()

# Interface do usuário
st.title("🤖 Sistema de Previsão de Desempenho Acadêmico")
st.markdown("""
    Este aplicativo utiliza um modelo de machine learning treinado para prever o desempenho acadêmico dos alunos com base em suas notas e frequência.   
            
    **Preencha os dados do aluno abaixo para obter uma previsão do seu desempenho acadêmico!**
    """)

# Input do nome do usuário
nome_user = st.text_input("Informe seu nome:", value="digite aqui...")

if st.button("Enviar"):
    if nome_user and nome_user != "digite aqui...":
        st.success(f"Seja bem-vindo(a), {nome_user}!")
    else:
        st.warning("Por favor, insira seu nome.")

# Formulário de previsão
if pipeline_modelo is not None:
    form = st.form(key='form_previsao')
    form.subheader("Insira os dados do aluno para previsão:")

    col1, col2 = form.columns(2)

    with col1:
        nota_p1 = form.slider("Nota P1:", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        media_trabalhos = form.slider("Média dos Trabalhos:", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        reprovacoes_anteriores = form.number_input("Número de Reprovações Anteriores:", min_value=0, max_value=10, value=0, step=1)

    with col2:
        nota_p2 = form.slider("Nota P2:", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        frequencia = form.slider("Frequência (%):", min_value=0.0, max_value=100.0, value=75.0, step=5.0)
        acessos_plataforma_mes = form.number_input("Acessos à Plataforma no Mês:", min_value=0, max_value=100, value=10, step=1)

    submitted = form.form_submit_button("Obter Previsão")

    if submitted:
        dados_aluno = pd.DataFrame(
            [[nota_p1, nota_p2, media_trabalhos, frequencia, reprovacoes_anteriores, acessos_plataforma_mes]],
            columns=FEATURES_NAMES
        )

        st.info("Processando dados e realizando a previsão...")

        try:
            previsao = pipeline_modelo.predict(dados_aluno)
            probabilidade = pipeline_modelo.predict_proba(dados_aluno)

            prob_reprovado = probabilidade[0][0]
            prob_aprovado = probabilidade[0][1]

            st.subheader("Resultado da Previsão")

            if previsao[0] == 1:
                st.success("Previsão: Aprovado!")
                st.markdown(f"""
                    Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                    **{prob_aprovado*100:.2f}%** de chance de ser **aprovado**

                    *Chance de reprovação: {prob_reprovado*100:.2f}%*
                """)
            else:
                st.error("Previsão: Reprovado (zona de risco)")
                st.markdown(f"""
                    Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                    **{prob_reprovado*100:.2f}%** de chance de ser **reprovado**

                    *Chance de aprovação: {prob_aprovado*100:.2f}%*
                """)

            # Atualizar histórico
            novo_registro = dados_aluno.copy()
            novo_registro['Previsao_Resultado'] = previsao[0]
            novo_registro['Prob_Aprovacao'] = prob_aprovado
            novo_registro['Prob_Reprovacao'] = prob_reprovado
            
            st.session_state.historico_previsoes = pd.concat(
                [st.session_state.historico_previsoes, novo_registro],
                ignore_index=True
            )

        except Exception as e:
            st.error(f"Erro ao realizar a previsão: {str(e)}")
else:
    st.warning("O aplicativo não pode fazer previsões porque o modelo não está disponível.")

# Botão para limpar histórico
if st.button("Limpar Histórico de Previsões"):
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)
    st.success("Histórico de previsões limpo com sucesso!")
    st.rerun()

# Exibir histórico se existir
if not st.session_state.historico_previsoes.empty:
    st.subheader("Histórico de Previsões")
    st.dataframe(st.session_state.historico_previsoes)