import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics


def carregar_dados(caminho_arquivo = "historico_Academico.csv"):

    try:
        if os.path.exists(caminho_arquivo):
            
            df = pd.read_csv(caminho_arquivo, encoding="latin1", sep=",")

            print("Dados carregados com sucesso.")

            return df
        
        else:
            print("O arquivo especificado não foi encontrado.")

    except Exception as e:
        print(f"Ocorreu um erro ao carregar os dados: {e}")

        return None

dados = carregar_dados()
print(dados)