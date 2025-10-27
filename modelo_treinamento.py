import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):

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


if dados is not None:
    print(dados.head())
    print("iniciando o pipeline de treinamento...")

    TARGET_COLUMN = "Status_Final"

    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]
        print("Features (X) definidas com sucesso.")
        print("Target (y) definida com sucesso.")

    except KeyError:

        print("erro critico")
        print(f"A coluna alvo '{TARGET_COLUMN}' não foi encontrada no conjunto de dados.")
        print(f"Colunas disponiveis para treinamento: {list(dados.columns)}")
        print(f"por favor, ajuste a variavel 'TARGET_COLUMN' e tente novamente.")
        exit()



        # Divisao entre treino e teste
    print("Dividindo os dados em treino e teste...")


    X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, #20% dos dados para teste
            random_state=42, #garantir reprodutibilidade
            stratify=y  #manter a proporcao das classes
    )

    print(f"Dados de treino: {len(X_train)} | Dados de teste: {len(X_test)}")

    print("Criano a pipeline de Ml")
    pipeline_modelo = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    print("Treinando o modelo...")

    pipeline_modelo.fit(X_train, y_train)

    print("Modelo treinado com sucesso.")
    y_pred = pipeline_modelo.predict(X_test)


    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)
    report = metrics.classification_report(y_test, y_pred)

    print("Relatorio de avaliação geral")
    print(f"Acuracia Geral: {accuracy * 100:.2f}%")
    print("Relatorio de classificação detalhado: ")
    print(report)

    #salvando modelo

    model_filename = 'modelo_desempenho_academico.joblib'

    print(f"Salvando o modelo treinado em.. {model_filename}")
    joblib.dump(pipeline_modelo, model_filename)

    print("Processo concluido com sucesso.")

else:
    print("O pipeline de treinamento nao pode ser executado devido a falha no carregamento dos dados.")