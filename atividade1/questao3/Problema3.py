import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv("atividade1\\questao3\\dados_satisfacao_tempo.csv")

# Exibe os dados
print("Dataset:")
print(dados)

print("\nInformações do dataset:")
dados.info()

print("\nEstatísticas descritivas:")
print(dados.describe())

# Correlação entre satisfação e tempo
print("\nMatriz de correlação:")
print(dados[["satisfacao", "tempo_tarefa_min"]].corr())

# Covariância entre satisfação e tempo
print("\nMatriz de covariância:")
print(dados[["satisfacao", "tempo_tarefa_min"]].cov())

correlacao = dados["satisfacao"].corr(dados["tempo_tarefa_min"])
covariancia = dados["satisfacao"].cov(dados["tempo_tarefa_min"])

print(f"\nCorrelação entre satisfação e tempo: {correlacao:.2f}")
print(f"Covariância entre satisfação e tempo: {covariancia:.2f}")

# Gráfico de dispersão
plt.scatter(
    dados["satisfacao"],
    dados["tempo_tarefa_min"]
)

plt.xlabel("Satisfação")
plt.ylabel("Tempo da tarefa (min)")
plt.title("Relação entre satisfação e tempo da tarefa")

plt.show()