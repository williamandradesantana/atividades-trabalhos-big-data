import pandas as pd

dados = {
    "Visitantes": [5.2, 6.5, 4.8, 7.1, 6.0, 5.5, 8.2, 4.5, 7.5, 6.8],
    "Vendas": [1.8, 2.1, 1.5, 2.5, 2.0, 1.9, 2.8, 1.4, 2.6, 2.2]
}

df = pd.DataFrame(dados)

print("Visitantes únicos:")
print("Média:", df["Visitantes"].mean())
print("Mediana:", df["Visitantes"].median())
print("1º Quartil (Q1):", df["Visitantes"].quantile(0.25))
print("2º Quartil (Q2):", df["Visitantes"].quantile(0.50))
print("3º Quartil (Q3):", df["Visitantes"].quantile(0.75))
print("Desvio padrão:", df["Visitantes"].std())

print("\nVendas")
print("Média:", df["Vendas"].mean())
print("Mediana:", df["Vendas"].median())
print("1º Quartil (Q1):", df["Vendas"].quantile(0.25))
print("2º Quartil (Q2):", df["Vendas"].quantile(0.50))
print("3º Quartil (Q3):", df["Vendas"].quantile(0.75))
print("Desvio padrão:", df["Vendas"].std())

print(df.describe())

print("\nCovariância:")
print(df["Visitantes"].cov(df["Vendas"]))

print("\nCorrelação:")
print(df["Visitantes"].corr(df["Vendas"]))