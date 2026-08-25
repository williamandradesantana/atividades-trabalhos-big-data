
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dados fornecidos na tabela do enunciado
acessos = [100, 150, 80, 120, 200, 90, 130, 180, 110, 160]
erros   = [1.5, 2.0, 1.2, 1.8, 2.5, 1.3, 1.9, 2.2, 1.7, 2.1]

df = pd.DataFrame({
    "Acessos_por_Minuto": acessos,
    "Taxa_de_Erros_%": erros
})

print("DADOS")
print(df.to_string(index=False))
print(f"\nNúmero de observações (n) = {len(df)}")

# 1. Estatísticas descritivas: média, mediana, quartis, desvio padrão
def estatisticas(serie, nome):
    media = serie.mean()
    mediana = serie.median()
    q1 = serie.quantile(0.25)
    q2 = serie.quantile(0.50)
    q3 = serie.quantile(0.75)
    # desvio padrão amostral (ddof=1)
    desvio_padrao = serie.std(ddof=1)
    variancia = serie.var(ddof=1)

    print(f"\n--- {nome} ---")
    print(f"Média            : {media:.4f}")
    print(f"Mediana          : {mediana:.4f}")
    print(f"Q1 (25%)         : {q1:.4f}")
    print(f"Q2 (50%)         : {q2:.4f}")
    print(f"Q3 (75%)         : {q3:.4f}")
    print(f"Desvio Padrão (s): {desvio_padrao:.4f}")
    print(f"Variância (s²)   : {variancia:.4f}")

    return {
        "media": media, "mediana": mediana,
        "q1": q1, "q2": q2, "q3": q3,
        "desvio_padrao": desvio_padrao, "variancia": variancia
    }

print("\n1. ESTATÍSTICAS DESCRITIVAS")
stats_acessos = estatisticas(df["Acessos_por_Minuto"], "Acessos por Minuto")
stats_erros   = estatisticas(df["Taxa_de_Erros_%"], "Taxa de Erros (%)")

# 2. Covariância e Coeficiente de Correlação (r)
print("\n2. COVARIÂNCIA E CORRELAÇÃO")

# Covariância amostral (ddof=1)
covariancia = df["Acessos_por_Minuto"].cov(df["Taxa_de_Erros_%"])
print(f"Covariância (Sxy)         : {covariancia:.4f}")

# Coeficiente de correlação de Pearson (r)
correlacao = df["Acessos_por_Minuto"].corr(df["Taxa_de_Erros_%"])
print(f"Coeficiente de Correlação (r): {correlacao:.4f}")

# Verificação manual da covariância e correlação (passo a passo)
x = df["Acessos_por_Minuto"].values
y = df["Taxa_de_Erros_%"].values
n = len(x)
x_mean = x.mean()
y_mean = y.mean()

cov_manual = np.sum((x - x_mean) * (y - y_mean)) / (n - 1)
sx = np.sqrt(np.sum((x - x_mean)**2) / (n - 1))
sy = np.sqrt(np.sum((y - y_mean)**2) / (n - 1))
r_manual = cov_manual / (sx * sy)

print(f"\n(Verificação manual)")
print(f"Covariância manual        : {cov_manual:.4f}")
print(f"r manual                  : {r_manual:.4f}")

# Resumo final em tabela
print("\nRESUMO FINAL")
resumo = pd.DataFrame({
    "Acessos por Minuto": stats_acessos,
    "Taxa de Erros (%)": stats_erros
}).round(4)
print(resumo)
print(f"\nCovariância = {covariancia:.4f}")
print(f"Correlação (r) = {correlacao:.4f}")

# 4. Gráficos
print("\n4. GERANDO GRÁFICOS")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Problema 6: Acessos por Minuto x Taxa de Erros (%)", fontsize=14, fontweight="bold")

# --- Gráfico 1: Dispersão (Scatter) com linha de regressão ---
ax1 = axs[0, 0]
ax1.scatter(x, y, color="royalblue", edgecolor="black", s=70, label="Dados")

# Linha de regressão (mínimos quadrados)
coef = np.polyfit(x, y, 1)  # [inclinação, intercepto]
x_linha = np.linspace(x.min(), x.max(), 100)
y_linha = coef[0] * x_linha + coef[1]
ax1.plot(x_linha, y_linha, color="crimson", linewidth=2,
          label=f"Regressão: y = {coef[0]:.4f}x + {coef[1]:.4f}")

ax1.set_title(f"Dispersão (r = {correlacao:.4f})")
ax1.set_xlabel("Acessos por Minuto")
ax1.set_ylabel("Taxa de Erros (%)")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

# --- Gráfico 2: Boxplot comparativo (dados padronizados p/ visualizar juntos) ---
ax2 = axs[0, 1]
x_padronizado = (x - x.mean()) / x.std(ddof=1)
y_padronizado = (y - y.mean()) / y.std(ddof=1)
try:
    ax2.boxplot([x_padronizado, y_padronizado],
                tick_labels=["Acessos\n(padronizado)", "Taxa de Erros\n(padronizado)"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="crimson", linewidth=2))
except TypeError:
    ax2.boxplot([x_padronizado, y_padronizado],
                labels=["Acessos\n(padronizado)", "Taxa de Erros\n(padronizado)"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="crimson", linewidth=2))
ax2.set_title("Boxplot (valores padronizados)")
ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

# --- Gráfico 3: Histograma - Acessos por Minuto ---
ax3 = axs[1, 0]
ax3.hist(x, bins=6, color="seagreen", edgecolor="black", alpha=0.8)
ax3.axvline(x.mean(), color="crimson", linestyle="--", linewidth=2, label=f"Média = {x.mean():.1f}")
ax3.set_title("Histograma - Acessos por Minuto")
ax3.set_xlabel("Acessos por Minuto")
ax3.set_ylabel("Frequência")
ax3.legend()
ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

# --- Gráfico 4: Histograma - Taxa de Erros ---
ax4 = axs[1, 1]
ax4.hist(y, bins=6, color="darkorange", edgecolor="black", alpha=0.8)
ax4.axvline(y.mean(), color="crimson", linestyle="--", linewidth=2, label=f"Média = {y.mean():.2f}")
ax4.set_title("Histograma - Taxa de Erros (%)")
ax4.set_xlabel("Taxa de Erros (%)")
ax4.set_ylabel("Frequência")
ax4.legend()
ax4.grid(True, linestyle="--", alpha=0.5, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Salva a imagem
plt.savefig("graficos_questao6.png", dpi=150)
print("Gráficos salvos em: graficos_questao6.png")

# Mostra na tela (quando rodar localmente no VS Code)
plt.show()

# 5. Classificação da correlação
if correlacao > 0.7:
    forca = "forte"
elif correlacao > 0.4:
    forca = "moderada"
else:
    forca = "fraca"

sentido = "positiva" if correlacao > 0 else "negativa"

print(f"\nClassificação da correlação: {sentido} e {forca} (r = {correlacao:.4f})")
