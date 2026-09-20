import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import norm

def print_awenser_text(question_number: int, text: list[str]):
    
    print(f"QUESTÃO {question_number}.")
    
    for t in text:
        print(t)
    
    print("="*50)
    

# Dados
tempos = [
    12, 15, 14, 16, 18, 12, 20, 22, 14, 15,
    16, 18, 19, 21, 17, 13, 14, 15, 16, 18,
    19, 20, 12, 15, 14, 13, 16, 18, 19, 21,
    22, 23, 24, 25, 20, 19, 18, 15, 14, 16,
    17, 15, 14, 12, 13, 18, 19, 20, 21, 22,
]

dados = pd.DataFrame({"Tempos_execucao": tempos})

# Salva o CSV na mesma pasta deste arquivo, independentemente do terminal.
arquivo_csv = Path(__file__).resolve().parent / "dataset.csv"
dados.to_csv(arquivo_csv, index=False)


# 1. Calcule a média amostral dos tempos de execução.
soma = dados.sum()
total_de_valores = dados.count()
media_amostral = soma / total_de_valores

media = sum(tempos) / len(tempos)

print_awenser_text(1, [f"Média: {media:.2f} ms"])

# 2. Calcule o desvio padrão amostral.

desvio_padrao = statistics.stdev(tempos)

print_awenser_text(2, [f"Desvio padrão amostral: {desvio_padrao:.2f} ms"])

# 3. Assumindo distribuição normal, determine a probabilidade de X > 20 ms.

z = (20 - media) / desvio_padrao
probabilidade_maior_20 = 1 - norm.cdf(z)

print_awenser_text(3, [
    f"Valor de Z: {z:.2f}",
    f"Probabilidade de X > 20 ms: {probabilidade_maior_20:.4f}",
    f"Probabilidade em porcentagem: {probabilidade_maior_20 * 100:.2f}%"
])

# 4. Determine a probabilidade de uma execucao apresentar tempo entre 15 ms e 20 ms.

desvio = dados["Tempos_execucao"].std()

# valores de Z
z15 = (15 - media) / desvio
z20 = (20 - media) / desvio

# probabilidade entre 15 e 20 ms
probabilidade_4 = norm.cdf(z20) - norm.cdf(z15)

print_awenser_text(4, [
    f"Z para 15 ms: {z15}",
    f"Z para 20 ms: {z20}",
    f"Probabilidade: {probabilidade_4}",
    f"Porcentagem: {probabilidade_4 * 100}%"
])

# 5. Calcule o percentil 90 da distribuicao obtida.

percentil_90 = norm.ppf(
    0.90,
    loc=media,
    scale=desvio
)


print_awenser_text(5, [
    f"Percentil 90: {percentil_90} ms"
])

# 6. Construa um intervalo de confianca de 95% para a media do tempo de execucao utilizando a aproximacao normal.

n = len(tempos)
z = 1.96
erro_padrao = desvio / np.sqrt(n)
margem_erro = z * erro_padrao

limite_inferior = media - margem_erro
limite_superior = media + margem_erro


print_awenser_text(6, [
    f"Média: {media}",
    f"Erro padrão: {erro_padrao}",
    f"Margem de erro: {margem_erro}",
    f"Limite inferior: {limite_inferior}",
    f"Limite superior: {limite_superior}"
])

# 7. Represente graficamente o histograma dos dados e a curva normal ajustada.

plt.hist(
    tempos, 
    bins=10, 
    density=True, 
    alpha=0.6, 
    color='#4C72B0', 
    edgecolor='black', 
    label='Histograma (Amostra)'
)

xmin, xmax = plt.xlim()
eixo_x = np.linspace(xmin, xmax, 100)
eixo_y = norm.pdf(eixo_x, media, desvio)

plt.plot(
    eixo_x, 
    eixo_y, 
    color='#C44E52', 
    linewidth=2.5, 
    label=f'Curva Normal (μ={media:.2f}, σ={desvio:.2f})'
)


plt.title('Histograma dos Tempos de Execução vs. Curva Normal Ajustada', fontsize=14)
plt.xlabel('Tempo de processamento (ms)', fontsize=12)
plt.ylabel('Densidade de probabilidade', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# 8. Interprete os resultados obtidos, discutindo se o algoritmo apresenta desempenho consistente e
# previsivel.

print_awenser_text(8, [
    "Com base na análise visual do histograma sobreposto à curva normal teórica, observa-se que o desempenho do algoritmo apresenta uma variabilidade que compromete parcialmente a sua previsibilidade em execuções individuais. Embora o modelo gaussiano ajustado utilize a média de 17,18 ms e o desvio padrão de 3,39 ms para descrever a tendência central, o gráfico revela que a amostra real não segue um formato de sino perfeito e perfeitamente simétrico. A distribuição dos tempos de processamento é irregular e algo dispersa, formando blocos de concentração em diferentes faixas, havendo picos de densidade perceptíveis tanto no início (próximo aos 12 ms e 13 ms) quanto em valores mais altos (na faixa dos 20 ms aos 21 ms). Além dessa flutuação na região central, o gráfico demonstra uma extensão dos dados à direita, com execuções que chegam a atingir até 25 ms, evidenciando atrasos esporádicos. Conclui-se, portanto, que o algoritmo não apresenta um desempenho estritamente consistente, pois oscila de forma perceptível entre execuções rápidas e mais demoradas, tornando-se menos adequado para sistemas críticos que exigem tempos de resposta altamente precisos e sem flutuações."
])

