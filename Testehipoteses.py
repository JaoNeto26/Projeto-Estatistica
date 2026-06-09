"""
Teste de Hipóteses — Comparação de Duas Amostras
=================================================
Casos cobertos:
  1. Teste Z  — igualdade de duas médias (variâncias conhecidas)
  2. Teste t  — igualdade de duas médias, variâncias iguais   (pooled)
  3. Teste t  — igualdade de duas médias, variâncias diferentes (Welch)
  4. Teste Z  — igualdade de duas proporções

"""
from Casos.caso1 import teste_z_duas_medias
from Casos.caso2 import teste_t_pooled
from Casos.caso3 import teste_t_welch
from Casos.caso4 import teste_z_duas_proporcoes
from graficos.grafico_caso1 import plotar_caso1
from graficos.grafico_caso2 import plotar_caso2
from graficos.grafico_caso3 import plotar_caso3
from graficos.grafico_caso4 import plotar_caso4
import math
from scipy import stats


# ─────────────────────────────────────────────
#  DADOS 
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTeste de Hipóteses: Comparação de Duas Amostras\n")

    # ── Caso 1: Z para médias (σ conhecidas) ──────────────────────
    teste_z_duas_medias(
        n1=30,  media1=72, sigma1=8,
        n2=35,  media2=68, sigma2=10,
        alpha=0.05,
        bilateral=True,
    )

    # ── Caso 2: t pooled (variâncias iguais) ──────────────────────
    teste_t_pooled(
        n1=30,  media1=72, s1=8,
        n2=35,  media2=68, s2=10,
        alpha=0.05,
        bilateral=True,
    )

    # ── Caso 3: t de Welch (variâncias diferentes) ────────────────
    teste_t_welch(
        n1=30,  media1=72, s1=8,
        n2=35,  media2=68, s2=10,
        alpha=0.05,
        bilateral=True,
    )

    # ── Caso 4: Z para proporções ─────────────────────────────────
    teste_z_duas_proporcoes(
        x1=40, n1=60,   # Turma A: 40 aprovados em 60
        x2=50, n2=80,   # Turma B: 50 aprovados em 80
        alpha=0.05,
        bilateral=True,
    )

    print("\nOs casos e dados de agora foram pra mostrar a estrutura geral, mas os dados são os mesmos dos casos anteriores. Em um cenário real, cada teste teria seus próprios dados específicos.")
    print("\nagora você pode modificar os dados de cada teste para ver como as decisões mudam, ou adicionar novos casos seguindo a mesma estrutura!")


    while (True):
        print("1 - Teste Z para duas médias")
        print("2 - Teste t pooled")
        print("3 - Teste t Welch")
        print("4 - Teste Z para proporções")
        print("0 - Sair")

        opcao = int(input("Escolha o teste: "))
        if opcao == 1:
            # ── Caso 1: Z para médias (σ conhecidas) ──────────────────────
            print("\nCASO 1 - Teste Z para duas médias")

            n1 = int(input("Digite n1: "))
            media1 = float(input("Digite a média 1: "))
            sigma1 = float(input("Digite o sigma 1: "))

            n2 = int(input("Digite n2: "))
            media2 = float(input("Digite a média 2: "))
            sigma2 = float(input("Digite o sigma 2: "))

            alpha = float(input("Digite o nível de significância (ex: 0.05): "))
            bilateral = input("Teste bilateral? (s/n): ").lower() == "s"
            

            if (not bilateral) and (alpha >= 0.5):
                print("Aviso: para teste unilateral, o nível de significância deve ser menor que 0.5. Ajustando para 0.05.")
                alpha = 0.05
            if (bilateral) and (alpha >= 1):
                print("Aviso: para teste bilateral, o nível de significância deve ser menor que 1. Ajustando para 0.05.")
                alpha = 0.05

            teste_z_duas_medias(
                n1=n1,
                media1=media1,
                sigma1=sigma1,
                n2=n2,
                media2=media2,
                sigma2=sigma2,
                alpha=alpha,
                bilateral=bilateral,
            )
            plotar_caso1(
                n1=n1,  media1=media1, sigma1=sigma1,
                n2=n2,  media2=media2, sigma2=sigma2,
                alpha=alpha,
                bilateral=bilateral,
                salvar=False,           # True para salvar como imagem
                nome_arquivo="caso1_z_medias.png",
            )
        elif opcao == 2:
            # ── Caso 2: t pooled (variâncias iguais) ──────────────────────
            print("\nCASO 2 - Teste t pooled")

            n1 = int(input("Digite n1: "))
            media1 = float(input("Digite a média 1: "))
            s1 = float(input("Digite o desvio padrão amostral s1: "))

            n2 = int(input("Digite n2: "))
            media2 = float(input("Digite a média 2: "))
            s2 = float(input("Digite o desvio padrão amostral s2: "))

            alpha = float(input("Digite o nível de significância: "))
            bilateral = input("Teste bilateral? (s/n): ").lower() == "s"

            if (not bilateral) and (alpha >= 0.5):
                print("Aviso: para teste unilateral, o nível de significância deve ser menor que 0.5. Ajustando para 0.05.")
                alpha = 0.05
            if (bilateral) and (alpha >= 1):
                print("Aviso: para teste bilateral, o nível de significância deve ser menor que 1. Ajustando para 0.05.")
                alpha = 0.05

            teste_t_pooled(
                n1=n1,
                media1=media1,
                s1=s1,
                n2=n2,
                media2=media2,
                s2=s2,
                alpha=alpha,
                bilateral=bilateral,
            )
            plotar_caso2(
                n1=n1,  media1=media1, s1=s1,
                n2=n2,  media2=media2, s2=s2,
                alpha=alpha,
                bilateral=bilateral,
                salvar=False,
                nome_arquivo="caso2_t_pooled.png",
            )
        elif opcao == 3:
            # ── Caso 3: t de Welch (variâncias diferentes) ────────────────
            print("\nCASO 3 - Teste t de Welch")

            n1 = int(input("Digite n1: "))
            media1 = float(input("Digite a média 1: "))
            s1 = float(input("Digite o desvio padrão amostral s1: "))

            n2 = int(input("Digite n2: "))
            media2 = float(input("Digite a média 2: "))
            s2 = float(input("Digite o desvio padrão amostral s2: "))

            alpha = float(input("Digite o nível de significância: "))
            bilateral = input("Teste bilateral? (s/n): ").lower() == "s"

            if (not bilateral) and (alpha >= 0.5):
                print("Aviso: para teste unilateral, o nível de significância deve ser menor que 0.5. Ajustando para 0.05.")
                alpha = 0.05
            if (bilateral) and (alpha >= 1):
                print("Aviso: para teste bilateral, o nível de significância deve ser menor que 1. Ajustando para 0.05.")
                alpha = 0.05

            teste_t_welch(
                n1=n1,
                media1=media1,
                s1=s1,
                n2=n2,
                media2=media2,
                s2=s2,
                alpha=alpha,
                bilateral=bilateral,
            )
            plotar_caso3(
                n1=n1,  media1=media1, s1=s1,
                n2=n2,  media2=media2, s2=s2,
                alpha=alpha,
                bilateral=bilateral,
                salvar=False,
                nome_arquivo="caso3_t_welch.png",
            )
        elif opcao == 4:
            # ── Caso 4: Z para proporções ─────────────────────────────────
            print("\nCASO 4 - Teste Z para duas proporções")

            x1 = int(input("Digite x1 (sucessos da amostra 1): "))
            n1 = int(input("Digite n1 (tamanho da amostra 1): "))

            x2 = int(input("Digite x2 (sucessos da amostra 2): "))
            n2 = int(input("Digite n2 (tamanho da amostra 2): "))

            alpha = float(input("Digite o nível de significância: "))
            bilateral = input("Teste bilateral? (s/n): ").lower() == "s"

            if (not bilateral) and (alpha >= 0.5):
                print("Aviso: para teste unilateral, o nível de significância deve ser menor que 0.5. Ajustando para 0.05.")
                alpha = 0.05
            if (bilateral) and (alpha >= 1):
                print("Aviso: para teste bilateral, o nível de significância deve ser menor que 1. Ajustando para 0.05.")
                alpha = 0.05

            teste_z_duas_proporcoes(
                x1=x1,
                n1=n1,
                x2=x2,
                n2=n2,
                alpha=alpha,
                bilateral=bilateral,
            )
            plotar_caso4(
                x1=x1,
                n1=n1,
                x2=x2,
                n2=n2,
                alpha=alpha,
                bilateral=bilateral,
                salvar=False,
                nome_arquivo="caso4_z_proporcoes.png",
            )
        elif opcao == 0:
            break
        else:
            print("Opção inválida. Tente novamente.")