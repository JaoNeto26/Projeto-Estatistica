"""
Teste de Hipóteses — Comparação de Duas Amostras
=================================================
Casos cobertos:
  1. Teste Z  — igualdade de duas médias (variâncias conhecidas)
  2. Teste t  — igualdade de duas médias, variâncias iguais   (pooled)
  3. Teste t  — igualdade de duas médias, variâncias diferentes (Welch)
  4. Teste Z  — igualdade de duas proporções

"""

from caso1 import teste_z_duas_medias
from caso2 import teste_t_pooled
from caso3 import teste_t_welch
from caso4 import teste_z_duas_proporcoes
import math
from scipy import stats


# ─────────────────────────────────────────────
#  Utilitários
# ─────────────────────────────────────────────

def separador(titulo: str):
    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def linha(descricao: str, valor):
    print(f"  {descricao:<40} {valor}")


def decisao(estatistica: float, critico: float, bilateral: bool) -> str:
    if bilateral:
        rejeita = abs(estatistica) > abs(critico)
    else:
        rejeita = estatistica > critico  # ajuste se for cauda esquerda
    if rejeita:
        return "REJEITAR H₀  →  evidência de diferença significativa"
    return "NÃO REJEITAR H₀  →  sem evidência suficiente de diferença"


# ─────────────────────────────────────────────
#  DADOS 
# ─────────────────────────────────────────────

if __name__ == "__main__":

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
