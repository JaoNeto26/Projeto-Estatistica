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

