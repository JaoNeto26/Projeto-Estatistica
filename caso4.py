from util import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 4 — Teste Z para duas proporções
# ─────────────────────────────────────────────

def teste_z_duas_proporcoes(
    x1: int, n1: int,
    x2: int, n2: int,
    alpha: float = 0.05,
    bilateral: bool = True,
):
    separador("CASO 4 — Teste Z: igualdade de duas proporções")

    p1 = x1 / n1
    p2 = x2 / n2

    linha("Grupo 1: sucessos (x₁)", x1);  
    linha("Grupo 1: total (n₁)", n1);  
    linha("p̂₁", round(p1, 4))
    linha("Grupo 2: sucessos (x₂)", x2);  
    linha("Grupo 2: total (n₂)", n2);  
    linha("p̂₂", round(p2, 4))
    linha("α", alpha); 
    linha("Bilateral?", bilateral)

    print("\n  H₀: p₁ = p₂")
    print("  H₁: p₁ ≠ p₂" if bilateral else "  H₁: p₁ > p₂")

    # Verificação das condições de validade
    condicoes = [n1*p1 >= 5, n1*(1-p1) >= 5, n2*p2 >= 5, n2*(1-p2) >= 5]
    print(f"\n  Condições (n·p̂ ≥ 5): {['OK' if c else 'FALHOU' for c in condicoes]}")

    # Cálculo do Z usando proporção combinada (pooled)
    p_pool = (x1 + x2) / (n1 + n2)
    erro   = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z      = (p1 - p2) / erro

    # Decisão
    alpha_ref = alpha / 2 if bilateral else alpha
    # Para teste bilateral, o valor crítico é simétrico (±z), para unilateral é apenas positivo
    z_critico = stats.norm.ppf(1 - alpha_ref)
    # O p-valor é a probabilidade de obter um resultado tão extremo quanto o observado, sob H₀
    p_valor   = 2 * (1 - stats.norm.cdf(abs(z))) if bilateral else 1 - stats.norm.cdf(z)

    print(f"\n  p̂ combinado (pooled)  = {p_pool:.4f}")
    print(f"  Erro-padrão           = {erro:.4f}")
    print(f"  Z calculado           = {z:.4f}")
    print(f"  Z crítico             = ±{z_critico:.4f}" if bilateral else f"  Z crítico = {z_critico:.4f}")
    print(f"  p-valor               = {p_valor:.4f}")
    print(f"\n  → {decisao(z, z_critico, bilateral)}")
