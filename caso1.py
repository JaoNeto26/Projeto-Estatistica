from Testehipoteses import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 1 — Teste Z para duas médias
#           (variâncias populacionais conhecidas)
# ─────────────────────────────────────────────

def teste_z_duas_medias(
    n1: int,   media1: float, sigma1: float,
    n2: int,   media2: float, sigma2: float,
    alpha: float = 0.05,
    bilateral: bool = True,
): 
    separador("CASO 1 — Teste Z: igualdade de duas médias (σ conhecidas)")

    # ── Dados ──
    linha("n₁", n1);  
    linha("x̄₁", media1);  
    linha("σ₁", sigma1)
    linha("n₂", n2);  
    linha("x̄₂", media2);  
    linha("σ₂", sigma2)
    linha("α", alpha); 
    linha("Bilateral?", bilateral)

    # ── Hipóteses ──
    print("\n  H₀: μ₁ = μ₂")
    print("  H₁: μ₁ ≠ μ₂" if bilateral else "  H₁: μ₁ > μ₂")

    # ── Cálculo ──
    erro_padrao = math.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    z = (media1 - media2) / erro_padrao

    alpha_ref = alpha / 2 if bilateral else alpha
    z_critico = stats.norm.ppf(1 - alpha_ref)
    p_valor = 2 * (1 - stats.norm.cdf(abs(z))) if bilateral else 1 - stats.norm.cdf(z)

    print(f"\n  Erro-padrão          = {erro_padrao:.4f}")
    print(f"  Z calculado          = {z:.4f}")
    print(f"  Z crítico            = ±{z_critico:.4f}" if bilateral else f"  Z crítico = {z_critico:.4f}")
    print(f"  p-valor              = {p_valor:.4f}")
    print(f"\n  → {decisao(z, z_critico, bilateral)}")
