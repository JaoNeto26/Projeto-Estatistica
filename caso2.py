from Testehipoteses import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 2 — Teste t para duas médias
#           variâncias iguais (pooled)
# ─────────────────────────────────────────────

def teste_t_pooled(
    n1: int,   media1: float, s1: float,
    n2: int,   media2: float, s2: float,
    alpha: float = 0.05,
    bilateral: bool = True,
):
    separador("CASO 2 — Teste t: igualdade de duas médias (variâncias iguais — pooled)")

    linha("n₁", n1);  
    linha("x̄₁", media1);  
    linha("s₁", s1)
    linha("n₂", n2);  
    linha("x̄₂", media2);  
    linha("s₂", s2)
    linha("α", alpha); 
    linha("Bilateral?", bilateral)

    print("\n  H₀: μ₁ = μ₂")
    print("  H₁: μ₁ ≠ μ₂" if bilateral else "  H₁: μ₁ > μ₂")

    gl = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / gl
    sp  = math.sqrt(sp2)
    t   = (media1 - media2) / (sp * math.sqrt(1/n1 + 1/n2))

    alpha_ref = alpha / 2 if bilateral else alpha
    t_critico = stats.t.ppf(1 - alpha_ref, df=gl)
    p_valor   = 2 * (1 - stats.t.cdf(abs(t), df=gl)) if bilateral else 1 - stats.t.cdf(t, df=gl)

    print(f"\n  Graus de liberdade   = {gl}")
    print(f"  Sp² (pooled)         = {sp2:.4f}")
    print(f"  Sp                   = {sp:.4f}")
    print(f"  t calculado          = {t:.4f}")
    print(f"  t crítico            = ±{t_critico:.4f}" if bilateral else f"  t crítico = {t_critico:.4f}")
    print(f"  p-valor              = {p_valor:.4f}")
    print(f"\n  → {decisao(t, t_critico, bilateral)}")

