from util import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 3 — Teste t de Welch para duas médias
#           variâncias diferentes (não pooled)
# ─────────────────────────────────────────────

def teste_t_welch(
    n1: int,   media1: float, s1: float,
    n2: int,   media2: float, s2: float,
    alpha: float = 0.05,
    bilateral: bool = True,
):
    separador("CASO 3 — Teste t de Welch: igualdade de duas médias (variâncias diferentes)")

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

    v1 = s1**2 / n1
    v2 = s2**2 / n2
    t  = (media1 - media2) / math.sqrt(v1 + v2)

    # Graus de liberdade de Welch–Satterthwaite
    gl = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))

    # Decisão
    alpha_ref = alpha / 2 if bilateral else alpha
    # Para teste bilateral, o valor crítico é simétrico (±t), para unilateral é apenas positivo
    t_critico = stats.t.ppf(1 - alpha_ref, df=gl)
    # O p-valor é a probabilidade de obter um resultado tão extremo quanto o observado, sob H₀
    p_valor   = 2 * (1 - stats.t.cdf(abs(t), df=gl)) if bilateral else 1 - stats.t.cdf(t, df=gl)

    print(f"\n  GL Welch–Satterthwaite = {gl:.2f}")
    print(f"  t calculado            = {t:.4f}")
    print(f"  t crítico              = ±{t_critico:.4f}" if bilateral else f"  t crítico = {t_critico:.4f}")
    print(f"  p-valor                = {p_valor:.4f}")
    print(f"\n  → {decisao(t, t_critico, bilateral)}")

