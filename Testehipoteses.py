"""
Teste de Hipóteses — Comparação de Duas Amostras
=================================================
Casos cobertos:
  1. Teste Z  — igualdade de duas médias (variâncias conhecidas)
  2. Teste t  — igualdade de duas médias, variâncias iguais   (pooled)
  3. Teste t  — igualdade de duas médias, variâncias diferentes (Welch)
  4. Teste Z  — igualdade de duas proporções

Como usar:
  - Preencha os dados na seção "DADOS" de cada caso.
  - Execute: python teste_hipoteses.py
  - O resultado imprime o passo a passo completo no terminal.
"""

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
    linha("n₁", n1);  linha("x̄₁", media1);  linha("σ₁", sigma1)
    linha("n₂", n2);  linha("x̄₂", media2);  linha("σ₂", sigma2)
    linha("α", alpha); linha("Bilateral?", bilateral)

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

    linha("n₁", n1);  linha("x̄₁", media1);  linha("s₁", s1)
    linha("n₂", n2);  linha("x̄₂", media2);  linha("s₂", s2)
    linha("α", alpha); linha("Bilateral?", bilateral)

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

    linha("n₁", n1);  linha("x̄₁", media1);  linha("s₁", s1)
    linha("n₂", n2);  linha("x̄₂", media2);  linha("s₂", s2)
    linha("α", alpha); linha("Bilateral?", bilateral)

    print("\n  H₀: μ₁ = μ₂")
    print("  H₁: μ₁ ≠ μ₂" if bilateral else "  H₁: μ₁ > μ₂")

    v1 = s1**2 / n1
    v2 = s2**2 / n2
    t  = (media1 - media2) / math.sqrt(v1 + v2)

    # Graus de liberdade de Welch–Satterthwaite
    gl = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))

    alpha_ref = alpha / 2 if bilateral else alpha
    t_critico = stats.t.ppf(1 - alpha_ref, df=gl)
    p_valor   = 2 * (1 - stats.t.cdf(abs(t), df=gl)) if bilateral else 1 - stats.t.cdf(t, df=gl)

    print(f"\n  GL Welch–Satterthwaite = {gl:.2f}")
    print(f"  t calculado            = {t:.4f}")
    print(f"  t crítico              = ±{t_critico:.4f}" if bilateral else f"  t crítico = {t_critico:.4f}")
    print(f"  p-valor                = {p_valor:.4f}")
    print(f"\n  → {decisao(t, t_critico, bilateral)}")


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

    linha("Grupo 1: sucessos (x₁)", x1);  linha("Grupo 1: total (n₁)", n1);  linha("p̂₁", round(p1, 4))
    linha("Grupo 2: sucessos (x₂)", x2);  linha("Grupo 2: total (n₂)", n2);  linha("p̂₂", round(p2, 4))
    linha("α", alpha); linha("Bilateral?", bilateral)

    print("\n  H₀: p₁ = p₂")
    print("  H₁: p₁ ≠ p₂" if bilateral else "  H₁: p₁ > p₂")

    # Verificação das condições de validade
    condicoes = [n1*p1 >= 5, n1*(1-p1) >= 5, n2*p2 >= 5, n2*(1-p2) >= 5]
    print(f"\n  Condições (n·p̂ ≥ 5): {['OK' if c else 'FALHOU' for c in condicoes]}")

    p_pool = (x1 + x2) / (n1 + n2)
    erro   = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z      = (p1 - p2) / erro

    alpha_ref = alpha / 2 if bilateral else alpha
    z_critico = stats.norm.ppf(1 - alpha_ref)
    p_valor   = 2 * (1 - stats.norm.cdf(abs(z))) if bilateral else 1 - stats.norm.cdf(z)

    print(f"\n  p̂ combinado (pooled)  = {p_pool:.4f}")
    print(f"  Erro-padrão           = {erro:.4f}")
    print(f"  Z calculado           = {z:.4f}")
    print(f"  Z crítico             = ±{z_critico:.4f}" if bilateral else f"  Z crítico = {z_critico:.4f}")
    print(f"  p-valor               = {p_valor:.4f}")
    print(f"\n  → {decisao(z, z_critico, bilateral)}")


# ─────────────────────────────────────────────
#  DADOS — altere aqui para seus exemplos
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