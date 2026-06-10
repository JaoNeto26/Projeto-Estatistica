"""
Exercícios Resolvidos — Testes de Hipóteses
============================================
Cobre TODOS os exercícios das imagens do livro:
  Q1  – Q5   Testes para a Média Populacional
  Q9  – Q10  Testes para a Proporção
  Q13 – Q16  Testes para Igualdade de Duas Variâncias
  Q17 – Q20  Testes para Igualdade de Duas Médias
  Q21 – Q23  Testes para Igualdade de Duas Proporções

Os exercícios que envolvem UMA amostra (média/proporção) têm lógica
própria aqui.  Os exercícios de DUAS amostras reutilizam as classes
de Casos/ sem alterá-las.

Execução standalone:  python Exercicios.py
"""

import math, sys, tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

# ── imports do projeto ────────────────────────────────────────────
try:
    from Casos.caso1 import TesteZDuasMedias
    from Casos.caso2 import TesteTPooled
    from Casos.caso3 import TesteTWelch
    from Casos.caso4 import TesteZDuasProporcoes
except ModuleNotFoundError:
    sys.path.insert(0, ".")
    from Casos.caso1 import TesteZDuasMedias
    from Casos.caso2 import TesteTPooled
    from Casos.caso3 import TesteTWelch
    from Casos.caso4 import TesteZDuasProporcoes

# ══════════════════════════════════════════════════════════════════
#  PALETA — idêntica ao Interface.py
# ══════════════════════════════════════════════════════════════════
COR_FUNDO     = "#1e1e2e"
COR_PAINEL    = "#2a2a3e"
COR_CARD      = "#313145"
COR_BORDA     = "#44445a"
COR_TEXTO     = "#cdd6f4"
COR_TEXTO_DIM = "#7f849c"
COR_DESTAQUE  = "#89b4fa"
COR_VERDE     = "#a6e3a1"
COR_VERMELHO  = "#f38ba8"
COR_AMARELO   = "#f9e2af"
COR_ROXO      = "#cba6f7"
COR_LARANJA   = "#fab387"

# ══════════════════════════════════════════════════════════════════
#  MOTORES INTERNOS — UMA amostra (não existem nos Casos/)
# ══════════════════════════════════════════════════════════════════

def _stats(dados):
    """Média e desvio-padrão amostral."""
    n  = len(dados)
    xm = sum(dados) / n
    s  = math.sqrt(sum((x - xm)**2 for x in dados) / (n - 1))
    return n, xm, s

def _teste_z_media(n, xbar, sigma, mu0, alpha, lado):
    """Z para uma média (sigma conhecido). lado: 'bilateral','direita','esquerda'."""
    z      = (xbar - mu0) / (sigma / math.sqrt(n))
    ar     = alpha/2 if lado == "bilateral" else alpha
    z_crit = stats.norm.ppf(1 - ar)
    if lado == "bilateral":
        p    = 2*(1 - stats.norm.cdf(abs(z)))
        rej  = abs(z) > z_crit
        h1   = f"μ ≠ {mu0}"
        reg  = f"|Z| > {z_crit:.4f}"
    elif lado == "direita":
        p    = 1 - stats.norm.cdf(z)
        rej  = z > z_crit
        h1   = f"μ > {mu0}"
        reg  = f"Z > {z_crit:.4f}"
    else:
        p    = stats.norm.cdf(z)
        rej  = z < -z_crit
        h1   = f"μ < {mu0}"
        reg  = f"Z < -{z_crit:.4f}"
    return dict(
        tipo="Z", estatistica=z, critico=z_crit, p_valor=p,
        rejeita=rej, bilateral=(lado=="bilateral"), alpha=alpha,
        h0=f"μ = {mu0}", h1=h1,
        extras={"Erro-padrão": f"{sigma/math.sqrt(n):.4f}",
                "Região crítica": reg},
        grupos={"labels": ["Amostra", "H₀ (μ₀)"],
                "centros": [xbar, mu0],
                "dispersoes": [sigma, sigma],
                "modo": "medias"},
        lado=lado, mu0=mu0,
    )

def _teste_t_media(n, xbar, s, mu0, alpha, lado):
    """t para uma média (sigma desconhecido)."""
    gl     = n - 1
    t      = (xbar - mu0) / (s / math.sqrt(n))
    ar     = alpha/2 if lado == "bilateral" else alpha
    t_crit = stats.t.ppf(1 - ar, df=gl)
    if lado == "bilateral":
        p    = 2*(1 - stats.t.cdf(abs(t), df=gl))
        rej  = abs(t) > t_crit
        h1   = f"μ ≠ {mu0}"
        reg  = f"|t| > {t_crit:.4f}"
    elif lado == "direita":
        p    = 1 - stats.t.cdf(t, df=gl)
        rej  = t > t_crit
        h1   = f"μ > {mu0}"
        reg  = f"t > {t_crit:.4f}"
    else:
        p    = stats.t.cdf(t, df=gl)
        rej  = t < -t_crit
        h1   = f"μ < {mu0}"
        reg  = f"t < -{t_crit:.4f}"
    return dict(
        tipo="t", estatistica=t, critico=t_crit, p_valor=p,
        rejeita=rej, bilateral=(lado=="bilateral"), alpha=alpha, gl=gl,
        h0=f"μ = {mu0}", h1=h1,
        extras={"GL": gl, "Erro-padrão (s/√n)": f"{s/math.sqrt(n):.4f}",
                "Região crítica": reg},
        grupos={"labels": ["Amostra", "H₀ (μ₀)"],
                "centros": [xbar, mu0],
                "dispersoes": [s, s],
                "modo": "medias"},
        lado=lado, mu0=mu0,
    )

def _teste_z_proporcao(x, n, p0, alpha, lado):
    """Z para uma proporção."""
    p_hat  = x / n
    z      = (p_hat - p0) / math.sqrt(p0*(1-p0)/n)
    ar     = alpha/2 if lado == "bilateral" else alpha
    z_crit = stats.norm.ppf(1 - ar)
    if lado == "bilateral":
        p    = 2*(1 - stats.norm.cdf(abs(z)))
        rej  = abs(z) > z_crit
        h1   = f"p ≠ {p0}"
        reg  = f"|Z| > {z_crit:.4f}"
    elif lado == "direita":
        p    = 1 - stats.norm.cdf(z)
        rej  = z > z_crit
        h1   = f"p > {p0}"
        reg  = f"Z > {z_crit:.4f}"
    else:
        p    = stats.norm.cdf(z)
        rej  = z < -z_crit
        h1   = f"p < {p0}"
        reg  = f"Z < -{z_crit:.4f}"
    ic    = 1.96*math.sqrt(p_hat*(1-p_hat)/n)
    return dict(
        tipo="Z", estatistica=z, critico=z_crit, p_valor=p,
        rejeita=rej, bilateral=(lado=="bilateral"), alpha=alpha,
        h0=f"p = {p0}", h1=h1,
        extras={"p̂": f"{p_hat:.4f}", "p₀": f"{p0:.4f}",
                "Erro-padrão": f"{math.sqrt(p0*(1-p0)/n):.4f}",
                "Região crítica": reg},
        grupos={"labels": ["p̂ (amostra)", "p₀ (H₀)"],
                "centros": [p_hat, p0],
                "dispersoes": [ic, 0.01],
                "modo": "proporcoes_uma"},
        lado=lado, p0=p0,
    )

def _teste_f_variancias(n1, s1, n2, s2, alpha, bilateral):
    """F para igualdade de duas variâncias."""
    F      = s1**2 / s2**2
    gl1    = n1 - 1
    gl2    = n2 - 1
    ar     = alpha/2 if bilateral else alpha
    f_sup  = stats.f.ppf(1 - ar, gl1, gl2)
    f_inf  = stats.f.ppf(ar,     gl1, gl2)
    p_val  = 2*min(stats.f.cdf(F, gl1, gl2), 1-stats.f.cdf(F, gl1, gl2)) if bilateral \
             else 1 - stats.f.cdf(F, gl1, gl2)
    rej    = (F > f_sup or F < f_inf) if bilateral else F > f_sup
    return dict(
        tipo="F", estatistica=F, critico=f_sup, critico_inf=f_inf,
        p_valor=p_val, rejeita=rej, bilateral=bilateral, alpha=alpha,
        gl1=gl1, gl2=gl2,
        h0="σ₁² = σ₂²", h1="σ₁² ≠ σ₂²" if bilateral else "σ₁² > σ₂²",
        extras={"GL₁ (num.)": gl1, "GL₂ (den.)": gl2,
                "s₁²": f"{s1**2:.4f}", "s₂²": f"{s2**2:.4f}",
                "F crítico sup.": f"{f_sup:.4f}",
                "F crítico inf.": f"{f_inf:.4f}"},
        grupos={"labels": ["Grupo 1", "Grupo 2"],
                "centros": [s1**2, s2**2],
                "dispersoes": [s1**2*0.1, s2**2*0.1],
                "modo": "variancias"},
    )

# ══════════════════════════════════════════════════════════════════
#  DEFINIÇÃO DOS EXERCÍCIOS
# ══════════════════════════════════════════════════════════════════

# ─── Q1 ──────────────────────────────────────────────────────────
def ex01():
    n, xbar, sigma, mu0, alpha = 25, 13.5, 4.4, 16, 0.05
    # a) bilateral  b) unilateral esquerda
    res_a = _teste_z_media(n, xbar, sigma, mu0, alpha, "bilateral")
    res_b = _teste_z_media(n, xbar, sigma, mu0, alpha, "esquerda")
    # retornamos o bilateral como principal, exibindo ambos nos passos
    ep = sigma / math.sqrt(n)
    za = res_a["estatistica"]; zca = res_a["critico"]
    zb = res_b["estatistica"]; zcb = res_b["critico"]
    passos = [
        f"Dados: n={n}, x̄={xbar}, σ={sigma}, μ₀={mu0}, α={alpha}",
        "─────────────────────────────────────────────",
        "  ITEM a) Bilateral  →  H₁: μ ≠ 16",
        f"  Erro-padrão = σ/√n = {sigma}/√{n} = {ep:.4f}",
        f"  Z = (x̄ − μ₀)/EP = ({xbar} − {mu0})/{ep:.4f} = {za:.4f}",
        f"  Z crítico (α/2={alpha/2}) = ±{zca:.4f}",
        f"  |Z|={abs(za):.4f} {'>' if res_a['rejeita'] else '≤'} {zca:.4f}  →  "
          + ("REJEITAR H₀" if res_a["rejeita"] else "NÃO REJEITAR H₀"),
        "─────────────────────────────────────────────",
        "  ITEM b) Unilateral esquerda  →  H₁: μ < 16",
        f"  Z = {zb:.4f}  |  Z crítico = −{zcb:.4f}",
        f"  Z={zb:.4f} {'<' if res_b['rejeita'] else '≥'} −{zcb:.4f}  →  "
          + ("REJEITAR H₀" if res_b["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res_a, passos

# ─── Q2 ──────────────────────────────────────────────────────────
def ex02():
    dados = [10,10,10,11,11,12,12,12,13,13,14,14,14,15]
    n, xbar, s = _stats(dados)
    mu0, alpha = 12.5, 0.05
    # a) bilateral  b) direita  c) esquerda
    res_a = _teste_t_media(int(n), xbar, s, mu0, alpha, "bilateral")
    res_b = _teste_t_media(int(n), xbar, s, mu0, alpha, "direita")
    res_c = _teste_t_media(int(n), xbar, s, mu0, alpha, "esquerda")
    passos = [
        f"Dados: {dados}",
        f"n={int(n)}, x̄={xbar:.4f}, s={s:.4f}, μ₀={mu0}, α={alpha}",
        "─────────────────────────────────────────────",
        "  ITEM a) H₀: μ=12,5  H₁: μ ≠ 12,5  (bilateral)",
        f"  GL = {int(n)-1}",
        f"  t = (x̄ − μ₀)/(s/√n) = ({xbar:.4f}−{mu0})/({s:.4f}/√{int(n)}) = {res_a['estatistica']:.4f}",
        f"  t crítico = ±{res_a['critico']:.4f}",
        f"  → " + ("REJEITAR H₀" if res_a["rejeita"] else "NÃO REJEITAR H₀"),
        "─────────────────────────────────────────────",
        "  ITEM b) H₁: μ > 12,5  (unilateral direita)",
        f"  t crítico = {res_b['critico']:.4f}",
        f"  → " + ("REJEITAR H₀" if res_b["rejeita"] else "NÃO REJEITAR H₀"),
        "─────────────────────────────────────────────",
        "  ITEM c) H₁: μ < 12,5  (unilateral esquerda)",
        f"  t crítico = −{res_c['critico']:.4f}",
        f"  → " + ("REJEITAR H₀" if res_c["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res_a, passos

# ─── Q3 ──────────────────────────────────────────────────────────
def ex03():
    # Distribuição amostral em classes: 5-10(3), 10-15(5), 15-20(8), 20-25(3), 25-30(2)
    midpts = [7.5, 12.5, 17.5, 22.5, 27.5]
    freqs  = [3, 5, 8, 3, 2]
    n      = sum(freqs)
    xbar   = sum(m*f for m,f in zip(midpts,freqs)) / n
    s2     = sum(f*(m-xbar)**2 for m,f in zip(midpts,freqs)) / (n-1)
    s      = math.sqrt(s2)
    mu0, alpha = 20, 0.025
    res = _teste_t_media(n, xbar, s, mu0, alpha, "bilateral")
    passos = [
        "Distribuição amostral em classes:",
        "  Classes:    5-10  10-15  15-20  20-25  25-30",
        f"  Frequências:   {freqs}",
        f"Pontos médios: {midpts}",
        f"n={n},  x̄ = Σ(fᵢmᵢ)/n = {xbar:.4f}",
        f"s² = Σfᵢ(mᵢ−x̄)²/(n−1) = {s2:.4f}   →   s = {s:.4f}",
        f"H₀: μ = {mu0}   H₁: μ ≠ {mu0}   α = {alpha} (bilateral)",
        f"GL = {n-1}",
        f"t = ({xbar:.4f} − {mu0}) / ({s:.4f}/√{n}) = {res['estatistica']:.4f}",
        f"t crítico = ±{res['critico']:.4f}",
        f"|t| = {abs(res['estatistica']):.4f} {'>' if res['rejeita'] else '≤'} {res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q4 ──────────────────────────────────────────────────────────
def ex04():
    dados = [41,50,49,54,50,47,52,49,
             50,52,50,47,49,51,46,50,49,50]
    n, xbar, s = _stats(dados)
    mu0, alpha = 50, 0.05
    # a) bilateral (variância desconhecida), b) unilateral
    res_a = _teste_t_media(int(n), xbar, s, mu0, alpha, "bilateral")
    res_b = _teste_t_media(int(n), xbar, s, mu0, alpha, "esquerda")
    passos = [
        f"Alturas de {int(n)} recém-nascidos (cm):",
        f"  {dados}",
        f"x̄ = {xbar:.4f}  s = {s:.4f}  n = {int(n)}",
        "─────────────────────────────────────────────",
        "  ITEM a) σ² = 2cm²  →  teste t  (variância composta)",
        f"  H₀: μ = {mu0}   H₁: μ ≠ {mu0}   α = {alpha}",
        f"  GL = {int(n)-1}",
        f"  t calc = {res_a['estatistica']:.4f}   t crit = ±{res_a['critico']:.4f}",
        "  → " + ("REJEITAR H₀" if res_a["rejeita"] else "NÃO REJEITAR H₀"),
        "─────────────────────────────────────────────",
        "  ITEM b) Variância desconhecida, unilateral esquerda",
        f"  t calc = {res_b['estatistica']:.4f}   t crit = −{res_b['critico']:.4f}",
        "  → " + ("REJEITAR H₀" if res_b["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res_a, passos

# ─── Q5 ──────────────────────────────────────────────────────────
def ex05():
    dados = [25,30,32,24,40,34,37,33,
             34,28,30,32,28,29,31]
    n, xbar, s = _stats(dados)
    mu0, alpha = 30, 0.10
    res = _teste_t_media(int(n), xbar, s, mu0, alpha, "bilateral")
    passos = [
        "15 animais — aumento de peso (g) após dieta:",
        f"  {dados}",
        f"n={int(n)},  x̄={xbar:.4f},  s={s:.4f}",
        f"H₀: μ = {mu0}   H₁: μ ≠ {mu0}   α = {alpha} (bilateral)",
        f"GL = {int(n)-1}",
        f"t = ({xbar:.4f}−{mu0}) / ({s:.4f}/√{int(n)}) = {res['estatistica']:.4f}",
        f"t crítico = ±{res['critico']:.4f}",
        f"|t| = {abs(res['estatistica']):.4f} {'>' if res['rejeita'] else '≤'} {res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q9 ──────────────────────────────────────────────────────────
def ex09():
    # 500 eleitores, 52% → x=260 favoráveis; p0=0.50
    n, x, p0, alpha = 500, 260, 0.50, 0.05
    res = _teste_z_proporcao(x, n, p0, alpha, "bilateral")
    p_hat = x/n
    ep = math.sqrt(p0*(1-p0)/n)
    passos = [
        f"n={n} eleitores,  x={x} favoráveis ao Partido Democrático",
        f"p̂ = {x}/{n} = {p_hat:.4f}",
        f"H₀: p = {p0}   H₁: p ≠ {p0}   α = {alpha}",
        f"Erro-padrão = √(p₀(1−p₀)/n) = √({p0}×{1-p0}/{n}) = {ep:.4f}",
        f"Z = (p̂ − p₀)/EP = ({p_hat:.4f}−{p0})/{ep:.4f} = {res['estatistica']:.4f}",
        f"Z crítico = ±{res['critico']:.4f}",
        f"|Z| = {abs(res['estatistica']):.4f} {'>' if res['rejeita'] else '≤'} {res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q10a ─────────────────────────────────────────────────────────
def ex10a():
    # Tabela: fumantes homens=12/90, mulheres=8/50; testar p_H=0,80 → p̂=78/140
    # item a: proporção de fumantes é 80%?  α=0,04
    total_fumantes = 12+8; n_total = 90+50
    p0, alpha = 0.80, 0.04
    res = _teste_z_proporcao(total_fumantes, n_total, p0, alpha, "bilateral")
    p_hat = total_fumantes/n_total
    ep = math.sqrt(p0*(1-p0)/n_total)
    passos = [
        "Tabela contingência: Fumantes vs Não-fumantes por sexo",
        f"Total fumantes: homens {12} + mulheres {8} = {total_fumantes}",
        f"Total amostrado: {n_total}   →   p̂ = {total_fumantes}/{n_total} = {p_hat:.4f}",
        f"H₀: p = {p0}   H₁: p ≠ {p0}   α = {alpha}",
        f"EP = √({p0}×{1-p0}/{n_total}) = {ep:.4f}",
        f"Z = ({p_hat:.4f}−{p0})/{ep:.4f} = {res['estatistica']:.4f}",
        f"Z crítico = ±{res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q10b ─────────────────────────────────────────────────────────
def ex10b():
    # proporção de homens que fumam cigarros com filtro é 70%?
    # Homens fumantes=12, com filtro=84 (do total 90)  →  p̂_filtro = 84/90? 
    # Relendo: homens fumam com filtro=84, sem filtro=16, total homens=90? não bate
    # tabela: cigarro c/ filtro: H=12, M=8; sem filtro: H=8, M=26; nao fumam: H=70, M=16
    # item b: p de homens com filtro = 70%
    n_h   = 90
    x_filtro_h = 12
    p0, alpha = 0.70, 0.04
    res = _teste_z_proporcao(x_filtro_h, n_h, p0, alpha, "bilateral")
    p_hat = x_filtro_h/n_h
    ep = math.sqrt(p0*(1-p0)/n_h)
    passos = [
        "Item b) Testar se proporção de homens que fumam com filtro = 70%",
        f"Homens fumantes com filtro: x={x_filtro_h},  n={n_h}",
        f"p̂ = {x_filtro_h}/{n_h} = {p_hat:.4f}",
        f"H₀: p = {p0}   H₁: p ≠ {p0}   α = {alpha}",
        f"EP = {ep:.4f}",
        f"Z = {res['estatistica']:.4f}   Z crit = ±{res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q10c ─────────────────────────────────────────────────────────
def ex10c():
    # proporção de fumantes femininas = 40%
    n_m = 50; x_fum_m = 8
    p0, alpha = 0.40, 0.01
    res = _teste_z_proporcao(x_fum_m, n_m, p0, alpha, "bilateral")
    p_hat = x_fum_m/n_m
    ep = math.sqrt(p0*(1-p0)/n_m)
    passos = [
        "Item c) Testar se proporção de fumantes femininas = 40%",
        f"Mulheres fumantes: x={x_fum_m},  n={n_m}",
        f"p̂ = {x_fum_m}/{n_m} = {p_hat:.4f}",
        f"H₀: p = {p0}   H₁: p ≠ {p0}   α = {alpha}",
        f"EP = {ep:.4f}",
        f"Z = {res['estatistica']:.4f}   Z crit = ±{res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q13 ─────────────────────────────────────────────────────────
def ex13():
    # 90 alunos reprovados em 400; testar se proporção > 0.40  α=8%
    n, x, p0, alpha = 400, 90, 0.40, 0.08
    # H1: p < 0.40 (estudante acredita que são menores)
    res = _teste_z_proporcao(x, n, p0, alpha, "esquerda")
    p_hat = x/n
    ep = math.sqrt(p0*(1-p0)/n)
    passos = [
        f"n={n} alunos,  reprovados={x}",
        f"p̂ = {x}/{n} = {p_hat:.4f}",
        "Afirmação: 40% reprovados. Testar H₁: p < 0,40 (α=8%)",
        f"H₀: p = {p0}   H₁: p < {p0}   α = {alpha}",
        f"EP = √({p0}×{1-p0}/{n}) = {ep:.4f}",
        f"Z = ({p_hat:.4f}−{p0})/{ep:.4f} = {res['estatistica']:.4f}",
        f"Z crítico = −{res['critico']:.4f}",
        f"Z={res['estatistica']:.4f} {'<' if res['rejeita'] else '≥'} −{res['critico']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q14 ─────────────────────────────────────────────────────────
def ex14():
    # Machos: n=47, s²=43  Fêmeas: n=31, s²=26.5  α=10%
    n1, s1 = 47, math.sqrt(43)
    n2, s2 = 31, math.sqrt(26.5)
    alpha, bil = 0.10, True
    res = _teste_f_variancias(n1, s1, n2, s2, alpha, bil)
    passos = [
        "Ratos injetados com doses em 72 horas.",
        f"Machos:  n₁={n1},  s₁² = 43",
        f"Fêmeas:  n₂={n2},  s₂² = 26,5",
        f"H₀: σ₁² = σ₂²   H₁: σ₁² ≠ σ₂²   α = {alpha} (bilateral)",
        f"GL₁ = {n1-1}   GL₂ = {n2-1}",
        f"F = s₁²/s₂² = 43/26,5 = {res['estatistica']:.4f}",
        f"F crítico superior = {res['critico']:.4f}",
        f"F crítico inferior = {res['critico_inf']:.4f}",
        f"F={res['estatistica']:.4f} {'∈ região de rejeição' if res['rejeita'] else '∉ região de rejeição'}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q15 ─────────────────────────────────────────────────────────
def ex15():
    # "Tome duas dietas quaisquer do exercício 19" — usamos A e B
    A = [22,26,24,26,27]; B = [28,21,22,21,15]
    n1, xb1, s1 = _stats(A)
    n2, xb2, s2 = _stats(B)
    alpha, bil = 0.10, True
    res = _teste_f_variancias(int(n1), s1, int(n2), s2, alpha, bil)
    res["grupos"]["labels"] = ["Dieta A", "Dieta B"]
    passos = [
        f"Dieta A: {A}   →  s_A = {s1:.4f}  s_A² = {s1**2:.4f}",
        f"Dieta B: {B}   →  s_B = {s2:.4f}  s_B² = {s2**2:.4f}",
        f"H₀: σ_A² = σ_B²   H₁: σ_A² ≠ σ_B²   α = {alpha}",
        f"GL₁={int(n1)-1}   GL₂={int(n2)-1}",
        f"F = {s1**2:.4f}/{s2**2:.4f} = {res['estatistica']:.4f}",
        f"F crítico sup = {res['critico']:.4f}   inf = {res['critico_inf']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q16 ─────────────────────────────────────────────────────────
def ex16():
    # α=10%, igualdade das variâncias para as duas marcas do Q18
    a = [14,16,2,11,9,10]; b = [4,16,1,9,10,10]
    n1, xb1, s1 = _stats(a)
    n2, xb2, s2 = _stats(b)
    alpha, bil = 0.10, True
    res = _teste_f_variancias(int(n1), s1, int(n2), s2, alpha, bil)
    res["grupos"]["labels"] = ["Marca A", "Marca B"]
    passos = [
        f"Marca A: {a}   s_A={s1:.4f}",
        f"Marca B: {b}   s_B={s2:.4f}",
        f"H₀: σ_A² = σ_B²   H₁: σ_A² ≠ σ_B²   α = {alpha}",
        f"GL₁={int(n1)-1}   GL₂={int(n2)-1}",
        f"F = {s1**2:.4f}/{s2**2:.4f} = {res['estatistica']:.4f}",
        f"F crítico sup = {res['critico']:.4f}   inf = {res['critico_inf']:.4f}",
        "→ " + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q17 ─────────────────────────────────────────────────────────
def ex17():
    n1, xb1, s1 = 60, 5.71, math.sqrt(43)
    n2, xb2, s2 = 35, 4.12, math.sqrt(28)
    alpha, bil  = 0.04, False
    res = TesteTPooled(n1, xb1, s1, n2, xb2, s2, alpha, bil).t_pooled()
    res["grupos"]["labels"] = ["Marca A", "Marca B"]
    gl  = n1+n2-2
    sp2 = ((n1-1)*s1**2+(n2-1)*s2**2)/gl; sp=math.sqrt(sp2)
    t=res["estatistica"]; tc=res["critico"]
    passos = [
        f"n₁={n1}, x̄₁={xb1}, s₁²=43   |   n₂={n2}, x̄₂={xb2}, s₂²=28",
        f"H₀: μ₁=μ₂   H₁: μ₁>μ₂   α={alpha} (unilateral direita)",
        f"GL = {n1}+{n2}−2 = {gl}",
        f"Sp² = [{n1-1}×{s1**2:.1f}+{n2-1}×{s2**2:.1f}]/{gl} = {sp2:.4f}",
        f"Sp  = √{sp2:.4f} = {sp:.4f}",
        f"t = ({xb1}−{xb2})/[{sp:.4f}·√(1/{n1}+1/{n2})] = {t:.4f}",
        f"t crítico = {tc:.4f}",
        f"t={t:.4f} {'>' if res['rejeita'] else '≤'} {tc:.4f}  →  "
          + ("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q18 ─────────────────────────────────────────────────────────
def ex18():
    a=[14,16,2,11,9,10]; b=[4,16,1,9,10,10]
    n1,xb1,s1=_stats(a); n2,xb2,s2=_stats(b)
    alpha,bil=0.05,True
    res=TesteTPooled(int(n1),xb1,s1,int(n2),xb2,s2,alpha,bil).t_pooled()
    res["grupos"]["labels"]=["Marca A","Marca B"]
    t=res["estatistica"]; tc=res["critico"]
    passos=[
        f"Marca A: {a}",f"Marca B: {b}",
        f"x̄_A={xb1:.4f}  s_A={s1:.4f}  n_A={int(n1)}",
        f"x̄_B={xb2:.4f}  s_B={s2:.4f}  n_B={int(n2)}",
        f"H₀: μ_A=μ_B   H₁: μ_A≠μ_B   α={alpha} (bilateral)",
        f"GL={int(n1+n2-2)}",
        f"t calc={t:.4f}   t crit=±{tc:.4f}",
        f"|t|={abs(t):.4f} {'>' if res['rejeita'] else '≤'} {tc:.4f}  →  "
          +("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q19 ─────────────────────────────────────────────────────────
def ex19():
    A=[22,26,24,26,27]; B=[28,21,22,21,15]
    n1,xb1,s1=_stats(A); n2,xb2,s2=_stats(B)
    alpha,bil=0.05,True
    res=TesteTPooled(int(n1),xb1,s1,int(n2),xb2,s2,alpha,bil).t_pooled()
    res["grupos"]["labels"]=["Dieta A","Dieta B"]
    t=res["estatistica"]; tc=res["critico"]
    passos=[
        f"Dieta A: {A}",f"Dieta B: {B}",
        f"x̄_A={xb1:.2f}  s_A={s1:.4f}  n_A={int(n1)}",
        f"x̄_B={xb2:.2f}  s_B={s2:.4f}  n_B={int(n2)}",
        f"H₀: μ_A=μ_B   H₁: μ_A≠μ_B   α={alpha}",
        f"GL={int(n1+n2-2)}",
        f"t calc={t:.4f}   t crit=±{tc:.4f}",
        f"|t|={abs(t):.4f} {'>' if res['rejeita'] else '≤'} {tc:.4f}  →  "
          +("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q20 ─────────────────────────────────────────────────────────
def ex20():
    def media_freq(faixas, freqs):
        n=sum(freqs); xm=sum(f*x for f,x in zip(freqs,faixas))/n
        s=math.sqrt(sum(f*(x-xm)**2 for f,x in zip(freqs,faixas))/(n-1))
        return n,xm,s
    faixas_m=[12.5,32.5,47.5,60.0,72.5]; fi_m=[7,10,8,5,3]
    faixas_h=[20.0,35.0,52.5,70.0,87.5]; fi_h=[8,15,12,7,3]
    n1,xb1,s1=media_freq(faixas_m,fi_m)
    n2,xb2,s2=media_freq(faixas_h,fi_h)
    alpha,bil=0.10,True
    res=TesteZDuasMedias(int(n1),xb1,s1,int(n2),xb2,s2,alpha,bil).z_duas_medias()
    res["grupos"]["labels"]=["Mulheres","Homens"]
    z=res["estatistica"]; zc=res["critico"]
    passos=[
        f"Mulheres: n={int(n1)}, x̄={xb1:.2f}, s={s1:.2f}",
        f"Homens:   n={int(n2)}, x̄={xb2:.2f}, s={s2:.2f}",
        f"H₀: μ_M=μ_H   H₁: μ_M≠μ_H   α={alpha} (bilateral)",
        "Grandes amostras → Teste Z  (σ ≈ s)",
        f"EP={res['extras']['Erro-padrão']}",
        f"Z calc={z:.4f}   Z crit=±{zc:.4f}",
        f"|Z|={abs(z):.4f} {'>' if res['rejeita'] else '≤'} {zc:.4f}  →  "
          +("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q21 ─────────────────────────────────────────────────────────
def ex21():
    x1,n1=162,300; x2,n2=120,200; alpha,bil=0.05,True
    res=TesteZDuasProporcoes(x1,n1,x2,n2,alpha,bil).z_proporcoes()
    res["grupos"]["labels"]=["São Paulo","Rio de Janeiro"]
    p1=x1/n1; p2=x2/n2; z=res["estatistica"]; zc=res["critico"]
    passos=[
        f"SP: n₁={n1}, x₁={x1} → p̂₁={p1:.4f}",
        f"RJ: n₂={n2}, x₂={x2} → p̂₂={p2:.4f}",
        f"H₀: p₁=p₂   H₁: p₁≠p₂   α={alpha} (bilateral)",
        f"p̂ pool={(x1+x2)/(n1+n2):.4f}",
        f"EP={res['extras']['Erro-padrão']}",
        f"Z={z:.4f}   Z crit=±{zc:.4f}",
        f"|Z|={abs(z):.4f} {'>' if res['rejeita'] else '≤'} {zc:.4f}  →  "
          +("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q22 ─────────────────────────────────────────────────────────
def ex22():
    x1,n1=150,1000; x2,n2=200,800; alpha,bil=0.05,True
    res=TesteZDuasProporcoes(x1,n1,x2,n2,alpha,bil).z_proporcoes()
    res["grupos"]["labels"]=["Processo 1","Processo 2"]
    p1=x1/n1; p2=x2/n2; z=res["estatistica"]; zc=res["critico"]
    passos=[
        f"Processo 1: n₁={n1}, rejeições={x1} → p̂₁={p1:.4f}",
        f"Processo 2: n₂={n2}, rejeições={x2} → p̂₂={p2:.4f}",
        f"H₀: p₁=p₂   H₁: p₁≠p₂   α={alpha}",
        f"p̂ pool={(x1+x2)/(n1+n2):.4f}",
        f"EP={res['extras']['Erro-padrão']}",
        f"Z={z:.4f}   Z crit=±{zc:.4f}",
        f"|Z|={abs(z):.4f} {'>' if res['rejeita'] else '≤'} {zc:.4f}  →  "
          +("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ─── Q23 ─────────────────────────────────────────────────────────
def ex23():
    x1,n1=240,500; x2,n2=340,500; alpha,bil=0.10,True
    res=TesteZDuasProporcoes(x1,n1,x2,n2,alpha,bil).z_proporcoes()
    res["grupos"]["labels"]=["Bairro X","Bairro Y"]
    p1=x1/n1; p2=x2/n2; z=res["estatistica"]; zc=res["critico"]
    passos=[
        f"Bairro X: n₁={n1}, possuidores={x1} → p̂₁={p1:.4f}",
        f"Bairro Y: n₂={n2}, possuidores={x2} → p̂₂={p2:.4f}",
        f"H₀: p₁=p₂   H₁: p₁≠p₂   α={alpha}",
        f"p̂ pool={(x1+x2)/(n1+n2):.4f}",
        f"EP={res['extras']['Erro-padrão']}",
        f"Z={z:.4f}   Z crit=±{zc:.4f}",
        f"|Z|={abs(z):.4f} {'>' if res['rejeita'] else '≤'} {zc:.4f}  →  "
          +("REJEITAR H₀" if res["rejeita"] else "NÃO REJEITAR H₀"),
    ]
    return res, passos

# ══════════════════════════════════════════════════════════════════
#  REGISTRO CENTRAL
# ══════════════════════════════════════════════════════════════════
EXERCICIOS = [
    # ── Média Populacional ────────────────────────────────────────
    {"id":"Q01","titulo":"Q01 — Média de 25 elementos (Z)","topico":"Média Populacional",
     "enunciado":"Amostra de 25 elementos: x̄=13,5; σ=4,4.\nTestar H₀: μ=16  contra  H₁: μ≠16 e μ<16  com α=0,05.",
     "resolver":ex01},
    {"id":"Q02","titulo":"Q02 — Diâmetros de 15 parafusos (t)","topico":"Média Populacional",
     "enunciado":"15 parafusos com diâmetros medidos.\nTestar H₀: μ=12,5 contra H₁: μ≠12,5 / μ>12,5 / μ<12,5  (α=0,05).",
     "resolver":ex02},
    {"id":"Q03","titulo":"Q03 — Distribuição amostral em classes (t)","topico":"Média Populacional",
     "enunciado":"Distribuição em 5 classes (5-10, 10-15, 15-20, 20-25, 25-30) com\nfrequências 3,5,8,3,2.\nTestar H₀: μ=20  H₁: μ≠20  α=2,5%.",
     "resolver":ex03},
    {"id":"Q04","titulo":"Q04 — Alturas de 20 recém-nascidos (t)","topico":"Média Populacional",
     "enunciado":"20 recém-nascidos; alturas em cm.\na) σ²=2 conhecida, testar μ=50 bilateral α=0,05.\nb) σ desconhecida, unilateral.",
     "resolver":ex04},
    {"id":"Q05","titulo":"Q05 — Aumento de peso de 15 animais (t)","topico":"Média Populacional",
     "enunciado":"15 animais com dieta 3 semanas. Aumentos de peso (g) registrados.\nTestar H₀: μ=30  H₁: μ≠30  α=10% (bilateral).",
     "resolver":ex05},
    # ── Proporção ─────────────────────────────────────────────────
    {"id":"Q09","titulo":"Q09 — Eleitores Partido Democrático (Z)","topico":"Proporção",
     "enunciado":"500 eleitores; 52% afirmam ser do Partido Democrático.\nTestar se a proporção real é 50%  α=0,05.",
     "resolver":ex09},
    {"id":"Q10a","titulo":"Q10a — Fumantes: proporção total 80%?","topico":"Proporção",
     "enunciado":"Tabela: fumantes vs não-fumantes por sexo.\na) Testar se proporção de fumantes é 80%  α=0,04.",
     "resolver":ex10a},
    {"id":"Q10b","titulo":"Q10b — Homens com cigarro filtro 70%?","topico":"Proporção",
     "enunciado":"b) Testar se proporção de homens que fumam com filtro é 70%  α=0,04.",
     "resolver":ex10b},
    {"id":"Q10c","titulo":"Q10c — Fumantes femininas 40%?","topico":"Proporção",
     "enunciado":"c) Testar se proporção de fumantes femininas é 40%  α=0,01.",
     "resolver":ex10c},
    {"id":"Q13","titulo":"Q13 — Reprovados em Estatística (Z)","topico":"Proporção",
     "enunciado":"Afirmação: 40% dos alunos são reprovados em Estatística.\n90 em 400 foram reprovados. Testar H₁: p<0,40  α=8%.",
     "resolver":ex13},
    # ── Igualdade de Duas Variâncias ──────────────────────────────
    {"id":"Q14","titulo":"Q14 — Variâncias Machos vs Fêmeas (F)","topico":"Igualdade de Duas Variâncias",
     "enunciado":"Machos: n=47, s²=43  |  Fêmeas: n=31, s²=26,5\nTestar igualdade das variâncias  α=10%.",
     "resolver":ex14},
    {"id":"Q15","titulo":"Q15 — Variâncias Dietas A e B (F)","topico":"Igualdade de Duas Variâncias",
     "enunciado":"Duas dietas do exercício 19 (grupos A e B).\nTestar igualdade das variâncias  α=10%.",
     "resolver":ex15},
    {"id":"Q16","titulo":"Q16 — Variâncias Marcas A e B (F)","topico":"Igualdade de Duas Variâncias",
     "enunciado":"Marcas A e B do exercício 18.\nTestar igualdade das variâncias  α=10%.",
     "resolver":ex16},
    # ── Igualdade de Duas Médias ──────────────────────────────────
    {"id":"Q17","titulo":"Q17 — Índice de vendas Marcas A e B (t)","topico":"Igualdade de Duas Médias",
     "enunciado":"Amostra 1: n₁=60, x̄=5,71, s²=43\nAmostra 2: n₂=35, x̄=4,12, s²=28\nH₀: μ₁=μ₂  H₁: μ₁>μ₂  α=0,04.",
     "resolver":ex17},
    {"id":"Q18","titulo":"Q18 — Supermercados Marca A vs B (t)","topico":"Igualdade de Duas Médias",
     "enunciado":"Índices de vendas em 6 supermercados por marca.\nTestar H₀: μ_A=μ_B  H₁: μ_A≠μ_B  α=5%.",
     "resolver":ex18},
    {"id":"Q19","titulo":"Q19 — Dietas A e B — ganho de peso (t)","topico":"Igualdade de Duas Médias",
     "enunciado":"Grupos A e B de ratos submetidos a dietas.\nTestar igualdade de médias  α=0,05.",
     "resolver":ex19},
    {"id":"Q20","titulo":"Q20 — Renda média Homens vs Mulheres (Z)","topico":"Igualdade de Duas Médias",
     "enunciado":"Tabelas de frequência de renda por sexo.\nTestar H₀: μ_M=μ_H  α=10%.",
     "resolver":ex20},
    # ── Igualdade de Duas Proporções ──────────────────────────────
    {"id":"Q21","titulo":"Q21 — Eleitores SP vs RJ (Z)","topico":"Igualdade de Duas Proporções",
     "enunciado":"SP: 300 eleitores  |  RJ: 200 eleitores\nTestar igualdade das proporções  α=0,05.",
     "resolver":ex21},
    {"id":"Q22","titulo":"Q22 — Processos de fechamento de latas (Z)","topico":"Igualdade de Duas Proporções",
     "enunciado":"Processo 1: 1000 latas, 150 rejeições\nProcesso 2: 800 latas, 200 rejeições\nTestar igualdade  α=0,05.",
     "resolver":ex22},
    {"id":"Q23","titulo":"Q23 — Videocassete Bairro X vs Y (Z)","topico":"Igualdade de Duas Proporções",
     "enunciado":"Bairro X: 240/500  |  Bairro Y: 340/500\nTestar diferença significativa  α=10%.",
     "resolver":ex23},
]

# ══════════════════════════════════════════════════════════════════
#  INTERFACE GRÁFICA
# ══════════════════════════════════════════════════════════════════
class PaginaExercicios(tk.Toplevel):

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Exercícios Resolvidos — Testes de Hipóteses")
        self.geometry("1380x920")
        self.minsize(1100, 740)
        self.configure(bg=COR_FUNDO)
        self._aplicar_estilo()
        self._construir_ui()

    def _aplicar_estilo(self):
        s = ttk.Style(self)
        try: s.theme_use("clam")
        except: pass
        s.configure(".",            background=COR_FUNDO, foreground=COR_TEXTO, borderwidth=0)
        s.configure("TFrame",       background=COR_FUNDO)
        s.configure("TLabel",       background=COR_FUNDO, foreground=COR_TEXTO)
        s.configure("TLabelframe",  background=COR_PAINEL, foreground=COR_DESTAQUE,
                    bordercolor=COR_BORDA, relief="groove")
        s.configure("TLabelframe.Label", background=COR_PAINEL,
                    foreground=COR_DESTAQUE, font=("Segoe UI",10,"bold"))
        s.configure("TSeparator",   background=COR_BORDA)
        s.configure("Accent.TButton", background=COR_DESTAQUE, foreground=COR_FUNDO,
                    font=("Segoe UI",10,"bold"), padding=(14,7), relief="flat")
        s.map("Accent.TButton", background=[("active","#74c7ec")])

    def _construir_ui(self):
        cab = tk.Frame(self, bg=COR_PAINEL, pady=10)
        cab.pack(fill="x")
        tk.Label(cab, text="📝  Exercícios Resolvidos", bg=COR_PAINEL,
                 fg=COR_DESTAQUE, font=("Segoe UI",15,"bold")).pack(side="left", padx=20)
        tk.Label(cab, text="Q01 – Q23 | Passo a passo + gráficos", bg=COR_PAINEL,
                 fg=COR_TEXTO_DIM, font=("Segoe UI",10)).pack(side="left", padx=4)

        corpo = ttk.Frame(self)
        corpo.pack(fill="both", expand=True, padx=10, pady=10)
        corpo.columnconfigure(0, weight=0, minsize=240)
        corpo.columnconfigure(1, weight=1)
        corpo.rowconfigure(0, weight=1)

        self._lista_lateral(corpo)
        self._area_principal(corpo)

    # ── Lista lateral ─────────────────────────────────────────────
    def _lista_lateral(self, pai):
        frame = ttk.LabelFrame(pai, text=" 📚  Exercícios ", padding=8)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Filtrar por tópico",
                  foreground=COR_TEXTO_DIM, font=("Segoe UI",9)).pack(anchor="w")
        self.var_topico = tk.StringVar(value="Todos")
        topicos = sorted(set(e["topico"] for e in EXERCICIOS))
        cb = ttk.Combobox(frame, textvariable=self.var_topico,
                          values=["Todos"]+topicos, state="readonly",
                          font=("Segoe UI",9), width=27)
        cb.pack(fill="x", pady=(2,8))
        cb.bind("<<ComboboxSelected>>", lambda _: self._atualizar_lista())
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)

        # Canvas rolável para os botões
        outer = tk.Frame(frame, bg=COR_FUNDO)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=COR_FUNDO, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._frame_botoes = tk.Frame(canvas, bg=COR_FUNDO)
        win = canvas.create_window((0,0), window=self._frame_botoes, anchor="nw")
        self._frame_botoes.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(win, width=e.width))
        # scroll com roda do mouse
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._botoes_ex = {}
        self._atualizar_lista()

    def _atualizar_lista(self):
        filtro = self.var_topico.get()
        for w in self._frame_botoes.winfo_children():
            w.destroy()
        self._botoes_ex.clear()

        # Agrupar por tópico
        topicos_ord = ["Média Populacional","Proporção",
                       "Igualdade de Duas Variâncias",
                       "Igualdade de Duas Médias",
                       "Igualdade de Duas Proporções"]
        grupos: dict = {}
        for ex in EXERCICIOS:
            if filtro != "Todos" and ex["topico"] != filtro:
                continue
            grupos.setdefault(ex["topico"], []).append(ex)

        for top in topicos_ord:
            if top not in grupos:
                continue
            # Cabeçalho do grupo
            tk.Label(self._frame_botoes, text=top, bg=COR_FUNDO,
                     fg=COR_ROXO, font=("Segoe UI",8,"bold"),
                     anchor="w", padx=4).pack(fill="x", pady=(6,2))
            for ex in grupos[top]:
                btn = tk.Button(
                    self._frame_botoes,
                    text=f"  {ex['id']}  {ex['titulo'].split('—')[1].strip()[:30]}",
                    bg=COR_CARD, fg=COR_TEXTO, relief="flat",
                    font=("Segoe UI",9), anchor="w", padx=8, pady=5,
                    cursor="hand2",
                    command=lambda e=ex: self._carregar(e),
                )
                btn.pack(fill="x", pady=1)
                btn.bind("<Enter>", lambda ev,b=btn: b.config(bg=COR_PAINEL, fg=COR_DESTAQUE))
                btn.bind("<Leave>", lambda ev,b=btn: b.config(bg=COR_CARD,   fg=COR_TEXTO))
                self._botoes_ex[ex["id"]] = btn

    # ── Área principal ────────────────────────────────────────────
    def _area_principal(self, pai):
        cont = ttk.Frame(pai)
        cont.grid(row=0, column=1, sticky="nsew")
        cont.columnconfigure(0, weight=1)
        cont.columnconfigure(1, weight=1)
        cont.rowconfigure(0, weight=1)
        cont.rowconfigure(1, weight=2)

        self._bloco_enunciado(cont)
        self._bloco_passos(cont)
        self._bloco_graficos(cont)
        self._placeholder()

    def _bloco_enunciado(self, pai):
        frame = ttk.LabelFrame(pai, text=" 📖  Enunciado & Resultado ", padding=12)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0,6), pady=(0,6))

        self.lbl_topico = tk.Label(frame, text="", bg=COR_CARD,
                                   fg=COR_ROXO, font=("Segoe UI",9,"italic"), padx=8, pady=3)
        self.lbl_topico.pack(anchor="w", pady=(0,6))

        self.txt_enunciado = tk.Text(frame, height=4, bg=COR_CARD, fg=COR_TEXTO,
                                     font=("Segoe UI",10), relief="flat",
                                     wrap="word", state="disabled")
        self.txt_enunciado.pack(fill="x")
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        # H0/H1
        hip = ttk.Frame(frame); hip.pack(fill="x", pady=(0,8))
        hip.columnconfigure(0, weight=1); hip.columnconfigure(1, weight=1)
        for col, (ch, tt) in enumerate([("h0","H₀"),("h1","H₁")]):
            f = tk.Frame(hip, bg=COR_CARD, padx=10, pady=8)
            f.grid(row=0, column=col, sticky="ew", padx=(0,6) if col==0 else 0)
            tk.Label(f, text=tt, bg=COR_CARD, fg=COR_TEXTO_DIM,
                     font=("Segoe UI",9)).pack(anchor="w")
            lbl = tk.Label(f, text="—", bg=COR_CARD, fg=COR_DESTAQUE,
                           font=("Consolas",11,"bold"))
            lbl.pack(anchor="w")
            setattr(self, f"lbl_{ch}", lbl)

        # Métricas
        met = ttk.Frame(frame); met.pack(fill="x", pady=(0,8))
        for i in range(4): met.columnconfigure(i, weight=1)
        self.cm = {}
        for i,(ch,tt) in enumerate([("estat","Estatística"),("crit","Valor Crítico"),
                                     ("pval","p-valor"),("gl","GL")]):
            f = tk.Frame(met, bg=COR_CARD, padx=8, pady=8)
            f.grid(row=0, column=i, sticky="ew", padx=(0,5) if i<3 else 0)
            tk.Label(f, text=tt, bg=COR_CARD, fg=COR_TEXTO_DIM,
                     font=("Segoe UI",8)).pack(anchor="w")
            lbl = tk.Label(f, text="—", bg=COR_CARD, fg=COR_TEXTO,
                           font=("Consolas",11,"bold"))
            lbl.pack(anchor="w")
            self.cm[ch] = lbl

        # Decisão
        self.lbl_dec = tk.Label(frame, text="Selecione um exercício →",
                                bg=COR_CARD, fg=COR_TEXTO_DIM,
                                font=("Segoe UI",11,"bold"), pady=10)
        self.lbl_dec.pack(fill="x")

    def _bloco_passos(self, pai):
        frame = ttk.LabelFrame(pai, text=" 🔢  Passo a Passo ", padding=12)
        frame.grid(row=0, column=1, sticky="nsew", pady=(0,6))

        self.txt_p = tk.Text(frame, bg=COR_CARD, fg=COR_TEXTO,
                             font=("Consolas",10), relief="flat",
                             wrap="word", state="disabled")
        sc = ttk.Scrollbar(frame, orient="vertical", command=self.txt_p.yview)
        self.txt_p.configure(yscrollcommand=sc.set)
        sc.pack(side="right", fill="y")
        self.txt_p.pack(fill="both", expand=True)
        self.txt_p.tag_configure("titulo",  foreground=COR_DESTAQUE, font=("Segoe UI",10,"bold"))
        self.txt_p.tag_configure("passo",   foreground=COR_TEXTO,    font=("Consolas",10))
        self.txt_p.tag_configure("sep",     foreground=COR_BORDA)
        self.txt_p.tag_configure("rejeita", foreground=COR_VERMELHO, font=("Consolas",10,"bold"))
        self.txt_p.tag_configure("ok",      foreground=COR_VERDE,    font=("Consolas",10,"bold"))
        self.txt_p.tag_configure("extra",   foreground=COR_TEXTO_DIM,font=("Consolas",9))

    def _bloco_graficos(self, pai):
        frame = ttk.LabelFrame(pai, text=" 📈  Visualização ", padding=6)
        frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 3.4), facecolor=COR_FUNDO)
        self.fig.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.87, wspace=0.30)
        for ax in self.axes:
            ax.set_facecolor(COR_PAINEL)
            ax.tick_params(colors=COR_TEXTO_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(COR_BORDA)

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_fig.get_tk_widget().pack(fill="both", expand=True)

    def _placeholder(self):
        for ax in self.axes:
            ax.clear(); ax.set_facecolor(COR_PAINEL)
            ax.text(0.5, 0.5, "Selecione um exercício na lista lateral →",
                    ha="center", va="center", color=COR_TEXTO_DIM,
                    fontsize=10, transform=ax.transAxes)
            for sp in ax.spines.values(): sp.set_edgecolor(COR_BORDA)
        self.canvas_fig.draw()

    # ── Carrega exercício ─────────────────────────────────────────
    def _carregar(self, ex: dict):
        for b in self._botoes_ex.values(): b.config(bg=COR_CARD, fg=COR_TEXTO)
        if ex["id"] in self._botoes_ex:
            self._botoes_ex[ex["id"]].config(bg=COR_PAINEL, fg=COR_DESTAQUE)

        res, passos = ex["resolver"]()

        # Enunciado
        self.lbl_topico.config(text=f"  {ex['topico']}  ")
        self.txt_enunciado.config(state="normal")
        self.txt_enunciado.delete("1.0","end")
        self.txt_enunciado.insert("end", ex["enunciado"])
        self.txt_enunciado.config(state="disabled")

        # Hipóteses e métricas
        self.lbl_h0.config(text=res["h0"])
        self.lbl_h1.config(text=res["h1"])
        tp=res["tipo"]; est=res["estatistica"]; crit=res["critico"]
        pv=res["p_valor"]; bil=res["bilateral"]
        self.cm["estat"].config(text=f"{est:.4f}")
        self.cm["crit"].config(text=f"±{crit:.4f}" if bil else f"{crit:.4f}")
        self.cm["pval"].config(text=f"{pv:.4f}",
                               fg=COR_VERMELHO if pv<res["alpha"] else COR_VERDE)
        gl_v = res.get("gl", res.get("gl1","—"))
        self.cm["gl"].config(text=f"{gl_v:.2f}" if isinstance(gl_v,float) else str(gl_v))

        if res["rejeita"]:
            self.lbl_dec.config(text="✖  REJEITAR H₀", bg="#3d1f2b", fg=COR_VERMELHO)
        else:
            self.lbl_dec.config(text="✔  NÃO REJEITAR H₀", bg="#1f3d2b", fg=COR_VERDE)

        # Passos
        self.txt_p.config(state="normal")
        self.txt_p.delete("1.0","end")
        self.txt_p.insert("end", f"{ex['titulo']}\n", "titulo")
        self.txt_p.insert("end", "─"*54+"\n", "sep")
        for i,p in enumerate(passos,1):
            txt = f"  {i:>2}.  {p}\n"
            if "REJEITAR H₀" in p and "NÃO" not in p:
                tag = "rejeita"
            elif "NÃO REJEITAR" in p:
                tag = "ok"
            elif p.startswith("─"):
                tag = "sep"
            else:
                tag = "passo"
            self.txt_p.insert("end", txt, tag)
        self.txt_p.insert("end", "\n── Valores intermediários ──\n", "titulo")
        for k,v in res.get("extras",{}).items():
            self.txt_p.insert("end", f"  {k:<26} {v}\n", "extra")
        self.txt_p.config(state="disabled")

        # Gráficos
        self._desenhar(res, ex["titulo"])

    # ── Gráficos ──────────────────────────────────────────────────
    def _desenhar(self, r, titulo):
        for ax in self.axes:
            ax.clear(); ax.set_facecolor(COR_PAINEL)
            ax.tick_params(colors=COR_TEXTO_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(COR_BORDA)

        tipo = r.get("tipo","Z")
        if tipo == "F":
            self._graf_f_dist(self.axes[0], r)
        else:
            self._graf_dist(self.axes[0], r)
        self._graf_grupos(self.axes[1], r)
        self.fig.suptitle(titulo, color=COR_TEXTO_DIM, fontsize=9, y=0.99)
        self.canvas_fig.draw()

    def _graf_dist(self, ax, r):
        tp=r["tipo"]; est=r["estatistica"]; crit=r["critico"]
        bil=r["bilateral"]; gl=r.get("gl",None)
        lado=r.get("lado","bilateral")
        lim=max(4.0,abs(est)+0.8)
        x=np.linspace(-lim,lim,600)
        y=stats.t.pdf(x,df=gl) if tp=="t" else stats.norm.pdf(x)
        ax.plot(x,y,color=COR_DESTAQUE,lw=1.8)
        # regiões
        if lado=="bilateral":
            ax.fill_between(x,y,where=(x>=crit), color=COR_VERMELHO,alpha=0.30)
            ax.fill_between(x,y,where=(x<=-crit),color=COR_VERMELHO,alpha=0.30)
            ax.axvline(-crit,color=COR_VERMELHO,lw=1.2,ls=":")
        elif lado=="direita":
            ax.fill_between(x,y,where=(x>=crit), color=COR_VERMELHO,alpha=0.30)
        else:
            ax.fill_between(x,y,where=(x<=-crit),color=COR_VERMELHO,alpha=0.30)
        cor_e=COR_VERMELHO if r["rejeita"] else COR_VERDE
        ax.axvline(est,  color=cor_e,      lw=2.0,ls="--",label=f"{tp} calc={est:.3f}")
        ax.axvline(crit, color=COR_VERMELHO,lw=1.2,ls=":",
                   label=f"{tp} crit={'±' if bil else ''}{crit:.3f}")
        nome=f"t (gl={gl:.0f})" if tp=="t" else "Distribuição Normal"
        ax.set_title(nome,color=COR_TEXTO,fontsize=9)
        ax.set_xlabel(tp,color=COR_TEXTO_DIM,fontsize=8)
        ax.legend(fontsize=7.5,facecolor=COR_CARD,edgecolor=COR_BORDA,
                  labelcolor=COR_TEXTO,loc="upper right")
        ax.annotate(f"p={r['p_valor']:.4f}",xy=(0.04,0.92),xycoords="axes fraction",
                    color=COR_AMARELO,fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",fc=COR_CARD,ec=COR_BORDA,lw=0.8))

    def _graf_f_dist(self, ax, r):
        gl1=r["gl1"]; gl2=r["gl2"]; F=r["estatistica"]
        f_sup=r["critico"]; f_inf=r["critico_inf"]
        x_max=max(f_sup*2.5, F*1.5, 6)
        x=np.linspace(0.01,x_max,600)
        y=stats.f.pdf(x,gl1,gl2)
        ax.plot(x,y,color=COR_DESTAQUE,lw=1.8)
        ax.fill_between(x,y,where=(x>=f_sup),color=COR_VERMELHO,alpha=0.30)
        ax.fill_between(x,y,where=(x<=f_inf),color=COR_VERMELHO,alpha=0.30)
        cor_e=COR_VERMELHO if r["rejeita"] else COR_VERDE
        ax.axvline(F,    color=cor_e,      lw=2.0,ls="--",label=f"F calc={F:.3f}")
        ax.axvline(f_sup,color=COR_VERMELHO,lw=1.2,ls=":",label=f"F crit sup={f_sup:.3f}")
        ax.axvline(f_inf,color=COR_AMARELO, lw=1.2,ls=":",label=f"F crit inf={f_inf:.3f}")
        ax.set_title(f"Distribuição F (gl₁={gl1}, gl₂={gl2})",color=COR_TEXTO,fontsize=9)
        ax.set_xlabel("F",color=COR_TEXTO_DIM,fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5,facecolor=COR_CARD,edgecolor=COR_BORDA,
                  labelcolor=COR_TEXTO,loc="upper right")
        ax.annotate(f"p={r['p_valor']:.4f}",xy=(0.04,0.92),xycoords="axes fraction",
                    color=COR_AMARELO,fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",fc=COR_CARD,ec=COR_BORDA,lw=0.8))

    def _graf_grupos(self, ax, r):
        g=r["grupos"]; modo=g["modo"]
        lbls=g["labels"]; cent=g["centros"]; disp=g["dispersoes"]
        cores=[COR_DESTAQUE,COR_ROXO]

        # Modo especial: uma proporção (dois grupos = p̂ vs p₀)
        if modo=="proporcoes_uma":
            bars=ax.bar(lbls,cent,color=[COR_DESTAQUE,COR_AMARELO],
                        width=0.4,edgecolor=COR_FUNDO,lw=1.2,zorder=3)
            for bar,v in zip(bars,cent):
                ax.text(bar.get_x()+bar.get_width()/2,v+0.02,
                        f"{v:.1%}",ha="center",va="bottom",
                        color=COR_TEXTO,fontsize=9,fontweight="bold",zorder=5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:.0%}"))
            ax.set_ylim(0, min(1.0, max(cent)+0.15))
            ax.set_title("p̂ vs p₀",color=COR_TEXTO,fontsize=9)
            ax.grid(axis="y",alpha=0.15,color=COR_BORDA)
            cor_b=COR_VERMELHO if r["rejeita"] else COR_VERDE
            for sp in ax.spines.values(): sp.set_edgecolor(cor_b); sp.set_linewidth(1.5)
            return

        # Modo variâncias
        if modo=="variancias":
            bars=ax.bar(lbls,cent,color=cores,width=0.4,edgecolor=COR_FUNDO,lw=1.2,zorder=3)
            for bar,v in zip(bars,cent):
                ax.text(bar.get_x()+bar.get_width()/2,v+max(cent)*0.04,
                        f"{v:.2f}",ha="center",va="bottom",
                        color=COR_TEXTO,fontsize=9,fontweight="bold",zorder=5)
            ax.set_title("Variâncias s²",color=COR_TEXTO,fontsize=9)
            ax.grid(axis="y",alpha=0.15,color=COR_BORDA)
            cor_b=COR_VERMELHO if r["rejeita"] else COR_VERDE
            for sp in ax.spines.values(): sp.set_edgecolor(cor_b); sp.set_linewidth(1.5)
            return

        # Médias e proporções (duas amostras)
        bars=ax.bar(lbls,cent,color=cores,width=0.45,edgecolor=COR_FUNDO,lw=1.2,zorder=3)
        ax.errorbar(lbls,cent,yerr=disp,fmt="none",color=COR_TEXTO,capsize=7,lw=1.5,zorder=4)
        for bar,v in zip(bars,cent):
            fmt=f"{v:.1%}" if modo in ("proporcoes","proporcoes_uma") else f"{v:.2f}"
            ax.text(bar.get_x()+bar.get_width()/2,v+max(disp)*0.12,
                    fmt,ha="center",va="bottom",
                    color=COR_TEXTO,fontsize=9,fontweight="bold",zorder=5)
        if modo=="proporcoes":
            p_pool=g.get("p_pool",sum(cent)/2)
            ax.axhline(p_pool,color=COR_AMARELO,lw=1.2,ls="--",label=f"p̂ pool={p_pool:.3f}")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:.0%}"))
            ax.set_ylim(0,min(1.0,max(cent)+max(disp)*2.5))
            ax.legend(fontsize=7.5,facecolor=COR_CARD,edgecolor=COR_BORDA,labelcolor=COR_TEXTO)
            titulo_g="Proporções Comparadas"; erro_lbl="±IC 95%"
        else:
            ymin=max(0,min(cent)-max(disp)*2.2)
            ymax=max(cent)+max(disp)*2.2
            ax.set_ylim(ymin,ymax)
            titulo_g="Médias Comparadas"; erro_lbl="±desvio-padrão"
        ax.set_title(titulo_g,color=COR_TEXTO,fontsize=9)
        ax.set_ylabel(erro_lbl,color=COR_TEXTO_DIM,fontsize=7.5)
        ax.grid(axis="y",alpha=0.15,color=COR_BORDA)
        cor_b=COR_VERMELHO if r["rejeita"] else COR_VERDE
        for sp in ax.spines.values(): sp.set_edgecolor(cor_b); sp.set_linewidth(1.5)


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = PaginaExercicios(root)
    app.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()