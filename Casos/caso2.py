from util import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 2 — Teste t para duas médias
#           variâncias iguais (pooled)
# ─────────────────────────────────────────────
class TesteTPooled:
    def __init__(self, n1, media1, s1, n2, media2, s2, alpha, bilateral):
        self.n1 = n1
        self.media1 = media1
        self.s1 = s1
        self.n2 = n2
        self.media2 = media2
        self.s2 = s2
        self.alpha = alpha
        self.bilateral = bilateral
        
    def t_pooled(self):
        gl      = self.n1 + self.n2 - 2
        sp2     = ((self.n1-1)*self.s1**2 + (self.n2-1)*self.s2**2) / gl
        sp      = math.sqrt(sp2)
        t       = (self.media1 - self.media2) / (sp * math.sqrt(1/self.n1 + 1/self.n2))
        ar      = self.alpha / 2 if self.bilateral else self.alpha
        t_crit  = stats.t.ppf(1 - ar, df=gl)
        p_valor = 2*(1 - stats.t.cdf(abs(t), df=gl)) if self.bilateral else 1 - stats.t.cdf(t, df=gl)
        rejeita = abs(t) > t_crit if self.bilateral else t > t_crit
        return dict(
            tipo="t", estatistica=t, critico=t_crit, p_valor=p_valor,
            rejeita=rejeita, bilateral=self.bilateral, alpha=self.alpha, gl=gl,
            h0="μ₁ = μ₂", h1="μ₁ ≠ μ₂" if self.bilateral else "μ₁ > μ₂",
            extras={"GL": gl, "Sp²": f"{sp2:.4f}", "Sp": f"{sp:.4f}"},
            grupos={"labels": ["Grupo 1", "Grupo 2"],
                    "centros": [self.media1, self.media2],
                    "dispersoes": [self.s1, self.s2],
                    "modo": "medias"},
        )

