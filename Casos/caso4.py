from util import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 4 — Teste Z para duas proporções
# ─────────────────────────────────────────────
class TesteZDuasProporcoes:
    def __init__(self, x1, n1, x2, n2, alpha, bilateral):
        self.x1 = x1
        self.n1 = n1
        self.x2 = x2
        self.n2 = n2
        self.alpha = alpha
        self.bilateral = bilateral
    
    def z_proporcoes(self):
            p1      = self.x1 / self.n1
            p2      = self.x2 / self.n2
            p_pool  = (self.x1 + self.x2) / (self.n1 + self.n2)
            erro    = math.sqrt(p_pool * (1 - p_pool) * (1/self.n1 + 1/self.n2))
            z       = (p1 - p2) / erro
            ar      = self.alpha / 2 if self.bilateral else self.alpha
            z_crit  = stats.norm.ppf(1 - ar)
            p_valor = 2*(1 - stats.norm.cdf(abs(z))) if self.bilateral else 1 - stats.norm.cdf(z)
            rejeita = abs(z) > z_crit if self.bilateral else z > z_crit
            conds   = all([self.n1*p1>=5, self.n1*(1-p1)>=5, self.n2*p2>=5, self.n2*(1-p2)>=5])
            ic1     = 1.96 * math.sqrt(p1*(1-p1)/self.n1)
            ic2     = 1.96 * math.sqrt(p2*(1-p2)/self.n2)
            return dict(
                tipo="Z", estatistica=z, critico=z_crit, p_valor=p_valor,
                rejeita=rejeita, bilateral=self.bilateral, alpha=self.alpha,
                h0="p₁ = p₂", h1="p₁ ≠ p₂" if self.bilateral else "p₁ > p₂",
                extras={"p̂₁": f"{p1:.4f}", "p̂₂": f"{p2:.4f}",
                        "p̂ pooled": f"{p_pool:.4f}", "Erro-padrão": f"{erro:.4f}",
                        "Condições válidas": "✔ Sim" if conds else "✘ Verificar"},
                grupos={"labels": ["Grupo 1", "Grupo 2"],
                        "centros": [p1, p2],
                        "dispersoes": [ic1, ic2],
                        "p_pool": p_pool,
                        "modo": "proporcoes"},
            )