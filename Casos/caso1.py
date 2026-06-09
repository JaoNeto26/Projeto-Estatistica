from util import separador, linha, decisao
import math
from scipy import stats
# ─────────────────────────────────────────────
#  CASO 1 — Teste Z para duas médias
#           (variâncias populacionais conhecidas)
# ─────────────────────────────────────────────
class TesteZDuasMedias:
    def __init__(self, n1, media1, sigma1, n2, media2, sigma2, alpha, bilateral):
        self.n1 = n1
        self.media1 = media1
        self.sigma1 = sigma1
        self.n2 = n2
        self.media2 = media2
        self.sigma2 = sigma2
        self.alpha = alpha
        self.bilateral = bilateral

    def z_duas_medias(self): 
        erro    = math.sqrt(self.sigma1**2 / self.n1 + self.sigma2**2 / self.n2)
        z       = (self.media1 - self.media2) / erro
        ar      = self.alpha / 2 if self.bilateral else self.alpha
        # Para teste bilateral, o valor crítico é simétrico (±z), para unilateral é apenas positivo
        z_crit  = stats.norm.ppf(1 - ar)
        # O p-valor é a probabilidade de obter um resultado tão extremo quanto o observado, sob H₀
        p_valor = 2*(1 - stats.norm.cdf(abs(z))) if self.bilateral else 1 - stats.norm.cdf(z)

        rejeita = abs(z) > z_crit if self.bilateral else z > z_crit
        return dict(
            tipo="Z", estatistica=z, critico=z_crit, p_valor=p_valor,
            rejeita=rejeita, bilateral=self.bilateral, alpha=self.alpha,
            h0="μ₁ = μ₂", h1="μ₁ ≠ μ₂" if self.bilateral else "μ₁ > μ₂",
            extras={"Erro-padrão": f"{erro:.4f}"},
            grupos={"labels": ["Grupo 1", "Grupo 2"],
                    "centros": [self.media1, self.media2],
                    "dispersoes": [self.sigma1, self.sigma2],
                    "modo": "medias"},
        )
   
