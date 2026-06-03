# 📊 Projeto Estatística — Testes de Hipóteses

Projeto em Python para realizar testes de hipóteses estatísticos de forma modular e didática.
Cada caso é independente, fácil de modificar e reutilizar.

---


## 🧪 Casos cobertos

| Caso | Teste       | Quando usar                                      |
|------|-------------|--------------------------------------------------|
| 1    | Teste Z     | Comparar duas médias com **variâncias conhecidas** |
| 2    | Teste t pooled | Comparar duas médias, variâncias **desconhecidas e iguais** |
| 3    | Teste t Welch  | Comparar duas médias, variâncias **desconhecidas e diferentes** |
| 4    | Teste Z     | Comparar duas **proporções**                     |

---

## ▶️ Como usar

### 1. Instale as dependências

```bash
python -m pip install scipy
```

Ou, se tiver um `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```


### 2. Execute

```bash
python Testehipoteses.py
```

O terminal mostrará o passo a passo completo de cada teste: hipóteses, estatística calculada, valor crítico, p-valor e conclusão.

---

## 📐 Parâmetros das funções

### Casos 1, 2 e 3 — Médias

| Parâmetro  | Descrição                          |
|------------|------------------------------------|
| `n1`, `n2` | Tamanho das amostras               |
| `media1`, `media2` | Médias amostrais             |
| `s1`, `s2` | Desvios-padrão amostrais (casos 2 e 3) |
| `sigma1`, `sigma2` | Desvios-padrão populacionais (caso 1) |
| `alpha`    | Nível de significância (padrão: 0.05) |
| `bilateral`| `True` para H₁: μ₁ ≠ μ₂ / `False` para H₁: μ₁ > μ₂ |

### Caso 4 — Proporções

| Parâmetro  | Descrição                          |
|------------|------------------------------------|
| `x1`, `x2` | Número de sucessos em cada grupo   |
| `n1`, `n2` | Tamanho total de cada grupo        |
| `alpha`    | Nível de significância (padrão: 0.05) |
| `bilateral`| `True` para H₁: p₁ ≠ p₂ / `False` para H₁: p₁ > p₂ |

---

## 📖 Conceitos

**Hipótese nula (H₀):** afirma que não há diferença entre os grupos.  
**Hipótese alternativa (H₁):** afirma que existe diferença.  
**Nível de significância (α):** probabilidade máxima de rejeitar H₀ sendo ela verdadeira (erro tipo I). Valor padrão: 0,05 (5%).  
**p-valor:** probabilidade de observar um resultado tão extremo quanto o obtido, assumindo H₀ verdadeira. Se p-valor < α, rejeita-se H₀.

> ⚠️ **Não rejeitar H₀ não significa provar que H₀ é verdadeira** — significa apenas que os dados não forneceram evidência suficiente para descartá-la.

---

## 🔧 Dependências

- Python 3.8+
- [scipy](https://scipy.org/) — cálculo de distribuições t e normal

---

## 📝 Licença

Projeto acadêmico de livre uso.