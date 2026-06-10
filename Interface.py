"""
Interface Gráfica — Testes de Hipóteses
========================================
Executa os 4 testes estatísticos com interface Tkinter + Matplotlib.
Compatível com as funções já existentes no projeto.


"""
from Casos.caso1 import TesteZDuasMedias
from Casos.caso2 import TesteTPooled
from Casos.caso3 import TesteTWelch
from Casos.caso4 import TesteZDuasProporcoes
from Exercicios import PaginaExercicios
import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats



# ══════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES VISUAIS
# ══════════════════════════════════════════════════════════════════

COR_FUNDO       = "#1e1e2e"
COR_PAINEL      = "#2a2a3e"
COR_CARD        = "#313145"
COR_BORDA       = "#44445a"
COR_TEXTO       = "#cdd6f4"
COR_TEXTO_DIM   = "#7f849c"
COR_DESTAQUE    = "#89b4fa"
COR_VERDE       = "#a6e3a1"
COR_VERMELHO    = "#f38ba8"
COR_AMARELO     = "#f9e2af"
COR_ROXO        = "#cba6f7"
FONTE_TITULO    = ("Segoe UI", 13, "bold")
FONTE_LABEL     = ("Segoe UI", 10)
FONTE_ENTRADA   = ("Segoe UI", 10)
FONTE_RESULTADO = ("Consolas", 10)
FONTE_MONO      = ("Consolas", 11, "bold")

DADOS_PADRAO = {
    "Caso 1 — Teste Z (σ conhecidas)": {
        "campos": ["n₁", "x̄₁", "σ₁", "n₂", "x̄₂", "σ₂"],
        "valores": [30, 72, 8, 35, 68, 10],
        "alpha": 0.05, "bilateral": True,
    },
    "Caso 2 — Teste t Pooled (var. iguais)": {
        "campos": ["n₁", "x̄₁", "s₁", "n₂", "x̄₂", "s₂"],
        "valores": [30, 72, 8, 35, 68, 10],
        "alpha": 0.05, "bilateral": True,
    },
    "Caso 3 — Teste t Welch (var. diferentes)": {
        "campos": ["n₁", "x̄₁", "s₁", "n₂", "x̄₂", "s₂"],
        "valores": [30, 72, 8, 35, 68, 10],
        "alpha": 0.05, "bilateral": True,
    },
    "Caso 4 — Teste Z (proporções)": {
        "campos": ["x₁ (sucess.)", "n₁", "x₂ (sucess.)", "n₂"],
        "valores": [40, 60, 50, 80],
        "alpha": 0.05, "bilateral": True,
    },
}


# ══════════════════════════════════════════════════════════════════
#  JANELA PRINCIPAL
# ══════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Testes de Hipóteses — Análise Estatística")
        self.geometry("1280x860")
        self.minsize(1100, 720)
        self.configure(bg=COR_FUNDO)
        self._aplicar_estilo()
        self._construir_ui()

    # ── Estilo ttk ────────────────────────────────────────────────
    def _aplicar_estilo(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=COR_FUNDO, foreground=COR_TEXTO,
                    font=FONTE_LABEL, borderwidth=0)
        s.configure("TFrame",      background=COR_FUNDO)
        s.configure("Card.TFrame", background=COR_CARD,  relief="flat")
        s.configure("TLabel",      background=COR_FUNDO, foreground=COR_TEXTO)
        s.configure("Card.TLabel", background=COR_CARD,  foreground=COR_TEXTO)
        s.configure("Dim.TLabel",  background=COR_CARD,  foreground=COR_TEXTO_DIM,
                    font=("Segoe UI", 9))
        s.configure("TLabelframe", background=COR_PAINEL, foreground=COR_DESTAQUE,
                    bordercolor=COR_BORDA, relief="groove")
        s.configure("TLabelframe.Label", background=COR_PAINEL,
                    foreground=COR_DESTAQUE, font=("Segoe UI", 10, "bold"))
        s.configure("TCombobox", fieldbackground=COR_CARD, background=COR_CARD,
                    foreground=COR_TEXTO, selectbackground=COR_DESTAQUE,
                    arrowcolor=COR_DESTAQUE)
        s.map("TCombobox", fieldbackground=[("readonly", COR_CARD)])
        s.configure("TEntry", fieldbackground=COR_CARD, foreground=COR_TEXTO,
                    insertcolor=COR_TEXTO, relief="flat")
        s.configure("Accent.TButton", background=COR_DESTAQUE, foreground=COR_FUNDO,
                    font=("Segoe UI", 11, "bold"), padding=(20, 8), relief="flat")
        s.map("Accent.TButton",
              background=[("active", "#74c7ec"), ("pressed", "#89dceb")])
        s.configure("TCheckbutton", background=COR_CARD, foreground=COR_TEXTO)
        s.map("TCheckbutton", background=[("active", COR_CARD)])
        s.configure("TSeparator", background=COR_BORDA)

    # ── Estrutura principal ───────────────────────────────────────
    def _construir_ui(self):
        # Cabeçalho
        cab = tk.Frame(self, bg=COR_PAINEL, pady=10)
        cab.pack(fill="x")
        tk.Label(cab, text="📊  Testes de Hipóteses", bg=COR_PAINEL,
                 fg=COR_DESTAQUE, font=("Segoe UI", 16, "bold")).pack(side="left", padx=20)
        tk.Label(cab, text="Comparação de Duas Amostras", bg=COR_PAINEL,
                 fg=COR_TEXTO_DIM, font=("Segoe UI", 10)).pack(side="left", padx=4)
        tk.Button(
            cab, text="📝  Exercícios Resolvidos",
            bg=COR_DESTAQUE, fg=COR_FUNDO,
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=self._abrir_exercicios,
        ).pack(side="right", padx=20)

        # Corpo
        corpo = ttk.Frame(self)
        corpo.pack(fill="both", expand=True, padx=12, pady=10)
        corpo.columnconfigure(0, weight=2, minsize=340)
        corpo.columnconfigure(1, weight=3, minsize=380)
        corpo.rowconfigure(0, weight=1)
        corpo.rowconfigure(1, weight=2)

        # Painéis
        self.painel_esq    = self._painel_esquerdo(corpo)
        self.painel_dir    = self._painel_direito(corpo)
        self.painel_graf   = self._painel_grafico(corpo)

        self.painel_esq.grid( row=0, column=0, rowspan=2, sticky="nsew", padx=(0,6), pady=0)
        self.painel_dir.grid( row=0, column=1, sticky="nsew", padx=(0,0), pady=(0,6))
        self.painel_graf.grid(row=1, column=1, sticky="nsew")

        # Popula campos com dados padrão do primeiro teste
        self._ao_trocar_teste()

    # ── Painel esquerdo ───────────────────────────────────────────
    def _painel_esquerdo(self, pai):
        frame = ttk.LabelFrame(pai, text=" ⚙  Configuração ", padding=12)

        # Seleção do teste
        ttk.Label(frame, text="Teste estatístico",
                  style="TLabel").pack(anchor="w", pady=(0,4))
        self.var_teste = tk.StringVar()
        self.combo_teste = ttk.Combobox(
            frame, textvariable=self.var_teste,
            values=list(DADOS_PADRAO.keys()),
            state="readonly", font=FONTE_ENTRADA,
        )
        self.combo_teste.current(0)
        self.combo_teste.pack(fill="x", pady=(0,10))
        self.combo_teste.bind("<<ComboboxSelected>>", lambda _: self._ao_trocar_teste())

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        # Campos dinâmicos
        ttk.Label(frame, text="Dados do teste",
                  style="TLabel").pack(anchor="w", pady=(0,6))
        self.frame_campos = ttk.Frame(frame)
        self.frame_campos.pack(fill="x")
        self.entradas: dict[str, ttk.Entry] = {}

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Alpha + bilateral
        cfg = ttk.Frame(frame)
        cfg.pack(fill="x")
        cfg.columnconfigure(0, weight=1)
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="Nível α").grid(row=0, column=0, sticky="w")
        self.var_alpha = tk.StringVar(value="0.05")
        ttk.Entry(cfg, textvariable=self.var_alpha, width=8,
                  font=FONTE_ENTRADA).grid(row=1, column=0, sticky="ew", padx=(0,6))

        self.var_bilateral = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg, text="Bilateral", variable=self.var_bilateral,
                        style="TCheckbutton").grid(row=1, column=1, sticky="w")

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Botão calcular
        ttk.Button(frame, text="▶  Calcular", style="Accent.TButton",
                   command=self._calcular).pack(fill="x", pady=(0,4))

        # Dados padrão (referência visual)
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(frame, text="Dados padrão do sistema",
                  foreground=COR_TEXTO_DIM, font=("Segoe UI", 9)).pack(anchor="w")
        self.lbl_padrao = tk.Text(
            frame, height=6, bg=COR_CARD, fg=COR_TEXTO_DIM,
            font=("Consolas", 9), relief="flat", state="disabled",
            insertbackground=COR_TEXTO,
        )
        self.lbl_padrao.pack(fill="x", pady=(4,0))

        return frame

    # ── Painel direito (resultados) ───────────────────────────────
    def _painel_direito(self, pai):
        frame = ttk.LabelFrame(pai, text=" 📋  Resultados ", padding=14)

        # Hipóteses
        hip = ttk.Frame(frame)
        hip.pack(fill="x", pady=(0,10))
        hip.columnconfigure(0, weight=1)
        hip.columnconfigure(1, weight=1)
        self._card_hip(hip, "H₀", "—", 0, "h0")
        self._card_hip(hip, "H₁", "—", 1, "h1")

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        # Métricas
        metricas = ttk.Frame(frame)
        metricas.pack(fill="x", pady=(0,10))
        for i in range(4):
            metricas.columnconfigure(i, weight=1)

        self.cards_metrica = {}
        itens = [
            ("estat",  "Estatística", "—"),
            ("critico","Valor Crítico","—"),
            ("pvalor", "p-valor",     "—"),
            ("gl",     "GL",          "—"),
        ]
        for i, (chave, titulo, val) in enumerate(itens):
            c = self._card_metrica(metricas, titulo, val, i)
            self.cards_metrica[chave] = c

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        # Região crítica
        rc_frame = ttk.Frame(frame)
        rc_frame.pack(fill="x", pady=(0,8))
        ttk.Label(rc_frame, text="Região crítica",
                  foreground=COR_TEXTO_DIM, font=("Segoe UI", 9)).pack(anchor="w")
        self.lbl_regiao = ttk.Label(rc_frame, text="—",
                                    font=FONTE_MONO, foreground=COR_AMARELO)
        self.lbl_regiao.pack(anchor="w")

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        # Decisão
        self.lbl_decisao = tk.Label(
            frame, text="Execute um cálculo para ver a decisão",
            bg=COR_CARD, fg=COR_TEXTO_DIM,
            font=("Segoe UI", 11, "bold"),
            relief="flat", pady=10,
        )
        self.lbl_decisao.pack(fill="x")

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        # Interpretação
        ttk.Label(frame, text="Interpretação",
                  foreground=COR_TEXTO_DIM, font=("Segoe UI", 9)).pack(anchor="w")
        self.txt_interpretacao = tk.Text(
            frame, height=4, bg=COR_CARD, fg=COR_TEXTO,
            font=("Segoe UI", 10), relief="flat", wrap="word",
            state="disabled",
        )
        self.txt_interpretacao.pack(fill="x", pady=(4,0))

        # Extras
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(frame, text="Valores intermediários",
                  foreground=COR_TEXTO_DIM, font=("Segoe UI", 9)).pack(anchor="w")
        self.txt_extras = tk.Text(
            frame, height=4, bg=COR_CARD, fg=COR_TEXTO_DIM,
            font=("Consolas", 9), relief="flat", state="disabled",
        )
        self.txt_extras.pack(fill="x", pady=(4,0))

        return frame

    def _card_hip(self, pai, titulo, valor, col, chave):
        f = tk.Frame(pai, bg=COR_CARD, padx=10, pady=8)
        f.grid(row=0, column=col, sticky="ew", padx=(0,6) if col==0 else 0)
        tk.Label(f, text=titulo, bg=COR_CARD, fg=COR_TEXTO_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w")
        lbl = tk.Label(f, text=valor, bg=COR_CARD, fg=COR_DESTAQUE,
                       font=FONTE_MONO)
        lbl.pack(anchor="w")
        setattr(self, f"lbl_{chave}", lbl)

    def _card_metrica(self, pai, titulo, valor, col):
        f = tk.Frame(pai, bg=COR_CARD, padx=8, pady=8)
        f.grid(row=0, column=col, sticky="ew", padx=(0,5) if col<3 else 0)
        tk.Label(f, text=titulo, bg=COR_CARD, fg=COR_TEXTO_DIM,
                 font=("Segoe UI", 8)).pack(anchor="w")
        lbl = tk.Label(f, text=valor, bg=COR_CARD, fg=COR_TEXTO,
                       font=("Consolas", 11, "bold"))
        lbl.pack(anchor="w")
        return lbl

    # ── Painel gráfico ────────────────────────────────────────────
    def _painel_grafico(self, pai):
        frame = ttk.LabelFrame(pai, text=" 📈  Visualização ", padding=6)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 3.2),
                                            facecolor=COR_FUNDO)
        self.fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14,
                                  top=0.88, wspace=0.32)
        for ax in self.axes:
            ax.set_facecolor(COR_PAINEL)
            ax.tick_params(colors=COR_TEXTO_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(COR_BORDA)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        return frame

    # ── Troca de teste ────────────────────────────────────────────
    def _ao_trocar_teste(self):
        nome   = self.var_teste.get()
        config = DADOS_PADRAO[nome]

        # Recria campos
        for w in self.frame_campos.winfo_children():
            w.destroy()
        self.entradas.clear()

        self.frame_campos.columnconfigure(0, weight=1)
        self.frame_campos.columnconfigure(1, weight=1)

        for i, (campo, valor) in enumerate(zip(config["campos"], config["valores"])):
            row, col = divmod(i, 2)
            sub = ttk.Frame(self.frame_campos)
            sub.grid(row=row, column=col, sticky="ew",
                     padx=(0,5) if col==0 else 0, pady=3)
            ttk.Label(sub, text=campo,
                      foreground=COR_TEXTO_DIM,
                      font=("Segoe UI", 9)).pack(anchor="w")
            e = ttk.Entry(sub, font=FONTE_ENTRADA, width=10)
            e.insert(0, str(valor))
            e.pack(fill="x")
            self.entradas[campo] = e

        self.var_alpha.set(str(config["alpha"]))
        self.var_bilateral.set(config["bilateral"])

        # Atualiza dados padrão
        self.lbl_padrao.config(state="normal")
        self.lbl_padrao.delete("1.0", "end")
        for campo, valor in zip(config["campos"], config["valores"]):
            self.lbl_padrao.insert("end", f"  {campo:<18} {valor}\n")
        self.lbl_padrao.insert("end", f"  {'α':<18} {config['alpha']}\n")
        self.lbl_padrao.insert("end", f"  {'Bilateral':<18} {config['bilateral']}")
        self.lbl_padrao.config(state="disabled")

    # ── Calcular ──────────────────────────────────────────────────
    def _calcular(self):
        try:
            alpha     = float(self.var_alpha.get())
            bilateral = self.var_bilateral.get()
            nome      = self.var_teste.get()
            vals      = {k: float(v.get()) for k, v in self.entradas.items()}

            if nome.startswith("Caso 1"):
                res = TesteZDuasMedias(
                    int(vals["n₁"]), vals["x̄₁"], vals["σ₁"],
                    int(vals["n₂"]), vals["x̄₂"], vals["σ₂"],
                    alpha, bilateral
                ).z_duas_medias()
            elif nome.startswith("Caso 2"):
                res = TesteTPooled(
                    int(vals["n₁"]), vals["x̄₁"], vals["s₁"],
                    int(vals["n₂"]), vals["x̄₂"], vals["s₂"],
                    alpha, bilateral,
                ).t_pooled()
            elif nome.startswith("Caso 3"):
                res = TesteTWelch(
                    int(vals["n₁"]), vals["x̄₁"], vals["s₁"],
                    int(vals["n₂"]), vals["x̄₂"], vals["s₂"],
                    alpha, bilateral,
                ).t_welch()
            else:
                res = TesteZDuasProporcoes(
                    int(vals["x₁ (sucess.)"]), int(vals["n₁"]),
                    int(vals["x₂ (sucess.)"]), int(vals["n₂"]),
                    alpha, bilateral,
                ).z_proporcoes()

            self._atualizar_resultados(res)
            self._atualizar_graficos(res)

        except ValueError as e:
            messagebox.showerror("Entrada inválida",
                                 f"Verifique os valores inseridos.\n\nDetalhe: {e}")
        except Exception as e:
            messagebox.showerror("Erro no cálculo", str(e))

    # ── Atualiza resultados ───────────────────────────────────────
    def _atualizar_resultados(self, r: dict):
        tp   = r["tipo"]
        est  = r["estatistica"]
        crit = r["critico"]
        pv   = r["p_valor"]
        bil  = r["bilateral"]

        self.lbl_h0.config(text=r["h0"])
        self.lbl_h1.config(text=r["h1"])

        self.cards_metrica["estat"].config( text=f"{est:.4f}")
        self.cards_metrica["critico"].config(
            text=f"±{crit:.4f}" if bil else f"{crit:.4f}")
        self.cards_metrica["pvalor"].config(
            text=f"{pv:.4f}",
            fg=COR_VERMELHO if pv < r["alpha"] else COR_VERDE,
        )
        gl_txt = f"{r['gl']:.2f}" if "gl" in r else "—"
        self.cards_metrica["gl"].config(text=gl_txt)

        if bil:
            self.lbl_regiao.config(
                text=f"|{tp}| > {crit:.4f}  →  {tp} < -{crit:.4f}  ou  {tp} > {crit:.4f}")
        else:
            self.lbl_regiao.config(text=f"{tp} > {crit:.4f}")

        if r["rejeita"]:
            self.lbl_decisao.config(
                text="✖  REJEITAR H₀", bg="#3d1f2b", fg=COR_VERMELHO)
        else:
            self.lbl_decisao.config(
                text="✔  NÃO REJEITAR H₀", bg="#1f3d2b", fg=COR_VERDE)

        interp = self._gerar_interpretacao(r)
        self.txt_interpretacao.config(state="normal")
        self.txt_interpretacao.delete("1.0", "end")
        self.txt_interpretacao.insert("end", interp)
        self.txt_interpretacao.config(state="disabled")

        self.txt_extras.config(state="normal")
        self.txt_extras.delete("1.0", "end")
        for chave, val in r.get("extras", {}).items():
            self.txt_extras.insert("end", f"  {chave:<22} {val}\n")
        self.txt_extras.config(state="disabled")

    def _gerar_interpretacao(self, r: dict) -> str:
        alpha = r["alpha"]
        pv    = r["p_valor"]
        nivel = int((1 - alpha) * 100)
        if r["rejeita"]:
            return (
                f"Com nível de significância de {alpha} ({nivel}% de confiança), "
                f"há evidência estatística suficiente para rejeitar H₀ "
                f"(p-valor = {pv:.4f} < α = {alpha}). "
                f"Conclui-se que existe diferença significativa entre os grupos."
            )
        return (
            f"Com nível de significância de {alpha} ({nivel}% de confiança), "
            f"não há evidência estatística suficiente para rejeitar H₀ "
            f"(p-valor = {pv:.4f} ≥ α = {alpha}). "
            f"Isso não prova que os grupos são iguais — apenas que os dados "
            f"não forneceram evidência suficiente de diferença."
        )

    # ── Atualiza gráficos ─────────────────────────────────────────
    def _atualizar_graficos(self, r: dict):
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(COR_PAINEL)
            ax.tick_params(colors=COR_TEXTO_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(COR_BORDA)

        self._grafico_distribuicao(self.axes[0], r)
        self._grafico_grupos(self.axes[1], r)
        self.fig.suptitle(
            self.var_teste.get(),
            color=COR_TEXTO_DIM, fontsize=9, y=0.98,
        )
        self.canvas.draw()

    def _grafico_distribuicao(self, ax, r: dict):
        tp   = r["tipo"]
        est  = r["estatistica"]
        crit = r["critico"]
        bil  = r["bilateral"]
        gl   = r.get("gl", None)

        lim  = max(4.0, abs(est) + 0.8)
        x    = np.linspace(-lim, lim, 600)
        y    = stats.t.pdf(x, df=gl) if tp == "t" else stats.norm.pdf(x)

        ax.plot(x, y, color=COR_DESTAQUE, linewidth=1.8)

        # Regiões de rejeição
        ax.fill_between(x, y, where=(x >= crit),  color=COR_VERMELHO, alpha=0.35)
        if bil:
            ax.fill_between(x, y, where=(x <= -crit), color=COR_VERMELHO, alpha=0.35)

        # Valor calculado
        cor_est = COR_VERMELHO if r["rejeita"] else COR_VERDE
        ax.axvline(est,  color=cor_est,    lw=1.8, ls="--",
                   label=f"{tp} calc. = {est:.3f}")
        ax.axvline(crit, color=COR_VERMELHO, lw=1.2, ls=":",
                   label=f"{tp} crit. = {'±' if bil else ''}{crit:.3f}")
        if bil:
            ax.axvline(-crit, color=COR_VERMELHO, lw=1.2, ls=":")

        nome_dist = f"t (gl={gl:.0f})" if tp == "t" else "Normal padrão"
        ax.set_title(nome_dist, color=COR_TEXTO, fontsize=9)
        ax.set_xlabel(tp, color=COR_TEXTO_DIM, fontsize=8)
        ax.legend(fontsize=7.5, facecolor=COR_CARD, edgecolor=COR_BORDA,
                  labelcolor=COR_TEXTO, loc="upper right")

        # Anotação do p-valor
        ax.annotate(f"p = {r['p_valor']:.4f}",
                    xy=(0.04, 0.92), xycoords="axes fraction",
                    color=COR_AMARELO, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc=COR_CARD,
                              ec=COR_BORDA, lw=0.8))

    def _grafico_grupos(self, ax, r: dict):
        g    = r["grupos"]
        modo = g["modo"]
        lbls = g["labels"]
        cent = g["centros"]
        disp = g["dispersoes"]

        cores = [COR_DESTAQUE, COR_ROXO]

        bars = ax.bar(lbls, cent, color=cores, width=0.45,
                      edgecolor=COR_FUNDO, linewidth=1.2, zorder=3)
        ax.errorbar(lbls, cent, yerr=disp, fmt="none",
                    color=COR_TEXTO, capsize=7, lw=1.5, zorder=4)

        for bar, v in zip(bars, cent):
            fmt = f"{v:.1%}" if modo == "proporcoes" else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + max(disp) * 0.12,
                    fmt, ha="center", va="bottom",
                    color=COR_TEXTO, fontsize=9, fontweight="bold", zorder=5)

        if modo == "proporcoes":
            p_pool = g.get("p_pool", sum(cent)/2)
            ax.axhline(p_pool, color=COR_AMARELO, lw=1.2, ls="--",
                       label=f"p̂ pool = {p_pool:.3f}")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_ylim(0, min(1.0, max(cent) + max(disp) * 2.5))
            ax.legend(fontsize=7.5, facecolor=COR_CARD,
                      edgecolor=COR_BORDA, labelcolor=COR_TEXTO)
            titulo = "Proporções"
            erro_lbl = "±IC 95%"
        else:
            ymin = max(0, min(cent) - max(disp) * 2.2)
            ymax = max(cent) + max(disp) * 2.2
            ax.set_ylim(ymin, ymax)
            titulo = "Médias"
            erro_lbl = "±desvio-padrão"

        ax.set_title(titulo, color=COR_TEXTO, fontsize=9)
        ax.set_ylabel(erro_lbl, color=COR_TEXTO_DIM, fontsize=7.5)
        ax.yaxis.label.set_color(COR_TEXTO_DIM)
        ax.grid(axis="y", alpha=0.15, color=COR_BORDA)

        # Indica decisão com borda colorida
        cor_borda = COR_VERMELHO if r["rejeita"] else COR_VERDE
        for spine in ax.spines.values():
            spine.set_edgecolor(cor_borda)
            spine.set_linewidth(1.5)


    # ── Abre página de exercícios ─────────────────────────────────
    def _abrir_exercicios(self):
        PaginaExercicios(self)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()