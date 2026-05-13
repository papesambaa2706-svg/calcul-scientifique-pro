import streamlit as st
import numpy as np
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, optimize, integrate, signal
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
FORMULES = {
    "Gaussienne 1D":          r"f(x) = A\,\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
    "Gaussienne normalisée":  r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
    "FWHM":                   r"\text{FWHM} = 2\sqrt{2\ln 2}\,\sigma \approx 2.3548\,\sigma",
    "Aire totale":            r"\int_{-\infty}^{+\infty}f(x)\,dx = A\sigma\sqrt{2\pi}",
    "Gaussienne 2D":          r"f(x,y)=A\exp\!\left(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\right)",
    "Entropie":               r"H = \tfrac{1}{2}\ln(2\pi e\sigma^2)",
    "Fonction caractéristique":r"\phi(t)=\exp\!\left(i\mu t - \tfrac{\sigma^2 t^2}{2}\right)",
    "Moment centré d'ordre n":r"\langle(x-\mu)^n\rangle = \begin{cases}0 & n\text{ impair}\\(n-1)!!\,\sigma^n & n\text{ pair}\end{cases}",
    "Convolution":            r"(f_1*f_2)(x) = A_{12}\exp\!\left(-\frac{(x-\mu_{12})^2}{2(\sigma_1^2+\sigma_2^2)}\right)",
    "Transformée de Fourier": r"\hat{f}(\xi)=A\sigma\sqrt{2\pi}\exp(-2\pi^2\sigma^2\xi^2-2\pi i\xi\mu)",
}

APPLICATIONS = {
    "Optique — profil laser TEM₀₀": "Intensité I(r) = I₀ exp(-2r²/w²)",
    "Statistiques — loi normale":    "Erreurs de mesure, TCL",
    "Mécanique quantique — paquet":  "ψ(x) = A exp(-x²/4σ²) exp(ik₀x)",
    "Traitement signal — filtre":    "h(t) = exp(-t²/2σ²) (filtre gaussien)",
    "Imagerie — PSF":                "Point Spread Function en microscopie",
    "Finance — VaR":                 "Distribution des rendements",
}


# ============================================================
# MOTEUR GAUSSIEN
# ============================================================
class GaussianEngine:
    """Moteur scientifique complet pour distributions gaussiennes."""

    def __init__(self, amplitude: float, sigma: float, mu: float = 0.0):
        if sigma <= 0:
            raise ValueError(f"σ doit être > 0, reçu : {sigma}")
        if amplitude <= 0:
            raise ValueError(f"Amplitude doit être > 0, reçu : {amplitude}")
        self.A = amplitude
        self.sigma = sigma
        self.mu = mu

    # --- Évaluation ---
    def eval_1d(self, x: np.ndarray) -> np.ndarray:
        return self.A * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    def eval_1d_normalise(self, x: np.ndarray) -> np.ndarray:
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    def eval_2d(self, X: np.ndarray, Y: np.ndarray,
                mu_y: float = 0.0, sigma_y: float = None) -> np.ndarray:
        sy = sigma_y or self.sigma
        return self.A * np.exp(
            -0.5 * ((X - self.mu) ** 2 / self.sigma ** 2 +
                    (Y - mu_y) ** 2 / sy ** 2)
        )

    # --- Propriétés analytiques ---
    @property
    def fwhm(self) -> float:
        return 2.3548 * self.sigma

    @property
    def aire(self) -> float:
        return self.A * self.sigma * np.sqrt(2 * np.pi)

    @property
    def entropie(self) -> float:
        return 0.5 * np.log(2 * np.pi * np.e * self.sigma ** 2)

    @property
    def variance(self) -> float:
        return self.sigma ** 2

    def moments(self, ordre_max: int = 6) -> dict:
        """Moments statistiques analytiques."""
        m = {"μ₁ (moyenne)": self.mu, "μ₂ (variance)": self.sigma**2,
             "Skewness": 0.0, "Kurtosis excess": 0.0}
        for n in range(3, ordre_max + 1):
            if n % 2 == 0:
                coef = 1
                for k in range(n - 1, 0, -2):
                    coef *= k
                m[f"Moment centré μ_{n}"] = coef * self.sigma ** n
            else:
                m[f"Moment centré μ_{n}"] = 0.0
        return m

    def fft_analytique(self, freqs: np.ndarray) -> np.ndarray:
        """Transformée de Fourier analytique."""
        return (self.A * self.sigma * np.sqrt(2 * np.pi) *
                np.exp(-2 * np.pi**2 * self.sigma**2 * freqs**2) *
                np.exp(-2j * np.pi * freqs * self.mu))

    def convolution(self, sigma2: float, A2: float = 1.0) -> "GaussianEngine":
        """Convolution analytique de deux gaussiennes."""
        sigma_conv = np.sqrt(self.sigma**2 + sigma2**2)
        A_conv = self.A * A2 * np.sqrt(2 * np.pi) * \
                 np.sqrt(self.sigma**2 * sigma2**2 / (self.sigma**2 + sigma2**2))
        return GaussianEngine(A_conv, sigma_conv, self.mu)

    def fit_data(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Ajustement gaussien sur données expérimentales."""
        try:
            def gauss(x, A, mu, sigma):
                return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            p0 = [y.max(), x[np.argmax(y)], (x[-1]-x[0])/4]
            popt, pcov = optimize.curve_fit(gauss, x, y, p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            y_pred = gauss(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            return {
                "A": popt[0], "μ": popt[1], "σ": abs(popt[2]),
                "σ_A": perr[0], "σ_μ": perr[1], "σ_σ": perr[2],
                "R²": 1 - ss_res/ss_tot,
                "FWHM_fit": 2.3548 * abs(popt[2]),
            }
        except Exception as e:
            return {"erreur": str(e)}

    def intervalle_confiance(self, niveau: float = 0.95) -> tuple:
        """Retourne [μ-kσ, μ+kσ] pour un niveau de confiance donné."""
        k = stats.norm.ppf((1 + niveau) / 2)
        return self.mu - k * self.sigma, self.mu + k * self.sigma

    def diagnostiquer(self) -> list:
        diag = []
        diag.append({"Test": "Amplitude", "Valeur": f"{self.A:.3f}",
                     "Statut": "✅ OK" if self.A > 0 else "❌",
                     "Note": "Amplitude positive"})
        diag.append({"Test": "Largeur σ", "Valeur": f"{self.sigma:.3f}",
                     "Statut": "✅ OK" if 0.01 < self.sigma < 100 else "⚠️ Extrême",
                     "Note": "σ dans plage raisonnable"})
        diag.append({"Test": "FWHM", "Valeur": f"{self.fwhm:.3f}",
                     "Statut": "✅", "Note": "2.3548σ"})
        diag.append({"Test": "Aire", "Valeur": f"{self.aire:.4f}",
                     "Statut": "✅", "Note": "Aσ√(2π)"})
        return diag


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def gaussian_page():
    st.markdown("## 📊 Profil Gaussien Avancé")
    st.markdown("*Modélisation, analyse multi-gaussienne, FFT, convolution, ajustement*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Gaussienne 1D",
        "🌐 Gaussienne 2D",
        "🔄 Multi-gaussiennes",
        "📡 Spectre & Convolution",
        "🔬 Ajustement données",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : GAUSSIENNE 1D
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")
            amplitude = st.slider("Amplitude A", 0.1, 10.0, 1.0, 0.1)
            sigma     = st.slider("Écart-type σ", 0.1, 5.0, 1.0, 0.05)
            mu        = st.slider("Centre μ", -10.0, 10.0, 0.0, 0.1)
            x_range   = st.slider("Plage [-R, R]", 5.0, 30.0, 10.0, 1.0)
            show_fill = st.checkbox("Remplissage", True)
            show_norm = st.checkbox("Superposer normalisée", False)
            show_ic   = st.checkbox("Intervalle de confiance 95%", True)
            mode_y    = st.radio("Échelle Y", ["Linéaire", "Log"], horizontal=True)

            try:
                engine = GaussianEngine(amplitude, sigma, mu)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            st.markdown("### 📐 Propriétés analytiques")
            st.metric("FWHM",  f"{engine.fwhm:.4f}")
            st.metric("Aire",  f"{engine.aire:.4f}")
            st.metric("Entropie", f"{engine.entropie:.4f}")

            ic_lo, ic_hi = engine.intervalle_confiance(0.95)
            st.metric("IC 95%", f"[{ic_lo:.3f}, {ic_hi:.3f}]")

        with col2:
            x = np.linspace(-x_range + mu, x_range + mu, 2000)
            y = engine.eval_1d(x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y if mode_y == "Linéaire" else np.log10(np.maximum(y, 1e-12)),
                mode='lines', name='f(x)',
                line=dict(color='#00ccff', width=3),
                fill='tozeroy' if show_fill else 'none',
                fillcolor='rgba(0,204,255,0.15)'
            ))

            if show_norm:
                y_norm = engine.eval_1d_normalise(x)
                fig.add_trace(go.Scatter(
                    x=x, y=y_norm if mode_y == "Linéaire" else np.log10(np.maximum(y_norm, 1e-12)),
                    mode='lines', name='Normalisée',
                    line=dict(color='#7700ff', width=2, dash='dash')
                ))

            if show_ic:
                fig.add_vrect(x0=ic_lo, x1=ic_hi,
                              fillcolor='rgba(0,255,136,0.08)',
                              line_color='rgba(0,255,136,0.4)',
                              annotation_text="IC 95%")

            # FWHM
            y_half = amplitude / 2
            x_fwhm = sigma * np.sqrt(2 * np.log(2))
            fig.add_shape(type='line',
                x0=mu - x_fwhm, x1=mu + x_fwhm,
                y0=y_half, y1=y_half,
                line=dict(color='#ffcc00', width=2, dash='dot'))
            fig.add_annotation(x=mu, y=y_half,
                text=f"FWHM={engine.fwhm:.3f}",
                font=dict(color='#ffcc00'), showarrow=False, yshift=10)

            # Centre μ
            fig.add_vline(x=mu, line_color='rgba(255,255,255,0.4)',
                          line_dash='dash', annotation_text=f"μ={mu:.2f}")

            fig.update_layout(
                title=f"Gaussienne — A={amplitude}, σ={sigma}, μ={mu}",
                xaxis_title="x",
                yaxis_title="f(x)" if mode_y == "Linéaire" else "log₁₀(f(x))",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Moments
            st.markdown("#### 📊 Moments statistiques")
            moments = engine.moments(6)
            cols_m = st.columns(3)
            for i, (k, v) in enumerate(moments.items()):
                with cols_m[i % 3]:
                    st.metric(k, f"{v:.4e}" if abs(v) > 1000 else f"{v:.4f}")

            # Export
            df_exp = pd.DataFrame({"x": x, "f(x)": y})
            st.download_button("💾 Export CSV",
                               df_exp.to_csv(index=False).encode(),
                               "gaussienne_1d.csv", "text/csv")

    # ============================================================
    # TAB 2 : GAUSSIENNE 2D
    # ============================================================
    with tab2:
        st.markdown("### 🌐 Gaussienne 2D")
        col1, col2 = st.columns([1, 2])

        with col1:
            amp_2d   = st.slider("Amplitude A", 0.1, 10.0, 1.0, 0.1, key="a2d")
            sigma_x  = st.slider("σ_x", 0.1, 5.0, 1.0, 0.1, key="sx")
            sigma_y  = st.slider("σ_y", 0.1, 5.0, 1.5, 0.1, key="sy")
            mu_x     = st.slider("μ_x", -5.0, 5.0, 0.0, 0.1, key="mx")
            mu_y     = st.slider("μ_y", -5.0, 5.0, 0.0, 0.1, key="my")
            mode_2d  = st.radio("Vue", ["Surface 3D", "Contour 2D", "Les deux"], horizontal=True)

        with col2:
            x2 = np.linspace(-8, 8, 120)
            y2 = np.linspace(-8, 8, 120)
            X2, Y2 = np.meshgrid(x2, y2)
            eng2 = GaussianEngine(amp_2d, sigma_x, mu_x)
            Z2 = eng2.eval_2d(X2, Y2, mu_y=mu_y, sigma_y=sigma_y)

            if mode_2d in ["Surface 3D", "Les deux"]:
                fig2 = go.Figure(data=[go.Surface(
                    z=Z2, x=x2, y=y2,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],
                                [0.65,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.6),
                )])
                fig2.update_layout(
                    title="Gaussienne 2D — Surface",
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff', title='x'),
                        yaxis=dict(color='#c0d0ff', title='y'),
                        zaxis=dict(color='#c0d0ff', title='f(x,y)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c0d0ff'),
                    height=480,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig2, use_container_width=True)

            if mode_2d in ["Contour 2D", "Les deux"]:
                fig_ct = go.Figure(data=[go.Contour(
                    z=Z2, x=x2, y=y2,
                    colorscale=[[0,'#020817'],[0.4,'#7700ff'],[0.7,'#00ccff'],[1,'#ffffff']],
                    contours=dict(coloring='heatmap', showlabels=True,
                                 labelfont=dict(color='white', size=9)),
                    colorbar=dict(tickfont=dict(color='#c0d0ff'), title='f(x,y)')
                )])
                # Ellipse FWHM
                theta = np.linspace(0, 2*np.pi, 200)
                kfwhm = np.sqrt(2 * np.log(2))
                fig_ct.add_trace(go.Scatter(
                    x=mu_x + sigma_x * kfwhm * np.cos(theta),
                    y=mu_y + sigma_y * kfwhm * np.sin(theta),
                    mode='lines', name='FWHM ellipse',
                    line=dict(color='#ffcc00', width=2.5, dash='dash')
                ))
                fig_ct.add_trace(go.Scatter(
                    x=[mu_x], y=[mu_y], mode='markers', name='Centre',
                    marker=dict(color='#ff00cc', size=12, symbol='cross')
                ))
                fig_ct.update_layout(
                    title="Contour 2D + Ellipse FWHM",
                    xaxis_title='x', yaxis_title='y',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=430,
                )
                st.plotly_chart(fig_ct, use_container_width=True)

    # ============================================================
    # TAB 3 : MULTI-GAUSSIENNES
    # ============================================================
    with tab3:
        st.markdown("### 🔄 Superposition de gaussiennes")
        col1, col2 = st.columns([1, 2])

        with col1:
            n_gauss = st.slider("Nombre de gaussiennes", 1, 6, 3)
            x_mg = np.linspace(-15, 15, 2000)
            gauss_params = []

            for i in range(n_gauss):
                st.markdown(f"**G{i+1}**")
                c1, c2_c, c3 = st.columns(3)
                with c1:
                    A_i = st.slider(f"A{i+1}", 0.1, 5.0, 1.0, 0.1, key=f"A{i}")
                with c2_c:
                    mu_i = st.slider(f"μ{i+1}", -10.0, 10.0, float((i-n_gauss//2)*3), 0.1, key=f"mu{i}")
                with c3:
                    s_i = st.slider(f"σ{i+1}", 0.1, 4.0, 1.0, 0.1, key=f"sig{i}")
                gauss_params.append((A_i, mu_i, s_i))

            show_total = st.checkbox("Afficher somme totale", True)
            show_indiv = st.checkbox("Afficher individuelles", True)

        with col2:
            colors_g = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88', '#ffcc00', '#ff4400']
            fig_mg = go.Figure()
            y_total = np.zeros_like(x_mg)

            for i, (Ai, mui, si) in enumerate(gauss_params):
                eng_i = GaussianEngine(Ai, si, mui)
                yi = eng_i.eval_1d(x_mg)
                y_total += yi
                if show_indiv:
                    fig_mg.add_trace(go.Scatter(
                        x=x_mg, y=yi, mode='lines',
                        name=f'G{i+1} (A={Ai}, μ={mui:.1f}, σ={si})',
                        line=dict(color=colors_g[i % len(colors_g)], width=2, dash='dot'),
                        opacity=0.7
                    ))

            if show_total:
                fig_mg.add_trace(go.Scatter(
                    x=x_mg, y=y_total, mode='lines', name='Somme',
                    line=dict(color='#ffffff', width=3),
                    fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'
                ))

            fig_mg.update_layout(
                title=f"Superposition de {n_gauss} gaussiennes",
                xaxis_title='x', yaxis_title='f(x)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=480,
            )
            st.plotly_chart(fig_mg, use_container_width=True)

            # Métriques globales
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Max total", f"{y_total.max():.4f}")
            with c2: st.metric("Aire totale", f"{np.trapz(y_total, x_mg):.4f}")
            with c3: st.metric("x (max)", f"{x_mg[np.argmax(y_total)]:.3f}")

    # ============================================================
    # TAB 4 : SPECTRE & CONVOLUTION
    # ============================================================
    with tab4:
        st.markdown("### 📡 Transformée de Fourier & Convolution")
        col1, col2 = st.columns([1, 2])

        with col1:
            amp_f  = st.slider("Amplitude A", 0.1, 5.0, 1.0, 0.1, key="af")
            sig_f  = st.slider("σ (gaussienne)", 0.1, 3.0, 1.0, 0.05, key="sf")
            mu_f   = st.slider("Centre μ", -5.0, 5.0, 0.0, 0.1, key="muf")
            sig2_f = st.slider("σ₂ (convolution)", 0.1, 3.0, 0.8, 0.05, key="sf2")

            eng_f = GaussianEngine(amp_f, sig_f, mu_f)
            st.metric("σ convolution √(σ₁²+σ₂²)",
                      f"{np.sqrt(sig_f**2 + sig2_f**2):.4f}")

        with col2:
            x_f = np.linspace(-15, 15, 4096)
            y_f = eng_f.eval_1d(x_f)

            # FFT numérique
            dt = x_f[1] - x_f[0]
            fft_num = np.fft.rfft(y_f)
            freqs_num = np.fft.rfftfreq(len(y_f), dt)
            mag_num = np.abs(fft_num) * dt

            # FFT analytique
            fft_ana = np.abs(eng_f.fft_analytique(freqs_num))

            # Convolution analytique
            eng_conv = eng_f.convolution(sig2_f)
            y_conv = eng_conv.eval_1d(x_f)

            fig_fft = make_subplots(rows=2, cols=1,
                subplot_titles=["Signal + Convolution", "Spectre |FFT(f)|"])

            fig_fft.add_trace(go.Scatter(
                x=x_f, y=y_f, mode='lines', name='f(x)',
                line=dict(color='#00ccff', width=2.5)
            ), row=1, col=1)
            fig_fft.add_trace(go.Scatter(
                x=x_f, y=y_conv, mode='lines', name=f'f*g (σ_conv={eng_conv.sigma:.2f})',
                line=dict(color='#ff00cc', width=2.5, dash='dash')
            ), row=1, col=1)

            fig_fft.add_trace(go.Scatter(
                x=freqs_num, y=mag_num, mode='lines', name='|FFT| numérique',
                line=dict(color='#7700ff', width=2),
                fill='tozeroy', fillcolor='rgba(119,0,255,0.15)'
            ), row=2, col=1)
            fig_fft.add_trace(go.Scatter(
                x=freqs_num, y=fft_ana, mode='lines', name='|FFT| analytique',
                line=dict(color='#ffcc00', width=2, dash='dash')
            ), row=2, col=1)

            fig_fft.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                height=540,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_fft.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_fft.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_fft.update_xaxes(title_text="Fréquence ξ", row=2, col=1,
                                  range=[0, min(3/sig_f, freqs_num[-1])])
            st.plotly_chart(fig_fft, use_container_width=True)

    # ============================================================
    # TAB 5 : AJUSTEMENT DONNÉES
    # ============================================================
    with tab5:
        st.markdown("### 🔬 Ajustement gaussien sur données expérimentales")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Saisir ou générer des données")
            mode_data = st.radio("Source", ["Données synthétiques", "Saisie manuelle"], horizontal=True)

            if mode_data == "Données synthétiques":
                A_true  = st.slider("A vrai", 0.5, 5.0, 2.0, 0.1)
                mu_true = st.slider("μ vrai", -5.0, 5.0, 1.0, 0.1)
                s_true  = st.slider("σ vrai", 0.1, 3.0, 1.0, 0.05)
                bruit   = st.slider("Bruit σ_bruit", 0.0, 1.0, 0.15, 0.01)
                n_pts   = st.slider("Points", 10, 200, 40)
                np.random.seed(42)
                x_data = np.linspace(mu_true - 4*s_true, mu_true + 4*s_true, n_pts)
                y_data = (A_true * np.exp(-0.5*((x_data-mu_true)/s_true)**2) +
                          np.random.normal(0, bruit, n_pts))
            else:
                x_str = st.text_input("x (virgule)", "-3,-2,-1,0,1,2,3")
                y_str = st.text_input("y (virgule)", "0.1,0.4,0.8,1.0,0.8,0.4,0.1")
                try:
                    x_data = np.array([float(v.strip()) for v in x_str.split(',')])
                    y_data = np.array([float(v.strip()) for v in y_str.split(',')])
                except:
                    st.error("Format invalide")
                    st.stop()

        with col2:
            eng_fit = GaussianEngine(1.0, 1.0, 0.0)
            fit = eng_fit.fit_data(x_data, y_data)

            if "erreur" not in fit:
                x_fine = np.linspace(x_data.min()-1, x_data.max()+1, 500)
                y_fit = fit["A"] * np.exp(-0.5 * ((x_fine - fit["μ"]) / fit["σ"])**2)

                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode='markers', name='Données',
                    marker=dict(color='#ff00cc', size=10, symbol='circle',
                               line=dict(width=2, color='#ffffff'))
                ))
                fig_fit.add_trace(go.Scatter(
                    x=x_fine, y=y_fit, mode='lines', name='Ajustement',
                    line=dict(color='#00ccff', width=3)
                ))
                fig_fit.update_layout(
                    title=f"Ajustement — R²={fit['R²']:.4f}",
                    xaxis_title='x', yaxis_title='y',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=400,
                )
                st.plotly_chart(fig_fit, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("A ajusté", f"{fit['A']:.4f} ± {fit['σ_A']:.4f}")
                with c2: st.metric("μ ajusté", f"{fit['μ']:.4f} ± {fit['σ_μ']:.4f}")
                with c3: st.metric("σ ajusté", f"{fit['σ']:.4f} ± {fit['σ_σ']:.4f}")
                with c4: st.metric("R²", f"{fit['R²']:.6f}")

                st.metric("FWHM ajusté", f"{fit['FWHM_fit']:.4f}")

                if mode_data == "Données synthétiques":
                    st.markdown("#### 🎯 Comparaison vrai vs ajusté")
                    comp = pd.DataFrame({
                        "Paramètre": ["A", "μ", "σ", "FWHM"],
                        "Vrai": [A_true, mu_true, s_true, 2.3548*s_true],
                        "Ajusté": [fit["A"], fit["μ"], fit["σ"], fit["FWHM_fit"]],
                        "Erreur (%)": [
                            abs(fit["A"]-A_true)/A_true*100,
                            abs(fit["μ"]-mu_true)/(abs(mu_true)+1e-10)*100,
                            abs(fit["σ"]-s_true)/s_true*100,
                            abs(fit["FWHM_fit"]-2.3548*s_true)/(2.3548*s_true)*100
                        ]
                    })
                    st.dataframe(comp.round(4), use_container_width=True)
            else:
                st.error(f"Ajustement échoué : {fit['erreur']}")

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire scientifique")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 🔬 Applications")
        for app, desc in APPLICATIONS.items():
            st.markdown(f"- **{app}** : {desc}")

        st.markdown("---")
        st.markdown("### ⚗️ Diagnostic")
        try:
            eng_diag = GaussianEngine(1.0, 1.0, 0.0)
            diag = eng_diag.diagnostiquer()
            st.dataframe(pd.DataFrame(diag), use_container_width=True)
        except:
            pass

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Abramowitz & Stegun — *Handbook of Mathematical Functions* (NIST, 1964)",
            "Goodman — *Statistical Optics* (Wiley, 2015)",
            "Papoulis — *Probability, Random Variables, and Stochastic Processes* (McGraw-Hill, 2002)",
        ]:
            st.markdown(f"- {r}")