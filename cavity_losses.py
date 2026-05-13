import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import fsolve, minimize_scalar
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
FORMULES = {
    "Loi de Beer-Lambert":      r"I(x) = I_0\,e^{-\alpha x}",
    "Pertes totales":           r"\alpha_{tot} = \alpha_{abs} + \alpha_{dif} + \alpha_{mir}",
    "Pertes miroirs":           r"\alpha_{mir} = -\frac{\ln(R_1 R_2)}{2L}",
    "Facteur Q":                r"Q = \frac{\omega_0 \tau_p}{1} = \frac{2\pi\nu_0}{\delta\nu}",
    "Finesse":                  r"\mathcal{F} = \frac{\pi\sqrt{R}}{1-R} \approx \frac{\pi}{T+\delta}",
    "Temps de vie photon":      r"\tau_p = \frac{n L}{c\,\alpha_{tot} L} = \frac{1}{c\,\alpha_{tot}}",
    "Transmittance":            r"T(\lambda) = e^{-\alpha(\lambda) L}",
    "Distance demi-atténuation":r"L_{1/2} = \frac{\ln 2}{\alpha}",
    "Distance 1/e":             r"L_{1/e} = \frac{1}{\alpha}",
    "Bilan puissance":          r"P_{out} = P_{in}(1-T_1)(1-T_2)\,e^{-2\alpha L}",
    "Seuil laser":              r"\alpha_{tot} = g_0 \Rightarrow g_{th} = \alpha_{int} + \alpha_{mir}",
}

MATERIAUX_OPTIQUES = {
    "Silice (SiO₂) — 1550 nm": {"alpha": 0.0003, "n": 1.444, "unite": "cm⁻¹"},
    "Nd:YAG — 1064 nm":         {"alpha": 0.002,  "n": 1.820, "unite": "cm⁻¹"},
    "ZnSe — CO₂ (10.6 μm)":    {"alpha": 0.001,  "n": 2.403, "unite": "cm⁻¹"},
    "Germanium — IR":           {"alpha": 0.05,   "n": 4.000, "unite": "cm⁻¹"},
    "Air ambiant":              {"alpha": 0.0001, "n": 1.000, "unite": "cm⁻¹"},
    "GaAs — 870 nm":            {"alpha": 0.5,    "n": 3.600, "unite": "cm⁻¹"},
    "Personnalisé":             {"alpha": 0.1,    "n": 1.5,   "unite": "cm⁻¹"},
}


# ============================================================
# MOTEUR PERTES DE CAVITÉ
# ============================================================
class CavityLossEngine:
    """Moteur d'analyse des pertes optiques et de cavité laser."""

    def __init__(self, I0: float, alpha: float):
        if alpha < 0:
            raise ValueError("α doit être ≥ 0")
        self.I0 = I0
        self.alpha = alpha

    # --- Propagation ---
    def intensite(self, x: np.ndarray) -> np.ndarray:
        return self.I0 * np.exp(-self.alpha * x)

    def transmittance(self, L: float) -> float:
        return np.exp(-self.alpha * L)

    def absorbance(self, L: float) -> float:
        return -np.log10(self.transmittance(L))

    # --- Distances caractéristiques ---
    @property
    def distance_1_e(self) -> float:
        return 1.0 / self.alpha if self.alpha > 0 else np.inf

    @property
    def distance_demi(self) -> float:
        return np.log(2) / self.alpha if self.alpha > 0 else np.inf

    def distance_pour_attenuation(self, ratio: float) -> float:
        """Distance pour I/I₀ = ratio."""
        if ratio <= 0 or ratio >= 1 or self.alpha <= 0:
            return np.inf
        return -np.log(ratio) / self.alpha

    # --- Cavité laser ---
    def pertes_miroirs(self, R1: float, R2: float, L: float) -> float:
        """Pertes miroirs en cm⁻¹."""
        if R1 <= 0 or R2 <= 0 or L <= 0:
            return 0
        return -np.log(R1 * R2) / (2 * L)

    def pertes_totales(self, R1: float, R2: float, L: float,
                       alpha_int: float = None) -> float:
        a = alpha_int or self.alpha
        return self.pertes_miroirs(R1, R2, L) + a

    def facteur_Q(self, nu0: float, R1: float, R2: float,
                  L: float, n: float = 1.5) -> float:
        """Facteur de qualité Q de la cavité."""
        c = 3e10  # cm/s
        alpha_tot = self.pertes_totales(R1, R2, L)
        tau_p = n * L / (c * alpha_tot * L) if alpha_tot > 0 else np.inf
        return 2 * np.pi * nu0 * tau_p

    def finesse(self, R1: float, R2: float, alpha_int: float,
                L: float) -> float:
        """Finesse effective de la cavité."""
        R_eff = np.sqrt(R1 * R2) * np.exp(-alpha_int * L)
        if R_eff >= 1:
            return np.inf
        return np.pi * np.sqrt(R_eff) / (1 - R_eff)

    def seuil_gain(self, R1: float, R2: float, L: float) -> float:
        """Gain seuil laser g_th = α_tot."""
        return self.pertes_totales(R1, R2, L)

    def optimiser_R2(self, R1: float, L: float,
                     g0: float, alpha_int: float) -> dict:
        """Optimise R2 pour maximiser la puissance de sortie."""
        def puissance_sortie(R2):
            alpha_mir = -np.log(R1 * R2) / (2 * L)
            alpha_tot = alpha_mir + alpha_int
            if g0 <= alpha_tot:
                return 0
            T2 = 1 - R2
            gain = g0 - alpha_int
            return T2 * (gain / alpha_tot - 1)

        R2_range = np.linspace(0.01, 0.999, 500)
        P_range = np.array([puissance_sortie(r) for r in R2_range])
        idx_opt = np.argmax(P_range)
        return {
            "R2_opt": R2_range[idx_opt],
            "P_max": P_range[idx_opt],
            "R2_range": R2_range,
            "P_range": P_range,
        }

    # --- Spectre ---
    def spectre_transmittance(self, lambdas: np.ndarray,
                               L: float, alpha_func=None) -> np.ndarray:
        """Transmittance spectrale T(λ)."""
        if alpha_func is not None:
            alphas = alpha_func(lambdas)
        else:
            alphas = np.full_like(lambdas, self.alpha)
        return np.exp(-alphas * L)

    def diagnostiquer(self, R1: float, R2: float, L: float) -> list:
        diag = []
        alpha_m = self.pertes_miroirs(R1, R2, L)
        alpha_tot = self.pertes_totales(R1, R2, L)

        diag.append({"Test": "Pertes internes α",
                     "Valeur": f"{self.alpha:.4f} cm⁻¹",
                     "Statut": "✅ OK" if self.alpha < 0.1 else "⚠️ Élevées",
                     "Note": "< 0.1 cm⁻¹ recommandé"})
        diag.append({"Test": "Pertes miroirs",
                     "Valeur": f"{alpha_m:.4f} cm⁻¹",
                     "Statut": "✅ OK",
                     "Note": f"-ln(R1·R2)/(2L)"})
        diag.append({"Test": "Pertes totales",
                     "Valeur": f"{alpha_tot:.4f} cm⁻¹",
                     "Statut": "✅ OK" if alpha_tot < 1 else "⚠️ Élevées",
                     "Note": "Seuil laser = g_th"})
        diag.append({"Test": "Finesse",
                     "Valeur": f"{self.finesse(R1, R2, self.alpha, L):.1f}",
                     "Statut": "✅ OK",
                     "Note": "π√R/(1-R)"})
        return diag


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def cavity_page():
    st.markdown("## 🔲 Pertes de Cavité Optique Avancées")
    st.markdown("*Beer-Lambert, cavité laser, finesse, optimisation, spectre*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📉 Propagation",
        "⚖️ Cavité Laser",
        "🌡️ Carte de pertes",
        "🎯 Optimisation R₂",
        "⚗️ Diagnostic",
        "📖 Théorie"
    ])

    # Config partagée
    with st.sidebar.expander("⚙️ Matériau & paramètres", expanded=True):
        mat = st.selectbox("Matériau optique", list(MATERIAUX_OPTIQUES.keys()))
        mat_info = MATERIAUX_OPTIQUES[mat]

        if mat == "Personnalisé":
            alpha_base = st.slider("α interne (cm⁻¹)", 0.001, 2.0, 0.1, 0.001)
            n_ref = st.slider("Indice n", 1.0, 5.0, 1.5, 0.01)
        else:
            alpha_base = mat_info["alpha"]
            n_ref = mat_info["n"]
            st.info(f"α = {alpha_base} cm⁻¹ | n = {n_ref}")

        I0 = st.slider("Intensité initiale I₀", 0.1, 200.0, 10.0, 0.5)
        L_max = st.slider("Distance max (cm)", 1.0, 200.0, 50.0, 1.0)

    engine = CavityLossEngine(I0, alpha_base)

    # ============================================================
    # TAB 1 : PROPAGATION
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")
            alpha_var = st.slider("α (cm⁻¹)", 0.001, 2.0, alpha_base, 0.001)
            eng1 = CavityLossEngine(I0, alpha_var)

            mode_y = st.radio("Échelle", ["Linéaire", "Logarithmique"], horizontal=True)
            show_markers = st.checkbox("Marqueurs caractéristiques", True)

            st.markdown("### 📐 Distances caractéristiques")
            st.metric("Distance 1/e (cm)", f"{eng1.distance_1_e:.3f}")
            st.metric("Distance 1/2 (cm)", f"{eng1.distance_demi:.3f}")
            st.metric("I à L_max", f"{eng1.intensite(np.array([L_max]))[0]:.4f}")
            perte_pct = (1 - eng1.transmittance(L_max)) * 100
            st.metric("Perte totale (%)", f"{perte_pct:.2f}")
            st.metric("Transmittance", f"{eng1.transmittance(L_max):.4f}")
            st.metric("Absorbance", f"{eng1.absorbance(L_max):.4f}")

        with col2:
            x = np.linspace(0, L_max, 2000)
            I = eng1.intensite(x)

            fig = go.Figure()

            if mode_y == "Logarithmique":
                y_plot = np.log10(np.maximum(I, 1e-12))
                y_theo = np.log10(I0) - alpha_var * x / np.log(10)
                fig.add_trace(go.Scatter(
                    x=x, y=y_theo, mode='lines', name='Droite théorique',
                    line=dict(color='#ffcc00', width=2, dash='dash')
                ))
            else:
                y_plot = I

            fig.add_trace(go.Scatter(
                x=x, y=y_plot, mode='lines',
                name=f'I(x) α={alpha_var:.3f}',
                line=dict(color='#00ccff', width=3),
                fill='tozeroy' if mode_y == "Linéaire" else 'none',
                fillcolor='rgba(0,204,255,0.1)'
            ))

            if show_markers and mode_y == "Linéaire":
                # 1/e
                L_1e = eng1.distance_1_e
                if L_1e < L_max:
                    fig.add_vline(x=L_1e, line_color='#00ff88', line_dash='dash',
                                  annotation_text=f"L_1/e={L_1e:.2f}cm")
                    fig.add_hline(y=I0/np.e, line_color='rgba(0,255,136,0.4)',
                                  line_dash='dot', annotation_text="I₀/e")
                # 1/2
                L_12 = eng1.distance_demi
                if L_12 < L_max:
                    fig.add_vline(x=L_12, line_color='#7700ff', line_dash='dash',
                                  annotation_text=f"L_1/2={L_12:.2f}cm")
                    fig.add_hline(y=I0/2, line_color='rgba(119,0,255,0.4)',
                                  line_dash='dot', annotation_text="I₀/2")

            fig.update_layout(
                title=f"Propagation Beer-Lambert — {mat}",
                xaxis_title="Distance x (cm)",
                yaxis_title="Intensité I(x)" if mode_y == "Linéaire" else "log₁₀(I(x))",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Comparaison multi-alpha
            st.markdown("#### 📊 Comparaison de coefficients α")
            alphas_comp = st.multiselect("Coefficients à comparer",
                [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                default=[0.05, 0.1, 0.2])

            if alphas_comp:
                colors_c = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00','#ff4400','#ffffff']
                fig_cmp = go.Figure()
                for i, ac in enumerate(sorted(alphas_comp)):
                    ec = CavityLossEngine(I0, ac)
                    Ic = ec.intensite(x)
                    y_c = Ic if mode_y == "Linéaire" else np.log10(np.maximum(Ic, 1e-12))
                    fig_cmp.add_trace(go.Scatter(
                        x=x, y=y_c, mode='lines', name=f'α={ac}',
                        line=dict(color=colors_c[i % len(colors_c)], width=2)
                    ))
                fig_cmp.update_layout(
                    title="Comparaison multi-α",
                    xaxis_title="Distance x (cm)",
                    yaxis_title="I(x)" if mode_y == "Linéaire" else "log₁₀(I)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=360,
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            df_exp = pd.DataFrame({"x_cm": x, "I": I, "T": I/I0})
            st.download_button("💾 Export CSV",
                               df_exp.to_csv(index=False).encode(),
                               "cavity_losses.csv", "text/csv")

    # ============================================================
    # TAB 2 : CAVITÉ LASER
    # ============================================================
    with tab2:
        st.markdown("### ⚖️ Analyse de cavité laser")
        col1, col2 = st.columns([1, 2])

        with col1:
            R1 = st.slider("Réflectivité R₁ (%)", 50.0, 99.9, 99.5, 0.1) / 100
            R2 = st.slider("Réflectivité R₂ (%)", 10.0, 99.9, 70.0, 0.1) / 100
            L_cav = st.slider("Longueur cavité L (cm)", 1.0, 100.0, 20.0, 0.5)
            alpha_int = st.slider("Pertes internes α_int (cm⁻¹)", 0.001, 0.5, alpha_base, 0.001)
            nu0 = st.slider("Fréquence ν₀ (THz)", 100.0, 600.0, 282.0, 1.0)

            eng_cav = CavityLossEngine(I0, alpha_int)

            alpha_mir = eng_cav.pertes_miroirs(R1, R2, L_cav)
            alpha_tot = eng_cav.pertes_totales(R1, R2, L_cav)
            fin = eng_cav.finesse(R1, R2, alpha_int, L_cav)

            st.markdown("### 📐 Résultats")
            st.metric("Pertes miroirs (cm⁻¹)", f"{alpha_mir:.4f}")
            st.metric("Pertes totales (cm⁻¹)", f"{alpha_tot:.4f}")
            st.metric("Finesse", f"{fin:.1f}")
            st.metric("Gain seuil g_th (cm⁻¹)", f"{alpha_tot:.4f}")

            c = 3e10
            tau_p = n_ref * L_cav / (c * alpha_tot * L_cav)
            st.metric("Temps vie photon τ_p (ns)", f"{tau_p*1e9:.3f}")
            delta_nu = 1 / (2 * np.pi * tau_p)
            st.metric("Largeur de raie Δν (MHz)", f"{delta_nu/1e6:.2f}")

        with col2:
            # Profil R1 et R2 vs pertes
            R1_arr = np.linspace(0.5, 0.999, 200)
            R2_arr = np.linspace(0.5, 0.999, 200)
            R1g, R2g = np.meshgrid(R1_arr, R2_arr)
            alpha_m_map = -np.log(R1g * R2g) / (2 * L_cav)
            alpha_tot_map = alpha_m_map + alpha_int

            fig_cav = go.Figure(data=go.Heatmap(
                z=alpha_tot_map,
                x=R1_arr * 100, y=R2_arr * 100,
                colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                colorbar=dict(title='α_tot (cm⁻¹)', tickfont=dict(color='#c0d0ff'))
            ))
            fig_cav.add_trace(go.Scatter(
                x=[R1*100], y=[R2*100], mode='markers', name='Opération',
                marker=dict(color='#ff0000', size=14, symbol='star',
                           line=dict(width=2, color='#ffffff'))
            ))
            fig_cav.update_layout(
                title="Pertes totales α_tot(R₁, R₂)",
                xaxis_title="R₁ (%)", yaxis_title="R₂ (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'),
                yaxis=dict(color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=430,
            )
            st.plotly_chart(fig_cav, use_container_width=True)

            # Finesse vs R
            R_arr = np.linspace(0.1, 0.999, 500)
            fin_arr = np.pi * np.sqrt(R_arr) / (1 - R_arr)
            fig_fin = go.Figure()
            fig_fin.add_trace(go.Scatter(
                x=R_arr*100, y=fin_arr, mode='lines', name='Finesse',
                line=dict(color='#00ccff', width=3)
            ))
            fig_fin.add_vline(x=R2*100, line_color='#ff00cc', line_dash='dash',
                              annotation_text=f"R₂={R2*100:.1f}%")
            fig_fin.update_layout(
                title="Finesse vs Réflectivité R",
                xaxis_title="R (%)", yaxis_title="Finesse",
                yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                height=320,
            )
            st.plotly_chart(fig_fin, use_container_width=True)

    # ============================================================
    # TAB 3 : CARTE DE PERTES
    # ============================================================
    with tab3:
        st.markdown("### 🌡️ Cartographie des pertes")
        col1, col2 = st.columns([1, 2])

        with col1:
            L_carte = st.slider("Longueur cavité L (cm)", 1.0, 100.0, 20.0, key="Lc")
            alpha_c = st.slider("α_int (cm⁻¹)", 0.001, 0.5, alpha_base, 0.001, key="ac")
            vue = st.radio("Vue", ["R₁ × R₂", "α × L"], horizontal=True)

        with col2:
            if vue == "R₁ × R₂":
                r1a = np.linspace(0.5, 0.999, 60)
                r2a = np.linspace(0.5, 0.999, 60)
                Rg1, Rg2 = np.meshgrid(r1a, r2a)
                Z = -np.log(Rg1 * Rg2) / (2 * L_carte) + alpha_c

                fig_c = go.Figure(data=[go.Surface(
                    z=Z, x=r1a*100, y=r2a*100,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                )])
                fig_c.update_layout(
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff', title='R₁ (%)'),
                        yaxis=dict(color='#c0d0ff', title='R₂ (%)'),
                        zaxis=dict(color='#c0d0ff', title='α_tot (cm⁻¹)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c0d0ff'),
                    height=480, margin=dict(l=0,r=0,t=40,b=0)
                )
            else:
                al_arr = np.linspace(0.001, 1.0, 60)
                Lar = np.linspace(1, 100, 60)
                Ag, Lg = np.meshgrid(al_arr, Lar)
                T_map = np.exp(-Ag * Lg)

                fig_c = go.Figure(data=[go.Surface(
                    z=T_map, x=al_arr, y=Lar,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                )])
                fig_c.update_layout(
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff', title='α (cm⁻¹)'),
                        yaxis=dict(color='#c0d0ff', title='L (cm)'),
                        zaxis=dict(color='#c0d0ff', title='Transmittance T'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c0d0ff'),
                    height=480, margin=dict(l=0,r=0,t=40,b=0)
                )
            st.plotly_chart(fig_c, use_container_width=True)

    # ============================================================
    # TAB 4 : OPTIMISATION R₂
    # ============================================================
    with tab4:
        st.markdown("### 🎯 Optimisation de R₂ pour puissance max")
        col1, col2 = st.columns([1, 2])

        with col1:
            R1_opt = st.slider("R₁ (%)", 50.0, 99.9, 99.0, 0.1, key="R1o") / 100
            L_opt  = st.slider("L (cm)", 1.0, 100.0, 20.0, 0.5, key="Lo")
            g0_opt = st.slider("Gain g₀ (cm⁻¹)", 0.01, 2.0, 0.5, 0.01)
            ai_opt = st.slider("α_int (cm⁻¹)", 0.001, 0.5, alpha_base, 0.001, key="aio")

            eng_opt = CavityLossEngine(I0, ai_opt)
            res_opt = eng_opt.optimiser_R2(R1_opt, L_opt, g0_opt, ai_opt)

            st.metric("R₂ optimal (%)", f"{res_opt['R2_opt']*100:.2f}")
            st.metric("Puissance max (u.a.)", f"{res_opt['P_max']:.4f}")
            st.metric("g₀/α_tot (seuil)",
                      f"{g0_opt / eng_opt.pertes_totales(R1_opt, res_opt['R2_opt'], L_opt):.3f}")

        with col2:
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=res_opt['R2_range']*100, y=res_opt['P_range'],
                mode='lines', name='P_out(R₂)',
                line=dict(color='#00ccff', width=3)
            ))
            fig_opt.add_vline(x=res_opt['R2_opt']*100,
                              line_color='#ffcc00', line_dash='dash',
                              annotation_text=f"R₂*={res_opt['R2_opt']*100:.1f}%")
            fig_opt.add_trace(go.Scatter(
                x=[res_opt['R2_opt']*100], y=[res_opt['P_max']],
                mode='markers', name='Optimum',
                marker=dict(color='#00ff88', size=14, symbol='star')
            ))
            fig_opt.update_layout(
                title="Puissance de sortie vs R₂",
                xaxis_title="R₂ (%)", yaxis_title="P_out (u.a.)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=430,
            )
            st.plotly_chart(fig_opt, use_container_width=True)

    # ============================================================
    # TAB 5 : DIAGNOSTIC
    # ============================================================
    with tab5:
        st.markdown("### ⚗️ Diagnostic automatique")

        R1_d = st.slider("R₁ (%)", 50.0, 99.9, 99.0, 0.1, key="R1d") / 100
        R2_d = st.slider("R₂ (%)", 10.0, 99.9, 70.0, 0.1, key="R2d") / 100
        L_d  = st.slider("L (cm)", 1.0, 100.0, 20.0, 0.5, key="Ld")

        eng_d = CavityLossEngine(I0, alpha_base)
        diag = eng_d.diagnostiquer(R1_d, R2_d, L_d)
        st.dataframe(pd.DataFrame(diag), use_container_width=True)

        st.markdown("#### 📋 Tableau d'erreurs optiques")
        erreurs = {
            "Problème": ["α trop élevé", "Finesse faible", "R₂ non optimisé",
                         "Instabilité transverse", "Pertes diffusion"],
            "Cause": ["Absorption matériau", "Réflectivité < 50%",
                      "R₂ ≠ R₂_opt", "Rayon de courbure", "Rugosité surface"],
            "Symptôme": ["I décroît vite", "Mauvaise sélectivité",
                         "Puissance sous-optimale", "Faisceau déformé", "Pertes excess"],
            "Solution": ["Changer matériau", "Miroirs HR", "Optimiser R₂",
                         "Ajuster géométrie", "Polir les miroirs"]
        }
        st.dataframe(pd.DataFrame(erreurs), use_container_width=True)

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
        st.markdown("### 🔬 Matériaux optiques")
        df_mat = pd.DataFrame([
            {"Matériau": k, "α (cm⁻¹)": v["alpha"], "n": v["n"]}
            for k, v in MATERIAUX_OPTIQUES.items() if k != "Personnalisé"
        ])
        st.dataframe(df_mat, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Saleh & Teich — *Fundamentals of Photonics* (Wiley, 2007)",
            "Svelto — *Principles of Lasers* (Springer, 2010)",
            "Yariv — *Optical Electronics in Modern Communications* (Oxford, 2006)",
        ]:
            st.markdown(f"- {r}")