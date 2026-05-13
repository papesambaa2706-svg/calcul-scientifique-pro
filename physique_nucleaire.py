import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

CONSTANTES = {
    "u (kg)":           1.66054e-27,
    "m_p (kg)":         1.67262e-27,
    "m_n (kg)":         1.67493e-27,
    "m_e (kg)":         9.10938e-31,
    "e (C)":            1.60218e-19,
    "ħ (J·s)":          1.05457e-34,
    "c (m/s)":          2.99792e8,
    "k_B (J/K)":        1.38065e-23,
    "N_A":              6.02214e23,
    "r₀ (fm)":          1.2,
    "1 MeV (J)":        1.60218e-13,
    "1 u (MeV/c²)":     931.494,
    "1 Curie (Bq)":     3.7e10,
}

FORMULES = {
    "Radioactivité":          r"N(t)=N_0\,e^{-\lambda t},\quad A(t)=A_0\,e^{-\lambda t}",
    "Période & constante":    r"T_{1/2}=\frac{\ln 2}{\lambda},\quad\lambda=\frac{\ln 2}{T_{1/2}}",
    "Énergie liaison":        r"B(A,Z)=\left[Z m_p+(A-Z)m_n-m(A,Z)\right]c^2",
    "Formule Bethe-Weizsäcker":r"B=a_v A-a_s A^{2/3}-a_c\frac{Z^2}{A^{1/3}}-a_{sym}\frac{(A-2Z)^2}{A}+\delta",
    "Énergie défaut masse":   r"E=\Delta m\cdot c^2",
    "Rayon nucléaire":        r"R=r_0 A^{1/3},\quad r_0\approx1.2\text{ fm}",
    "Réaction nucléaire Q":   r"Q=(m_{in}-m_{out})c^2",
    "Section efficace":       r"\sigma=\frac{\text{réactions/noyau/s}}{\text{flux }\phi}",
    "Fission (Q≈200 MeV)":   r"^{235}_{92}\text{U}+n\rightarrow\text{fragments}+\bar{\nu}_e+\gamma",
    "Fusion D-T":             r"^2_1\text{H}+^3_1\text{H}\rightarrow^4_2\text{He}+n+17.6\text{MeV}",
    "Loi Geiger-Nuttall":     r"\log\lambda = a\log E_\alpha+b\quad(\text{désintégration}\,\alpha)",
    "Loi de Bateman":         r"\frac{dN_i}{dt}=\lambda_{i-1}N_{i-1}-\lambda_i N_i",
}

NOYAUX_CONNUS = {
    "¹H":   {"Z":1,"A":1,"masse_u":1.007825,"stable":True},
    "²H":   {"Z":1,"A":2,"masse_u":2.014102,"stable":True},
    "³H":   {"Z":1,"A":3,"masse_u":3.016049,"stable":False,"T12_s":3.888e8},
    "⁴He":  {"Z":2,"A":4,"masse_u":4.002602,"stable":True},
    "¹²C":  {"Z":6,"A":12,"masse_u":11.996709,"stable":True},
    "¹⁴C":  {"Z":6,"A":14,"masse_u":14.003242,"stable":False,"T12_s":1.81e11},
    "²³⁵U": {"Z":92,"A":235,"masse_u":235.043930,"stable":False,"T12_s":2.22e16},
    "²³⁸U": {"Z":92,"A":238,"masse_u":238.050788,"stable":False,"T12_s":1.41e17},
    "²³⁹Pu":{"Z":94,"A":239,"masse_u":239.052163,"stable":False,"T12_s":7.61e11},
    "⁶⁰Co": {"Z":27,"A":60,"masse_u":59.933817,"stable":False,"T12_s":1.663e8},
    "¹³⁷Cs":{"Z":55,"A":137,"masse_u":136.907090,"stable":False,"T12_s":9.457e8},
    "²²⁶Ra":{"Z":88,"A":226,"masse_u":226.025410,"stable":False,"T12_s":5.057e10},
}


# ============================================================
# MOTEUR PHYSIQUE NUCLÉAIRE
# ============================================================
class NuclearEngine:
    """Moteur de calcul en physique nucléaire."""

    def __init__(self):
        self.c2 = 931.494  # MeV/u
        self.mp = 1.007276  # u
        self.mn = 1.008665  # u
        self.me = 0.000549  # u

    # --- Radioactivité ---
    def constante_decroissance(self, T12: float) -> float:
        return np.log(2) / T12

    def activite(self, N0: float, lam: float) -> float:
        return lam * N0

    @st.cache_data
    def N_t(_self, N0: float, lam: float, t: np.ndarray) -> np.ndarray:
        return N0 * np.exp(-lam * t)

    @st.cache_data
    def A_t(_self, A0: float, lam: float, t: np.ndarray) -> np.ndarray:
        return A0 * np.exp(-lam * t)

    def temps_pour_fraction(self, fraction: float, lam: float) -> float:
        """Temps pour que N(t)/N₀ = fraction."""
        return -np.log(fraction) / lam if lam > 0 else np.inf

    def periode_effective(self, T_phys: float, T_bio: float) -> float:
        """Période effective en médecine nucléaire."""
        return T_phys * T_bio / (T_phys + T_bio)

    # --- Énergie de liaison ---
    @st.cache_data
    def bethe_weizsacker(_self, A: int, Z: int) -> float:
        """Formule de Bethe-Weizsäcker (MeV)."""
        av = 15.85
        as_ = 18.34
        ac = 0.711
        asym = 23.23
        N = A - Z

        if A == 0:
            return 0
        terme_vol = av * A
        terme_surf = -as_ * A**(2/3)
        terme_coul = -ac * Z**2 / A**(1/3)
        terme_asym = -asym * (N - Z)**2 / A

        # Terme d'appariement
        if A % 2 == 1:
            delta = 0
        elif Z % 2 == 0:
            delta = +11.2 / A**0.5
        else:
            delta = -11.2 / A**0.5

        return terme_vol + terme_surf + terme_coul + terme_asym + delta

    def energie_liaison_nucleon(self, A: int, Z: int) -> float:
        return self.bethe_weizsacker(A, Z) / A if A > 0 else 0

    def rayon_nucleaire(self, A: int) -> float:
        """Rayon en fm."""
        return 1.2 * A**(1/3)

    def energie_Q(self, masses_in: list, masses_out: list) -> float:
        """Énergie Q d'une réaction (MeV)."""
        delta_m = sum(masses_in) - sum(masses_out)
        return delta_m * self.c2

    # --- Carte des noyaux ---
    def vallee_stabilite(self, A_range: np.ndarray) -> np.ndarray:
        """Ligne de stabilité Z_stable(A)."""
        return A_range / (1.98 + 0.0155 * A_range**(2/3))

    # --- Décroissance en chaîne (Bateman) ---
    @st.cache_data
    def chaine_bateman(_self, N0_arr: list, lambda_arr: list,
                       t: np.ndarray) -> np.ndarray:
        """Résolution des équations de Bateman pour une chaîne."""
        from scipy.integrate import odeint
        n = len(N0_arr)
        def dydt(y, t):
            dy = np.zeros(n)
            for i in range(n):
                if i > 0:
                    dy[i] += lambda_arr[i-1] * y[i-1]
                dy[i] -= lambda_arr[i] * y[i]
            return dy
        sol = odeint(dydt, N0_arr, t, rtol=1e-8)
        return sol

    # --- Section efficace ---
    def section_efficace_geometrique(self, A: int) -> float:
        """Section efficace géométrique en barns."""
        R = self.rayon_nucleaire(A) * 1e-15
        return np.pi * R**2 * 1e28

    def taux_reaction(self, phi: float, n: float, sigma: float) -> float:
        """Taux de réaction R = φ·n·σ."""
        return phi * n * sigma

    # --- Fission ---
    def energie_fission_U235(self) -> dict:
        """Bilan énergétique fission ²³⁵U."""
        return {
            "Fragments (MeV)": 167,
            "Neutrons (MeV)": 5,
            "γ prompts (MeV)": 7,
            "β + γ différés (MeV)": 13,
            "Neutrinos (MeV)": 10,
            "Total (MeV)": 202,
            "Neutrons émis <ν>": 2.43,
        }

    # --- Fusion ---
    def energie_fusion_DT(self) -> dict:
        """Bilan D+T → ⁴He + n."""
        m_D = 2.014102
        m_T = 3.016049
        m_He4 = 4.002602
        m_n = 1.008665
        Q = (m_D + m_T - m_He4 - m_n) * self.c2
        return {"Q (MeV)": Q, "E_He4 (MeV)": 3.52, "E_n (MeV)": 14.1,
                "Critère Lawson nτ (m⁻³s)": 1e20}

    # --- Dosimétrie ---
    def dose_absorbee(self, E_dep: float, masse: float) -> float:
        """Dose absorbée en Gray."""
        return E_dep / masse

    def dose_equivalente(self, dose_Gy: float, facteur_Q: float) -> float:
        """Dose équivalente en Sievert."""
        return dose_Gy * facteur_Q

    def dose_exponentielle(self, D0: float, mu: float,
                            x: np.ndarray) -> np.ndarray:
        """Atténuation de rayonnement γ."""
        return D0 * np.exp(-mu * x)

    def couche_demi_valeur(self, mu: float) -> float:
        """CDV = ln(2)/μ."""
        return np.log(2) / mu if mu > 0 else np.inf

    # --- Diagnostics ---
    def diagnostiquer_noyau(self, A: int, Z: int) -> list:
        N = A - Z
        diag = []
        B = self.bethe_weizsacker(A, Z)
        B_A = B / A if A > 0 else 0

        diag.append({"Test": "Énergie de liaison",
                     "Valeur": f"{B:.2f} MeV",
                     "Statut": "✅ Stable" if B > 0 else "❌ Instable",
                     "Note": "B > 0 nécessaire"})
        diag.append({"Test": "Énergie/nucléon",
                     "Valeur": f"{B_A:.3f} MeV",
                     "Statut": "✅ Optimal" if 7 < B_A < 9 else "ℹ️ Hors max",
                     "Note": "Max ~8.8 MeV/A pour Fe-56"})
        diag.append({"Test": "Rapport N/Z",
                     "Valeur": f"{N/Z:.3f}" if Z > 0 else "∞",
                     "Statut": "✅ Stable" if 1 <= N/Z <= 1.6 else "⚠️ Extrême",
                     "Note": "1 ≤ N/Z ≤ 1.6 pour noyaux stables"})
        diag.append({"Test": "Rayon (fm)",
                     "Valeur": f"{self.rayon_nucleaire(A):.3f}",
                     "Statut": "✅", "Note": "R = 1.2·A^(1/3) fm"})
        return diag


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def physique_nucleaire_page():
    st.markdown("## ☢️ Physique Nucléaire Avancée")
    st.markdown("*Radioactivité, énergie de liaison, fission, fusion, dosimétrie*")
    st.markdown("---")

    engine = NuclearEngine()
    colors_n = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00','#ff4400']

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "☢️ Radioactivité",
        "⚡ Énergie de liaison",
        "💥 Fission & Fusion",
        "🔬 Dosimétrie",
        "🗺️ Carte des noyaux",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : RADIOACTIVITÉ
    # ============================================================
    with tab1:
        st.markdown("### ☢️ Loi de radioactivité")
        col1, col2 = st.columns([1, 2])

        with col1:
            noyau_sel = st.selectbox("Noyau", [k for k,v in NOYAUX_CONNUS.items()
                                               if not v["stable"]])
            noy = NOYAUX_CONNUS[noyau_sel]
            T12_s = noy["T12_s"]
            T12_ans = T12_s / (3.156e7)

            N0 = st.slider("N₀ (×10¹²)", 0.01, 1000.0, 100.0, 0.1) * 1e12
            t_max_T = st.slider("Durée (× T½)", 1, 10, 5)

            lam = engine.constante_decroissance(T12_s)
            A0 = engine.activite(N0, lam)
            t_arr = np.linspace(0, t_max_T * T12_s, 500)  # Réduit de 1000 à 500 points
            N_arr = engine.N_t(N0, lam, t_arr)
            A_arr = engine.A_t(A0, lam, t_arr)

            st.metric("T½", f"{T12_ans:.3e} ans")
            st.metric("λ (s⁻¹)", f"{lam:.4e}")
            st.metric("A₀ (Bq)", f"{A0:.3e}")
            st.metric("A₀ (Ci)", f"{A0/3.7e10:.3e}")

        with col2:
            fig_rad = make_subplots(rows=2, cols=1,
                subplot_titles=["N(t) — Nombre de noyaux",
                                 "A(t) — Activité (Bq)"])

            fig_rad.add_trace(go.Scatter(
                x=t_arr/T12_s, y=N_arr/N0, mode='lines', name='N(t)/N₀',
                line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
            ), row=1, col=1)
            fig_rad.add_hline(y=0.5, line_color='#ffcc00', line_dash='dash',
                              annotation_text="N₀/2", row=1, col=1)

            fig_rad.add_trace(go.Scatter(
                x=t_arr/T12_s, y=A_arr, mode='lines', name='A(t)',
                line=dict(color='#7700ff', width=3)
            ), row=2, col=1)

            fig_rad.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=450,  # Réduit de 520 à 450
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                showlegend=False  # Désactive la légende pour plus de rapidité
            )
            fig_rad.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                  title_text="t / T½")
            fig_rad.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_rad, use_container_width=True)

        # Chaîne de désintégration (Bateman)
        st.markdown("### ⛓️ Chaîne de désintégration (Bateman)")
        col1b, col2b = st.columns([1, 2])
        with col1b:
            T12_1 = st.slider("T½₁ (jours)", 0.1, 365.0, 10.0, 0.1) * 86400
            T12_2 = st.slider("T½₂ (jours)", 0.1, 365.0, 2.0, 0.1) * 86400
            T12_3 = st.slider("T½₃ (jours)", 0.1, 365.0, 50.0, 0.1) * 86400
            N0_chain = [1e12, 0, 0]

        with col2b:
            lams = [engine.constante_decroissance(T) for T in [T12_1, T12_2, T12_3]]
            t_chain = np.linspace(0, 5*max(T12_1, T12_2, T12_3), 1000)
            sol_chain = engine.chaine_bateman(N0_chain, lams, t_chain)

            fig_chain = go.Figure()
            for i, (lab, col) in enumerate(zip(
                ["N₁ (parent)", "N₂ (fils)", "N₃ (petit-fils)"],
                ['#00ccff','#7700ff','#ff00cc']
            )):
                fig_chain.add_trace(go.Scatter(
                    x=t_chain/86400, y=sol_chain[:, i]/N0_chain[0],
                    mode='lines', name=lab, line=dict(color=col, width=2.5)
                ))
            fig_chain.update_layout(
                title="Équilibre radioactif (Bateman)",
                xaxis_title="t (jours)", yaxis_title="N_i/N₀",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
            )
            st.plotly_chart(fig_chain, use_container_width=True)

    # ============================================================
    # TAB 2 : ÉNERGIE DE LIAISON
    # ============================================================
    with tab2:
        st.markdown("### ⚡ Formule de Bethe-Weizsäcker")
        col1, col2 = st.columns([1, 2])

        with col1:
            A_bw = st.slider("Nombre de masse A", 1, 250, 56)
            Z_bw = st.slider("Nombre de protons Z", 1, min(A_bw, 120), max(1, A_bw//2))

            B = engine.bethe_weizsacker(A_bw, Z_bw)
            B_A = engine.energie_liaison_nucleon(A_bw, Z_bw)
            R = engine.rayon_nucleaire(A_bw)

            st.metric("B (MeV)", f"{B:.3f}")
            st.metric("B/A (MeV)", f"{B_A:.4f}")
            st.metric("R (fm)", f"{R:.3f}")

            # Termes individuels
            N = A_bw - Z_bw
            av, as_, ac, asym = 15.85, 18.34, 0.711, 23.23
            st.markdown("**Termes Bethe-Weizsäcker :**")
            termes = {
                "Volume av·A":    av * A_bw,
                "Surface -as·A²/³": -as_ * A_bw**(2/3),
                "Coulomb -ac·Z²/A¹/³": -ac * Z_bw**2 / A_bw**(1/3),
                "Asymétrie": -asym*(N-Z_bw)**2/A_bw,
            }
            for k, v in termes.items():
                st.metric(k, f"{v:.2f} MeV")

            diag = engine.diagnostiquer_noyau(A_bw, Z_bw)
            st.markdown("### ⚗️ Diagnostic")
            st.dataframe(pd.DataFrame(diag), use_container_width=True)

        with col2:
            A_arr = np.arange(2, 251)
            B_A_arr = [engine.energie_liaison_nucleon(A, max(1, int(engine.vallee_stabilite(np.array([A]))[0])))
                       for A in A_arr]

            fig_BA = go.Figure()
            fig_BA.add_trace(go.Scatter(x=A_arr, y=B_A_arr, mode='lines',
                name='B/A (MeV)', line=dict(color='#00ccff', width=3)))

            # Noyaux remarquables
            noyaux_rmq = {
                "⁴He":  (4,  7.07),
                "¹²C":  (12, 7.68),
                "⁵⁶Fe": (56, 8.79),
                "²³⁵U": (235,7.59),
            }
            for nom, (A, B) in noyaux_rmq.items():
                fig_BA.add_trace(go.Scatter(x=[A], y=[B], mode='markers+text',
                    text=[nom], textposition='top center',
                    marker=dict(color='#ff00cc', size=10),
                    name=nom, showlegend=False))

            fig_BA.add_hline(y=8.8, line_color='#ffcc00', line_dash='dash',
                             annotation_text="Max B/A ≈ 8.8 (⁵⁶Fe)")
            fig_BA.update_layout(
                title="Énergie de liaison par nucléon B/A vs A",
                xaxis_title="A (nombre de masse)", yaxis_title="B/A (MeV)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420,
            )
            st.plotly_chart(fig_BA, use_container_width=True)

            # Termes BW
            termes_arr = {
                "Volume": av * A_arr,
                "Surface": as_ * A_arr**(2/3),
                "Coulomb": ac * np.array([int(engine.vallee_stabilite(np.array([a]))[0])**2
                                          /a**(1/3) for a in A_arr]),
            }
            fig_termes = go.Figure()
            for i, (nom, vals) in enumerate(termes_arr.items()):
                fig_termes.add_trace(go.Scatter(x=A_arr, y=vals/A_arr, mode='lines',
                    name=nom, line=dict(color=colors_n[i], width=2)))
            fig_termes.update_layout(
                title="Contributions/A de Bethe-Weizsäcker",
                xaxis_title="A", yaxis_title="Contribution (MeV/A)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
            )
            st.plotly_chart(fig_termes, use_container_width=True)

    # ============================================================
    # TAB 3 : FISSION & FUSION
    # ============================================================
    with tab3:
        st.markdown("### 💥 Fission & Fusion nucléaire")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💥 Fission ²³⁵U")
            bilan_fission = engine.energie_fission_U235()
            for k, v in bilan_fission.items():
                st.metric(k, str(v))

            st.markdown("#### 🔢 Facteur de multiplication k_eff")
            nu = st.slider("ν (neutrons/fission)", 1.0, 4.0, 2.43, 0.01)
            eta = st.slider("η (efficacité)", 0.5, 1.0, 0.95, 0.01)
            p = st.slider("p (probabilité résonance)", 0.5, 1.0, 0.9, 0.01)
            f = st.slider("f (fraction thermalisation)", 0.5, 1.0, 0.85, 0.01)
            eps = st.slider("ε (facteur fission rapide)", 1.0, 1.1, 1.03, 0.001)

            k_eff = nu * eta * p * f * eps
            st.metric("k_eff = ν·η·p·f·ε", f"{k_eff:.4f}")
            st.metric("Régime",
                      "💥 Sur-critique" if k_eff > 1 else
                      "⚖️ Critique" if abs(k_eff-1) < 0.001 else
                      "🟢 Sous-critique")

        with col2:
            st.markdown("#### 🌟 Fusion D-T")
            bilan_fusion = engine.energie_fusion_DT()
            for k, v in bilan_fusion.items():
                st.metric(k, str(v))

            st.markdown("#### 🌡️ Critère de Lawson")
            T_keV = st.slider("Température T (keV)", 1, 100, 10)
            n_tau = st.slider("nτ (m⁻³s, log)", 17, 22, 20)

            n_tau_val = 10**n_tau
            critere_ok = n_tau_val >= 1e20 and T_keV >= 10
            st.metric("nτ (m⁻³s)", f"{n_tau_val:.1e}")
            st.metric("Critère Lawson", "✅ Atteint" if critere_ok else "❌ Non atteint")
            st.metric("Q (gain énergétique) estimé",
                      f">{5:.0f}" if critere_ok else "<1")

        # Q-valeurs
        st.markdown("### ⚡ Q-valeur de réactions")
        reactions = {
            "D + T → ⁴He + n":   ([2.014102, 3.016049], [4.002602, 1.008665]),
            "D + D → T + p":      ([2.014102, 2.014102], [3.016049, 1.007276]),
            "n + ²³⁵U → fission": ([1.008665, 235.04393], [140+92+3*1.008665]),
            "p + p → D + e⁺ + ν": ([1.007276, 1.007276], [2.014102, 0.000549]),
        }
        Q_data = []
        for nom, (m_in, m_out) in reactions.items():
            if isinstance(m_out, list):
                Q = engine.energie_Q(m_in, m_out)
            else:
                Q = (sum(m_in) - m_out) * engine.c2
            Q_data.append({"Réaction": nom, "Q (MeV)": round(Q, 3),
                           "Type": "Exoénergétique" if Q > 0 else "Endoénergétique"})
        st.dataframe(pd.DataFrame(Q_data), use_container_width=True)

    # ============================================================
    # TAB 4 : DOSIMÉTRIE
    # ============================================================
    with tab4:
        st.markdown("### 🔬 Dosimétrie & Radioprotection")
        col1, col2 = st.columns([1, 2])

        with col1:
            type_ray = st.selectbox("Type de rayonnement",
                ["γ (photons)", "α (alpha)", "β (bêta)",
                 "n (neutrons thermiques)"])
            facteurs_Q = {"γ (photons)": 1, "α (alpha)": 20,
                          "β (bêta)": 1, "n (neutrons thermiques)": 5}
            fQ = facteurs_Q[type_ray]

            D_Gy = st.slider("Dose absorbée (mGy)", 0.01, 10000.0, 10.0, 0.01) * 1e-3
            masse = st.slider("Masse irradiée (kg)", 0.001, 100.0, 70.0, 0.1)

            H_Sv = engine.dose_equivalente(D_Gy, fQ)
            E_dep = D_Gy * masse

            st.metric("Facteur Q", fQ)
            st.metric("Dose equiv. H (mSv)", f"{H_Sv*1000:.4f}")
            st.metric("Énergie déposée (J)", f"{E_dep:.4e}")

            st.markdown("### 📋 Limites réglementaires")
            limites = {
                "Public (annuel)": "1 mSv",
                "Travailleur (annuel)": "20 mSv",
                "Accident corps entier": "1000 mSv → ARS",
                "Limite peau": "500 mSv",
                "Fond naturel France": "~2.4 mSv/an",
            }
            for k, v in limites.items():
                st.markdown(f"- **{k}** : {v}")

        with col2:
            # Atténuation γ
            st.markdown("#### 📡 Atténuation de rayonnement γ")
            materiaux_mu = {
                "Eau":    0.0696,
                "Béton":  0.170,
                "Plomb":  1.640,
                "Fer":    0.195,
                "Air":    0.000118,
            }
            mat_atten = st.selectbox("Matériau", list(materiaux_mu.keys()))
            mu = materiaux_mu[mat_atten]
            D0_dos = st.slider("D₀ (mGy/h)", 0.01, 1000.0, 100.0, 0.01) * 1e-3

            x_arr = np.linspace(0, 5/mu, 500)
            D_x = engine.dose_exponentielle(D0_dos, mu, x_arr)
            CDV = engine.couche_demi_valeur(mu)

            c1, c2 = st.columns(2)
            with c1: st.metric("CDV (cm)", f"{CDV*100:.2f}")
            with c2: st.metric("μ (cm⁻¹)", f"{mu/100:.4f}")

            fig_dos = go.Figure()
            fig_dos.add_trace(go.Scatter(
                x=x_arr*100, y=D_x*1000, mode='lines',
                name=f'D(x) — {mat_atten}',
                line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
            ))
            fig_dos.add_hline(y=D0_dos*500, line_color='#ffcc00', line_dash='dash',
                              annotation_text="D₀/2")
            fig_dos.add_vline(x=CDV*100, line_color='#ff00cc', line_dash='dash',
                              annotation_text=f"CDV={CDV*100:.1f}cm")
            fig_dos.update_layout(
                title=f"Atténuation γ dans {mat_atten}",
                xaxis_title="x (cm)", yaxis_title="D (mGy/h)",
                yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
            )
            st.plotly_chart(fig_dos, use_container_width=True)

            # Comparaison matériaux
            fig_mat = go.Figure()
            x_comp = np.linspace(0, 0.3, 300)
            for i, (mat, mu_m) in enumerate(materiaux_mu.items()):
                D_m = engine.dose_exponentielle(1.0, mu_m, x_comp)
                fig_mat.add_trace(go.Scatter(x=x_comp*100, y=D_m, mode='lines',
                    name=mat, line=dict(color=colors_n[i%len(colors_n)], width=2)))
            fig_mat.update_layout(
                title="Comparaison atténuation γ (D/D₀)",
                xaxis_title="x (cm)", yaxis_title="D/D₀", yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
            )
            st.plotly_chart(fig_mat, use_container_width=True)

    # ============================================================
    # TAB 5 : CARTE DES NOYAUX
    # ============================================================
    with tab5:
        st.markdown("### 🗺️ Carte des noyaux (Charte de Segrè simplifiée)")
        col1, col2 = st.columns([1, 3])

        with col1:
            Z_max = st.slider("Z max", 20, 120, 60)
            show_ba = st.checkbox("Colorier par B/A", True)
            show_stable = st.checkbox("Mettre en valeur stables", True)

        with col2:
            Z_range = np.arange(1, Z_max+1)
            A_range = np.arange(1, min(Z_max*2+50, 251))

            Z_list, A_list, BA_list = [], [], []
            for Z in range(1, Z_max+1):
                for N in range(1, min(Z+50, 161)):
                    A = Z + N
                    if A > 250:
                        continue
                    B = engine.bethe_weizsacker(A, Z)
                    B_A = B / A
                    if B_A > 0:
                        Z_list.append(Z)
                        A_list.append(A)
                        BA_list.append(B_A)

            Z_arr = np.array(Z_list)
            A_arr_c = np.array(A_list)
            N_arr_c = A_arr_c - Z_arr
            BA_arr = np.array(BA_list)

            fig_sегре = go.Figure()

            if show_ba:
                fig_sегре.add_trace(go.Scatter(
                    x=N_arr_c, y=Z_arr, mode='markers',
                    marker=dict(
                        color=BA_arr, size=3,
                        colorscale=[[0,'#020817'],[0.4,'#7700ff'],
                                    [0.7,'#00ccff'],[1,'#00ff88']],
                        showscale=True,
                        colorbar=dict(title='B/A (MeV)', tickfont=dict(color='#c0d0ff'))
                    ),
                    name='Noyaux', hovertemplate="N=%{x}, Z=%{y}, B/A=%{marker.color:.2f}"
                ))
            else:
                fig_sегре.add_trace(go.Scatter(
                    x=N_arr_c, y=Z_arr, mode='markers',
                    marker=dict(color='#00ccff', size=2),
                    name='Noyaux'
                ))

            # Vallée de stabilité
            A_vs = np.arange(1, 251)
            Z_vs = engine.vallee_stabilite(A_vs)
            N_vs = A_vs - Z_vs
            fig_sегре.add_trace(go.Scatter(
                x=N_vs, y=Z_vs, mode='lines', name='Ligne de stabilité',
                line=dict(color='#ff0000', width=2)
            ))

            # Ligne N = Z
            z_diag = np.arange(1, Z_max)
            fig_sегре.add_trace(go.Scatter(
                x=z_diag, y=z_diag, mode='lines', name='N=Z',
                line=dict(color='rgba(255,255,255,0.3)', width=1.5, dash='dot')
            ))

            if show_stable:
                for nom, noy in NOYAUX_CONNUS.items():
                    if noy["stable"] and noy["Z"] <= Z_max:
                        N_noy = noy["A"] - noy["Z"]
                        fig_sегре.add_trace(go.Scatter(
                            x=[N_noy], y=[noy["Z"]], mode='markers+text',
                            text=[nom], textposition='top right',
                            marker=dict(color='#ffcc00', size=8, symbol='star'),
                            name=nom, showlegend=False
                        ))

            fig_sегре.update_layout(
                title="Charte de Segrè — Carte des noyaux",
                xaxis_title="N (neutrons)", yaxis_title="Z (protons)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=600,
            )
            st.plotly_chart(fig_sегре, use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire Physique Nucléaire")
        for nom, formule in FORMULES.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("---")
        st.markdown("### 🔬 Constantes nucléaires")
        df_c = pd.DataFrame([{"Constante": k, "Valeur": v}
                              for k, v in CONSTANTES.items()])
        st.dataframe(df_c, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Noyaux de référence")
        df_noy = pd.DataFrame([
            {"Noyau": k, "Z": v["Z"], "A": v["A"],
             "Masse (u)": v["masse_u"],
             "Stable": "✅" if v["stable"] else "❌",
             "T½": f"{v.get('T12_s',0):.2e} s" if not v["stable"] else "—"}
            for k, v in NOYAUX_CONNUS.items()
        ])
        st.dataframe(df_noy, use_container_width=True)

        st.markdown("---")
        for r in ["Krane — *Introductory Nuclear Physics* (Wiley, 1988)",
                  "Basdevant et al. — *Fundamentals in Nuclear Physics* (Springer, 2005)",
                  "Segré — *Nuclei and Particles* (Benjamin, 1977)"]:
            st.markdown(f"- {r}")