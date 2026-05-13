import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from scipy import signal as sp_signal
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES PHYSIQUES
# ============================================================
CONSTANTES = {
    "h_Planck":     (6.626e-34,  "J·s",   "Constante de Planck"),
    "hbar":         (1.055e-34,  "J·s",   "Constante de Planck réduite"),
    "c_lumiere":    (2.998e8,    "m/s",   "Vitesse de la lumière"),
    "k_Boltzmann":  (1.381e-23,  "J/K",   "Constante de Boltzmann"),
    "epsilon_0":    (8.854e-12,  "F/m",   "Permittivité du vide"),
    "e_electron":   (1.602e-19,  "C",     "Charge élémentaire"),
}

# ============================================================
# FORMULAIRE SCIENTIFIQUE
# ============================================================
FORMULES = {
    "Décroissance laser":       r"I(t) = I_0 \cdot e^{-\gamma t}",
    "Demi-vie":                 r"t_{1/2} = \frac{\ln 2}{\gamma}",
    "Équations de taux (N₂)":  r"\frac{dN_2}{dt} = R_p - \frac{N_2}{\tau}",
    "Inversion de population":  r"\Delta N = N_2 - N_1",
    "Gain laser":               r"g(\nu) = \sigma(\nu)\,\Delta N",
    "Section efficace":         r"\sigma(\nu) = \frac{\lambda^2}{8\pi n^2 \tau}\,g(\nu)",
    "Équation de Frantz-Nodvik":r"I_{out} = I_{sat}\ln\left[1 + \left(e^{I_{in}/I_{sat}}-1\right)e^{g_0 L}\right]",
    "Puissance seuil":          r"P_{th} = \frac{h\nu \cdot V \cdot \Delta N_{th}}{\eta_p \tau_p}",
    "Longueur d'onde":          r"\lambda = \frac{c}{\nu} = \frac{hc}{E_2 - E_1}",
    "Fréquence modes cavité":   r"\nu_m = \frac{mc}{2nL}, \quad m \in \mathbb{N}",
    "Facteur Q cavité":         r"Q = \frac{2\pi\nu_0 \tau_p}{1} = \frac{\omega_0}{2\delta}",
    "Finesse":                  r"\mathcal{F} = \frac{\pi\sqrt{R}}{1-R}",
    "Équation Bloch optique":   r"\dot{\rho}_{12} = -\left(\frac{1}{T_2} + i\delta\right)\rho_{12} + \frac{i\Omega}{2}(\rho_{11}-\rho_{22})",
}

TYPES_LASER = {
    "He-Ne":        {"λ": 632.8,  "type": "Gaz",       "η": 0.001, "τ": 10e-9,  "P_typ": "1-50 mW"},
    "CO₂":          {"λ": 10600,  "type": "Gaz",       "η": 0.20,  "τ": 1e-6,   "P_typ": "10 W - 100 kW"},
    "Nd:YAG":       {"λ": 1064,   "type": "Solide",    "η": 0.04,  "τ": 230e-6, "P_typ": "1 mW - 10 kW"},
    "Ti:Saphir":    {"λ": 800,    "type": "Solide",    "η": 0.10,  "τ": 3.2e-6, "P_typ": "femtoseconde"},
    "GaAs (diode)": {"λ": 870,    "type": "Semi-cond", "η": 0.50,  "τ": 1e-9,   "P_typ": "1 mW - 10 W"},
    "Excimère ArF": {"λ": 193,    "type": "Gaz",       "η": 0.02,  "τ": 5e-9,   "P_typ": "1-100 W"},
    "Fibre Yb":     {"λ": 1030,   "type": "Fibre",     "η": 0.80,  "τ": 1e-3,   "P_typ": "1 W - 10 kW"},
}


# ============================================================
# MOTEUR LASER
# ============================================================
class LaserEngine:
    """Moteur de simulation laser complet : taux, modes, cavité."""

    def __init__(self, I0: float, gamma: float):
        self.I0 = I0
        self.gamma = gamma

    # --- Modèle simple ---
    def intensite_simple(self, t: np.ndarray) -> np.ndarray:
        return self.I0 * np.exp(-self.gamma * t)

    @property
    def demi_vie(self) -> float:
        return np.log(2) / self.gamma if self.gamma > 0 else np.inf

    @property
    def temps_vie(self) -> float:
        return 1.0 / self.gamma if self.gamma > 0 else np.inf

    # --- Équations de taux à 4 niveaux ---
    def equations_taux_4niveaux(self, t: np.ndarray,
                                 Rp: float, tau21: float,
                                 tau32: float, tau10: float,
                                 N_total: float = 1e20) -> dict:
        """
        Système 4 niveaux : pompage → N3 → N2 → N1 → N0
        """
        def dydt(t, y):
            N0, N1, N2, N3 = y
            dN3 = Rp * N0 - N3 / tau32
            dN2 = N3 / tau32 - N2 / tau21
            dN1 = N2 / tau21 - N1 / tau10
            dN0 = N1 / tau10 - Rp * N0
            return [dN0, dN1, dN2, dN3]

        y0 = [N_total, 0, 0, 0]
        sol = solve_ivp(dydt, [t[0], t[-1]], y0, t_eval=t,
                        method='RK45', rtol=1e-6, atol=1e-10)
        return {
            "t": sol.t,
            "N0": sol.y[0], "N1": sol.y[1],
            "N2": sol.y[2], "N3": sol.y[3],
            "inversion": sol.y[2] - sol.y[1],
        }

    # --- Oscillations de relaxation ---
    def oscillations_relaxation(self, t: np.ndarray,
                                 Rp: float, tau_c: float,
                                 tau_sp: float, N_th: float,
                                 S0: float = 1e3) -> dict:
        """Équations couplées photons/population autour du seuil."""
        def dydt(t, y):
            N, S = y
            dN = Rp - N / tau_sp - N * S / tau_c
            dS = N * S / tau_c - S / tau_c + N / tau_sp
            return [dN, dS]

        y0 = [N_th * 0.9, S0]
        sol = solve_ivp(dydt, [t[0], t[-1]], y0, t_eval=t,
                        method='RK45', rtol=1e-8, atol=1e-12,
                        max_step=(t[-1]-t[0])/2000)
        return {"t": sol.t, "N": sol.y[0], "S": sol.y[1]}

    # --- Profil spectral ---
    def profil_spectral(self, lambda_center: float,
                        delta_lambda: float, n_modes: int,
                        gain: float) -> tuple:
        """Spectre laser avec modes de cavité et profil de gain."""
        lambda_arr = np.linspace(lambda_center - 5*delta_lambda,
                                  lambda_center + 5*delta_lambda, 2000)
        # Profil de gain (lorentzien)
        gamma_l = delta_lambda / 2
        gain_profile = gain / (1 + ((lambda_arr - lambda_center)/gamma_l)**2)

        # Modes de cavité
        modes_lambda = []
        modes_gain = []
        delta_mode = delta_lambda / max(n_modes, 1)
        for m in range(-n_modes//2, n_modes//2 + 1):
            lm = lambda_center + m * delta_mode
            gm = gain / (1 + ((lm - lambda_center)/gamma_l)**2)
            modes_lambda.append(lm)
            modes_gain.append(gm)

        return lambda_arr, gain_profile, np.array(modes_lambda), np.array(modes_gain)

    # --- Frantz-Nodvik (amplification) ---
    def frantz_nodvik(self, I_in: np.ndarray,
                       g0: float, L: float, I_sat: float) -> np.ndarray:
        """Équation de Frantz-Nodvik pour amplificateur laser."""
        return I_sat * np.log(1 + (np.exp(I_in / I_sat) - 1) * np.exp(g0 * L))

    # --- Modes TEMₘₙ gaussiens ---
    def mode_gaussien(self, x: np.ndarray, y: np.ndarray,
                       w0: float, z: float,
                       lambda_um: float = 1.064) -> np.ndarray:
        """Profil TEM₀₀ gaussien en champ lointain.

        x, y sont en mètres, w0 est en millimètres et lambda_um en micromètres.
        """
        w0_m = w0 * 1e-3
        lambda_m = lambda_um * 1e-6
        zR = np.pi * w0_m**2 / lambda_m
        wz = w0_m * np.sqrt(1 + (z / zR)**2)
        X, Y = np.meshgrid(x, y)
        return np.exp(-2 * (X**2 + Y**2) / wz**2)

    # --- Énergie & puissance ---
    def energie_impulsion(self, I0: float, tau_p: float,
                           w0: float, forme: str = "gaussien") -> float:
        """Énergie d'une impulsion laser."""
        A_eff = np.pi * w0**2 / 2
        if forme == "gaussien":
            return I0 * A_eff * tau_p * np.sqrt(np.pi / (4 * np.log(2)))
        elif forme == "sech2":
            return I0 * A_eff * tau_p * 1.7627
        return I0 * A_eff * tau_p

    # --- Diagnostics ---
    def diagnostiquer(self, I0: float, gamma: float,
                       t_max: float) -> list:
        """Analyse automatique des paramètres laser."""
        diag = []
        t12 = np.log(2) / gamma if gamma > 0 else np.inf
        tau = 1 / gamma if gamma > 0 else np.inf

        diag.append({
            "Paramètre": "Demi-vie t₁/₂",
            "Valeur": f"{t12:.4f} s",
            "Statut": "✅ OK" if t12 < t_max else "⚠️ Hors fenêtre",
            "Note": f"{'Visible' if t12 < t_max else 'Augmenter t_max'}"
        })
        diag.append({
            "Paramètre": "Temps de vie τ",
            "Valeur": f"{tau:.4f} s",
            "Statut": "✅ OK",
            "Note": f"I(τ) = I₀/e = {I0/np.e:.3f}"
        })
        diag.append({
            "Paramètre": "Rapport I(t_max)/I₀",
            "Valeur": f"{np.exp(-gamma*t_max)*100:.2f}%",
            "Statut": "✅ OK" if np.exp(-gamma*t_max) > 1e-6 else "⚠️ Signal nul",
            "Note": "Augmenter I₀ ou réduire γ si signal nul"
        })
        diag.append({
            "Paramètre": "Régime",
            "Valeur": f"γ = {gamma:.3f} s⁻¹",
            "Statut": "🔴 Rapide" if gamma > 1 else "🟡 Moyen" if gamma > 0.1 else "🟢 Lent",
            "Note": f"{'Sur-amorti' if gamma > 1 else 'Normal'}"
        })
        return diag


@st.cache_data(show_spinner=False)
def compute_decay(I0: float, gamma: float, t_max: float, n_points: int):
    t = np.linspace(0, t_max, n_points)
    return t, I0 * np.exp(-gamma * t)


@st.cache_data(show_spinner=False)
def solve_4niveaux(Rp: float, tau21: float, tau32: float,
                  tau10: float, t_max: float, n_points: int):
    t = np.linspace(0, t_max, n_points)

    def dydt(t, y):
        N0, N1, N2, N3 = y
        dN3 = Rp * N0 - N3 / tau32
        dN2 = N3 / tau32 - N2 / tau21
        dN1 = N2 / tau21 - N1 / tau10
        dN0 = N1 / tau10 - Rp * N0
        return [dN0, dN1, dN2, dN3]

    sol = solve_ivp(dydt, [t[0], t[-1]], [1e20, 0, 0, 0], t_eval=t,
                    method='RK45', rtol=1e-6, atol=1e-10)
    return {
        "t": sol.t,
        "N0": sol.y[0], "N1": sol.y[1],
        "N2": sol.y[2], "N3": sol.y[3],
        "inversion": sol.y[2] - sol.y[1],
    }


@st.cache_data(show_spinner=False)
def solve_relaxation(Rp_abs: float, tau_c: float, tau_sp: float,
                     N_th: float, t_max_osc: float, n_points: int = 2000):
    t = np.linspace(0, t_max_osc, n_points)

    def dydt(t, y):
        N, S = y
        dN = Rp_abs - N / tau_sp - N * S / tau_c
        dS = N * S / tau_c - S / tau_c + N / tau_sp
        return [dN, dS]

    sol = solve_ivp(dydt, [t[0], t[-1]], [N_th * 0.9, 1e3], t_eval=t,
                    method='RK45', rtol=1e-8, atol=1e-12,
                    max_step=(t[-1]-t[0]) / 2000)
    return {"t": sol.t, "N": sol.y[0], "S": sol.y[1]}


@st.cache_data(show_spinner=False)
def compute_spectral(lambda_center: float, delta_lambda: float,
                     n_modes: int, gain: float):
    lambda_arr = np.linspace(lambda_center - 5*delta_lambda,
                              lambda_center + 5*delta_lambda, 2000)
    gamma_l = delta_lambda / 2
    gain_profile = gain / (1 + ((lambda_arr - lambda_center)/gamma_l)**2)
    delta_mode = delta_lambda / max(n_modes, 1)
    modes_lambda = np.array([lambda_center + m * delta_mode
                             for m in range(-n_modes//2, n_modes//2 + 1)])
    modes_gain = gain / (1 + ((modes_lambda - lambda_center)/gamma_l)**2)
    return lambda_arr, gain_profile, modes_lambda, modes_gain


@st.cache_data(show_spinner=False)
def compute_beam_profile(w0: float, z: float, lambda_um: float):
    w0_m = w0 * 1e-3
    lambda_m = lambda_um * 1e-6
    zR = np.pi * w0_m**2 / lambda_m
    wz = w0_m * np.sqrt(1 + (z / zR)**2)
    x = np.linspace(-5*wz, 5*wz, 100)
    y = np.linspace(-5*wz, 5*wz, 100)
    X, Y = np.meshgrid(x, y)
    return x, y, np.exp(-2 * (X**2 + Y**2) / wz**2), zR, wz


@st.cache_data(show_spinner=False)
def compute_frantz_nodvik(g0: float, L: float, I_sat: float, I_max: float):
    I_in = np.linspace(0.01, I_max, 500)
    I_out = I_sat * np.log(1 + (np.exp(I_in / I_sat) - 1) * np.exp(g0 * L))
    return I_in, I_out, I_out / I_in


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def laser_page():
    st.markdown("## 💡 Simulation Laser Avancée")
    st.markdown("*Dynamique laser, équations de taux, modes de cavité, amplification*")
    st.markdown("---")

    section = st.radio(
        "Section",
        [
            "📉 Décroissance & Taux",
            "🌊 Oscillations de relaxation",
            "📡 Spectre & Modes",
            "🔬 Profil TEM gaussien",
            "⚡ Amplification (F-N)",
            "📖 Théorie & Références",
        ],
        horizontal=True,
    )

    # ============================================================
    # SECTION 1 : DÉCROISSANCE & ÉQUATIONS DE TAUX
    # ============================================================
    if section == "📉 Décroissance & Taux":
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")

            laser_type = st.selectbox("Type de laser", ["Personnalisé"] + list(TYPES_LASER.keys()))
            if laser_type != "Personnalisé":
                info_l = TYPES_LASER[laser_type]
                st.info(f"λ = {info_l['λ']} nm | η = {info_l['η']*100:.0f}% | {info_l['P_typ']}")

            I0 = st.slider("Intensité initiale I₀", 0.1, 200.0, 10.0, 0.5)
            gamma = st.slider("Coefficient γ (s⁻¹)", 0.01, 5.0, 0.1, 0.01)
            t_max = st.slider("Temps max (s)", 1.0, 100.0, 30.0, 1.0)
            n_points = st.slider("Résolution", 200, 5000, 1000, 100)

            mode_affichage = st.radio("Affichage", ["Linéaire", "Semi-log"], horizontal=True)

            engine = LaserEngine(I0, gamma)

            st.markdown("### 📐 Résultats analytiques")
            st.metric("Demi-vie t₁/₂ (s)", f"{engine.demi_vie:.4f}")
            st.metric("Temps de vie τ (s)", f"{engine.temps_vie:.4f}")
            st.metric("I(t_max)", f"{I0 * np.exp(-gamma * t_max):.4e}")
            st.metric("I(τ) = I₀/e", f"{I0/np.e:.4f}")

            # Paramètres 4 niveaux
            st.markdown("### 🔬 Système 4 niveaux")
            show_4n = st.checkbox("Simuler système 4 niveaux", True)
            if show_4n:
                Rp = st.slider("Taux pompage Rp (s⁻¹)", 1e5, 1e8, 1e6,
                               format="%.0e", step=1e5)
                tau21 = st.slider("τ₂₁ (μs)", 0.1, 500.0, 230.0, 1.0)
                tau32 = st.slider("τ₃₂ (ns)", 0.1, 100.0, 1.0, 0.1)

        with col2:
            t, I = compute_decay(I0, gamma, t_max, n_points)

            fig = go.Figure()

            if mode_affichage == "Linéaire":
                fig.add_trace(go.Scatter(
                    x=t, y=I, mode='lines', name='I(t)',
                    line=dict(color='#00ccff', width=3)
                ))
                fig.add_hline(y=I0/np.e, line_dash='dash', line_color='#ffcc00',
                              annotation_text=f"I₀/e = {I0/np.e:.3f}")
                fig.add_hline(y=I0/2, line_dash='dot', line_color='#7700ff',
                              annotation_text=f"I₀/2 = {I0/2:.3f}")
                fig.add_vline(x=engine.demi_vie, line_dash='dash', line_color='#ff00cc',
                              annotation_text=f"t₁/₂={engine.demi_vie:.2f}s")
                fig.add_vline(x=engine.temps_vie, line_dash='dot', line_color='#00ff88',
                              annotation_text=f"τ={engine.temps_vie:.2f}s")
            else:
                I_log = np.where(I > 0, I, 1e-12)
                fig.add_trace(go.Scatter(
                    x=t, y=np.log10(I_log), mode='lines',
                    name='log₁₀(I(t))', line=dict(color='#00ccff', width=3)
                ))
                # Droite théorique
                fig.add_trace(go.Scatter(
                    x=t, y=np.log10(I0) - gamma * t / np.log(10),
                    mode='lines', name='Droite théorique',
                    line=dict(color='#ffcc00', width=2, dash='dash')
                ))

            fig.update_layout(
                title="Décroissance laser I(t) = I₀·e^{-γt}",
                xaxis_title="Temps (s)",
                yaxis_title="Intensité" if mode_affichage == "Linéaire" else "log₁₀(I)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Système 4 niveaux
            if show_4n:
                sol4 = solve_4niveaux(
                    Rp=Rp,
                    tau21=tau21*1e-6,
                    tau32=tau32*1e-9,
                    tau10=tau32*1e-9*0.1,
                    t_max=t_max * 1e-6,
                    n_points=2000,
                )
                fig4 = go.Figure()
                colors4 = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88']
                for i, (k, v) in enumerate([("N₀", sol4["N0"]), ("N₁", sol4["N1"]),
                                             ("N₂", sol4["N2"]), ("N₃", sol4["N3"])]):
                    fig4.add_trace(go.Scatter(
                        x=sol4["t"]*1e6, y=v, mode='lines', name=k,
                        line=dict(color=colors4[i], width=2)
                    ))
                fig4.add_trace(go.Scatter(
                    x=sol4["t"]*1e6, y=sol4["inversion"],
                    mode='lines', name='ΔN (inversion)',
                    line=dict(color='#ffffff', width=3, dash='dash')
                ))
                fig4.update_layout(
                    title="Populations système 4 niveaux",
                    xaxis_title="Temps (μs)", yaxis_title="Population (m⁻³)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                    height=380,
                )
                st.plotly_chart(fig4, use_container_width=True)

            # Export
            df_exp = pd.DataFrame({"temps_s": t, "intensite": I})
            st.download_button("💾 Export CSV",
                               df_exp.to_csv(index=False).encode(),
                               "laser_simulation.csv", "text/csv")

    # ============================================================
    # SECTION 2 : OSCILLATIONS DE RELAXATION
    # ============================================================
    elif section == "🌊 Oscillations de relaxation":
        st.markdown("### 🌊 Oscillations de relaxation laser")
        st.markdown("*Dynamique couplée photons ↔ population autour du seuil*")

        col1, col2 = st.columns([1, 2])
        with col1:
            Rp_osc = st.slider("Taux de pompe Rp (×Rp_th)", 1.01, 5.0, 1.5, 0.01)
            tau_c_osc = st.slider("Durée de vie photon τ_c (ns)", 0.1, 100.0, 10.0, 0.1)
            tau_sp_osc = st.slider("Durée de vie spontanée τ_sp (μs)", 0.1, 1000.0, 230.0, 1.0)
            t_max_osc = st.slider("Durée simulation (μs)", 1.0, 500.0, 50.0, 1.0)

            tau_c = tau_c_osc * 1e-9
            tau_sp = tau_sp_osc * 1e-6
            N_th = 1.0 / (tau_c)
            Rp_abs = Rp_osc * N_th / tau_sp

            # Fréquence théorique
            omega_r = 0.0
            if Rp_abs * tau_sp > 1:
                omega_r = np.sqrt((Rp_abs * tau_sp - 1) / (tau_c * tau_sp))
            f_osc = omega_r / (2 * np.pi) if omega_r > 0 else 0.0

            st.metric("Fréquence oscillations (MHz)",
                      f"{f_osc/1e6:.3f}" if f_osc > 0 else "N/A")
            st.metric("Période (ns)",
                      f"{1/f_osc*1e9:.1f}" if f_osc > 0 else "N/A")

        with col2:
            sol_osc = solve_relaxation(
                Rp_abs=Rp_abs,
                tau_c=tau_c,
                tau_sp=tau_sp,
                N_th=N_th,
                t_max_osc=t_max_osc * 1e-6,
                n_points=2000,
            )

            fig_osc = make_subplots(rows=2, cols=1,
                subplot_titles=["Densité photons S(t)", "Population N(t)"])

            fig_osc.add_trace(go.Scatter(
                x=sol_osc["t"]*1e6, y=sol_osc["S"], mode='lines',
                name='S(t)', line=dict(color='#00ccff', width=2)
            ), row=1, col=1)
            fig_osc.add_trace(go.Scatter(
                x=sol_osc["t"]*1e6, y=sol_osc["N"]/N_th, mode='lines',
                name='N(t)/N_th', line=dict(color='#7700ff', width=2)
            ), row=2, col=1)

            fig_osc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                height=480,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_osc.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                  title_text="Temps (μs)")
            fig_osc.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_osc, use_container_width=True)

    # ============================================================
    # SECTION 3 : SPECTRE & MODES
    # ============================================================
    elif section == "📡 Spectre & Modes":
        st.markdown("### 📡 Spectre d'émission et modes de cavité")

        col1, col2 = st.columns([1, 2])
        with col1:
            laser_sel = st.selectbox("Laser", list(TYPES_LASER.keys()), key="spec_laser")
            info_spec = TYPES_LASER[laser_sel]
            lambda_c = info_spec["λ"]

            delta_lambda = st.slider("Largeur spectrale Δλ (nm)", 0.001, 50.0,
                                      0.1 if "Nd" in laser_sel else 1.0, 0.001)
            n_modes = st.slider("Nombre de modes longitudinaux", 1, 20, 5)
            gain_val = st.slider("Gain g₀", 0.1, 10.0, 1.0, 0.1)

            L_cav = st.slider("Longueur de cavité L (cm)", 1, 200, 30)
            n_ref = st.slider("Indice de réfraction n", 1.0, 3.5, 1.0, 0.01)

            delta_nu = 3e8 / (2 * n_ref * L_cav * 1e-2)
            st.metric("Espacement modes (MHz)", f"{delta_nu/1e6:.1f}")
            st.metric("Finesse théorique", f"{np.pi*0.99/(1-0.99):.0f}")

        with col2:
            lambda_arr, gain_prof, modes_l, modes_g = compute_spectral(
                lambda_center=lambda_c,
                delta_lambda=delta_lambda,
                n_modes=n_modes,
                gain=gain_val,
            )

            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=lambda_arr, y=gain_prof, mode='lines',
                name='Profil de gain', line=dict(color='rgba(0,204,255,0.5)', width=2)
            ))

            colors_modes = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88',
                            '#ffcc00', '#ff4444', '#44ff88']
            for i, (lm, gm) in enumerate(zip(modes_l, modes_g)):
                fig_spec.add_trace(go.Scatter(
                    x=[lm, lm], y=[0, gm], mode='lines',
                    name=f'Mode {i+1}',
                    line=dict(color=colors_modes[i % len(colors_modes)], width=3)
                ))

            fig_spec.update_layout(
                title=f"Spectre laser — {laser_sel} (λ={lambda_c} nm)",
                xaxis_title="Longueur d'onde (nm)", yaxis_title="Gain (u.a.)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                          range=[0, gain_val * 1.1]),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig_spec, use_container_width=True)

    # ============================================================
    # SECTION 4 : PROFIL TEM GAUSSIEN
    # ============================================================
    elif section == "🔬 Profil TEM gaussien":
        st.markdown("### 🔬 Profil de faisceau TEM₀₀ gaussien")

        col1, col2 = st.columns([1, 2])
        with col1:
            w0_beam = st.slider("Waist w₀ (mm)", 0.1, 10.0, 1.0, 0.1)
            z_beam = st.slider("Distance z (m)", 0.0, 10.0, 0.0, 0.1)
            lambda_beam = st.slider("Longueur d'onde (μm)", 0.4, 11.0, 1.064, 0.001)

            zR = np.pi * (w0_beam*1e-3)**2 / (lambda_beam*1e-6)
            wz = w0_beam * np.sqrt(1 + (z_beam/zR)**2)
            Rz = z_beam * (1 + (zR/z_beam)**2) if z_beam > 0 else np.inf

            st.metric("Longueur Rayleigh z_R (m)", f"{zR:.3f}")
            st.metric("Waist à z: w(z) (mm)", f"{wz:.3f}")
            st.metric("Divergence θ (mrad)", f"{lambda_beam*1e-6/(np.pi*w0_beam*1e-3)*1e3:.3f}")

        with col2:
            x_b, y_b, Z_beam, zR, wz = compute_beam_profile(
                w0=w0_beam,
                z=z_beam,
                lambda_um=lambda_beam,
            )

            fig_beam = go.Figure(data=[go.Surface(
                z=Z_beam, x=x_b*1e3, y=y_b*1e3,
                colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#ff4400'],[1,'#ffffff']],
                showscale=True,
                lighting=dict(ambient=0.5, diffuse=0.8),
            )])
            fig_beam.update_layout(
                title=f"Profil TEM₀₀ — z={z_beam} m, w(z)={wz:.2f} mm",
                scene=dict(
                    bgcolor='rgba(5,0,20,0.8)',
                    xaxis=dict(color='#c0d0ff', title='x (mm)'),
                    yaxis=dict(color='#c0d0ff', title='y (mm)'),
                    zaxis=dict(color='#c0d0ff', title='I (u.a.)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0d0ff'),
                height=480,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_beam, use_container_width=True)

            # Propagation du waist
            z_prop = np.linspace(0, 10*zR, 500)
            wz_prop = w0_beam * np.sqrt(1 + (z_prop/zR)**2)
            fig_prop = go.Figure()
            fig_prop.add_trace(go.Scatter(
                x=z_prop, y=wz_prop, mode='lines', name='+w(z)',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_prop.add_trace(go.Scatter(
                x=z_prop, y=-wz_prop, mode='lines', name='-w(z)',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_prop.add_vline(x=zR, line_color='#ffcc00', line_dash='dash',
                               annotation_text=f"z_R={zR:.2f}m")
            fig_prop.update_layout(
                title="Propagation du faisceau gaussien",
                xaxis_title="z (m)", yaxis_title="w(z) (mm)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                height=300,
            )
            st.plotly_chart(fig_prop, use_container_width=True)

    # ============================================================
    # SECTION 5 : AMPLIFICATION FRANTZ-NODVIK
    # ============================================================
    elif section == "⚡ Amplification (F-N)":
        st.markdown("### ⚡ Amplificateur laser — Frantz-Nodvik")

        col1, col2 = st.columns([1, 2])
        with col1:
            g0_fn = st.slider("Gain petit signal g₀ (cm⁻¹)", 0.01, 2.0, 0.5, 0.01)
            L_fn = st.slider("Longueur amplificateur L (cm)", 1.0, 50.0, 10.0, 0.5)
            I_sat_fn = st.slider("Intensité saturation I_sat (W/cm²)", 0.1, 1000.0, 100.0, 1.0)
            I_max = st.slider("I_in max (W/cm²)", 1.0, 5000.0, 500.0, 10.0)

            G_lin = np.exp(g0_fn * L_fn)
            st.metric("Gain linéaire G = e^(g₀L)", f"{G_lin:.2f}")
            st.metric("Gain (dB)", f"{10*np.log10(G_lin):.2f}")

        with col2:
            I_in, I_out, G_eff = compute_frantz_nodvik(
                g0=g0_fn,
                L=L_fn,
                I_sat=I_sat_fn,
                I_max=I_max,
            )

            fig_fn = make_subplots(rows=2, cols=1,
                subplot_titles=["I_out vs I_in", "Gain effectif G(I_in)"])

            fig_fn.add_trace(go.Scatter(
                x=I_in, y=I_out, mode='lines', name='I_out (F-N)',
                line=dict(color='#00ccff', width=3)
            ), row=1, col=1)
            fig_fn.add_trace(go.Scatter(
                x=I_in, y=G_lin * I_in, mode='lines', name='I_out (linéaire)',
                line=dict(color='rgba(255,200,0,0.5)', width=2, dash='dash')
            ), row=1, col=1)
            fig_fn.add_vline(x=I_sat_fn, line_color='#ff00cc', line_dash='dot',
                             annotation_text="I_sat", row=1, col=1)

            fig_fn.add_trace(go.Scatter(
                x=I_in, y=G_eff, mode='lines', name='G effectif',
                line=dict(color='#7700ff', width=2.5)
            ), row=2, col=1)
            fig_fn.add_hline(y=G_lin, line_color='#ffcc00', line_dash='dash',
                             annotation_text=f"G₀={G_lin:.1f}", row=2, col=1)

            fig_fn.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                height=520,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_fn.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                 title_text="I_in (W/cm²)")
            fig_fn.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_fn, use_container_width=True)

    # ============================================================
    # SECTION 6 : THÉORIE
    # ============================================================
    elif section == "📖 Théorie & Références":
        st.markdown("### 📖 Formulaire scientifique laser")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 🔬 Types de lasers")
        df_lasers = pd.DataFrame([
            {"Laser": k, "λ (nm)": v["λ"], "Type": v["type"],
             "η (%)": f"{v['η']*100:.0f}", "τ": str(v["τ"]), "Puissance": v["P_typ"]}
            for k, v in TYPES_LASER.items()
        ])
        st.dataframe(df_lasers, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📐 Diagnostic laser")
        engine_diag = LaserEngine(10.0, 0.1)
        diag = engine_diag.diagnostiquer(10.0, 0.1, 30.0)
        st.dataframe(pd.DataFrame(diag), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        refs = [
            "Saleh & Teich — *Fundamentals of Photonics* (Wiley, 2007)",
            "Svelto — *Principles of Lasers* (Springer, 2010)",
            "Siegman — *Lasers* (University Science Books, 1986)",
            "Yariv — *Quantum Electronics* (Wiley, 1989)",
        ]
        for r in refs:
            st.markdown(f"- {r}")