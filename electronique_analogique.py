import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sp_signal
from scipy.optimize import fsolve, brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Cache pour les calculs coûteux
@st.cache_data
def compute_filter_response(R, C, f_arr, ftype):
    """Calcul mis en cache pour les filtres."""
    omega = 2 * np.pi * f_arr
    tau = R * C
    if ftype == "LP":
        H = 1 / (1 + 1j * omega * tau)
    else:
        H = 1j * omega * tau / (1 + 1j * omega * tau)
    return H

@st.cache_data
def compute_aop_response(gain_DC, GBW, f_arr):
    """Calcul mis en cache pour AOP."""
    f_c = GBW / abs(gain_DC) if gain_DC != 0 else GBW
    H = gain_DC / (1 + 1j * f_arr / f_c)
    return H

FORMULES = {
    "Loi d'Ohm":              r"U = R \cdot I,\quad P = UI = \frac{U^2}{R} = RI^2",
    "Diviseur de tension":    r"U_{out} = U_{in}\frac{R_2}{R_1+R_2}",
    "Diviseur de courant":    r"I_{R1} = I\frac{R_2}{R_1+R_2}",
    "Filtre RC passe-bas":    r"H(j\omega)=\frac{1}{1+j\omega RC},\quad f_c=\frac{1}{2\pi RC}",
    "Filtre RL passe-haut":   r"H(j\omega)=\frac{j\omega L/R}{1+j\omega L/R}",
    "RLC série (résonance)":  r"\omega_0=\frac{1}{\sqrt{LC}},\quad Q=\frac{\omega_0 L}{R}=\frac{1}{R}\sqrt{\frac{L}{C}}",
    "Gain AOP":               r"G=-\frac{R_f}{R_1}\text{ (inverseur)},\quad G=1+\frac{R_f}{R_1}\text{ (non-inv.)}",
    "Diode Shockley":         r"I = I_s\!\left(e^{V/(nV_T)}-1\right),\quad V_T=\frac{kT}{q}\approx26\text{mV}",
    "Transistor BJT (Ic)":    r"I_C=\beta I_B=I_s e^{V_{BE}/V_T}",
    "Bande passante AOP":     r"f_{-3dB}=\frac{GBW}{|A_v|}",
    "Bruit thermique":        r"V_n=\sqrt{4kTR\Delta f}\quad\text{(tension de bruit)}",
    "Énergie condensateur":   r"E=\frac{1}{2}CV^2",
    "Énergie inductance":     r"E=\frac{1}{2}LI^2",
    "Puissance réactive":     r"Q=UI\sin\phi,\quad S=UI,\quad P=UI\cos\phi",
}

COMPOSANTS_STANDARDS = {
    "Résistances (E24)":  [10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91],
    "Condensateurs (pF)": [10,12,15,18,22,27,33,39,47,56,68,82,100,120,150,180,220,270,330,390,470],
    "Inductances (μH)":   [0.1,0.22,0.47,1,2.2,4.7,10,22,47,100,220,470,1000],
}


# ============================================================
# MOTEUR ELECTRONIQUE
# ============================================================
class ElecEngine:
    """Moteur de calcul en électronique analogique."""

    def __init__(self, fs: float = 1e6):
        self.fs = fs

    # --- Impédances ---
    def Z_R(self, R: float, f: float) -> complex:
        return complex(R, 0)

    def Z_C(self, C: float, f: float) -> complex:
        omega = 2 * np.pi * f
        return complex(0, -1/(omega*C)) if f > 0 else complex(np.inf, 0)

    def Z_L(self, L: float, f: float) -> complex:
        omega = 2 * np.pi * f
        return complex(0, omega*L)

    def Z_serie(self, *impedances) -> complex:
        return sum(impedances)

    def Z_parallele(self, *impedances) -> complex:
        return 1 / sum(1/z for z in impedances if z != 0)

    # --- Filtres ---
    def filtre_RC(self, R: float, C: float,
                  f_arr: np.ndarray, ftype: str = "LP") -> np.ndarray:
        """Filtre RC passe-bas/haut."""
        return compute_filter_response(R, C, f_arr, ftype)

    def filtre_RLC(self, R: float, L: float, C: float,
                   f_arr: np.ndarray, ftype: str = "BP") -> np.ndarray:
        """Filtre RLC."""
        omega = 2 * np.pi * f_arr
        omega0 = 1 / np.sqrt(L * C)
        Q = omega0 * L / R
        if ftype == "BP":
            H = (1j * omega / (omega0 * Q)) / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        elif ftype == "LP":
            H = 1 / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        elif ftype == "HP":
            H = -(omega/omega0)**2 / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        else:
            H = (1 - (omega/omega0)**2) / (1 - (omega/omega0)**2 + 1j*omega/(omega0*Q))
        return H

    def fc_RC(self, R: float, C: float) -> float:
        return 1 / (2 * np.pi * R * C)

    def freq_resonance(self, L: float, C: float) -> float:
        return 1 / (2 * np.pi * np.sqrt(L * C))

    def facteur_Q(self, R: float, L: float, C: float) -> float:
        f0 = self.freq_resonance(L, C)
        omega0 = 2 * np.pi * f0
        return omega0 * L / R

    # --- AOP ---
    def gain_inverseur(self, Rf: float, R1: float) -> float:
        return -Rf / R1

    def gain_non_inverseur(self, Rf: float, R1: float) -> float:
        return 1 + Rf / R1

    def gain_differentiel(self, R1: float, R2: float,
                           R3: float, R4: float) -> float:
        return (R4 / (R3 + R4)) * (1 + R2/R1) - R2/R1

    def bode_aop(self, gain_DC: float, GBW: float,
                 f_arr: np.ndarray) -> np.ndarray:
        """Réponse fréquentielle AOP idéal avec GBW."""
        return compute_aop_response(gain_DC, GBW, f_arr)

    def reponse_impulsionnelle_filtre(self, R: float, C: float,
                                       t: np.ndarray) -> np.ndarray:
        """Réponse indicielle RC."""
        tau = R * C
        return 1 - np.exp(-t / tau)

    # --- Diode ---
    def courant_diode(self, V: np.ndarray, Is: float = 1e-12,
                       n: float = 1.0, T: float = 300) -> np.ndarray:
        VT = 1.3806e-23 * T / 1.6022e-19
        return Is * (np.exp(np.clip(V/(n*VT), -100, 100)) - 1)

    def point_fonctionnement_diode(self, Vcc: float, R: float,
                                    Is: float = 1e-12) -> dict:
        """Point de fonctionnement diode-résistance (méthode graphique)."""
        VT = 0.02585
        def equations(V_d):
            I_d = Is * (np.exp(V_d/(VT)) - 1)
            I_R = (Vcc - V_d) / R
            return I_d - I_R
        try:
            V_d = brentq(equations, 0, Vcc * 0.99)
            I_d = Is * (np.exp(V_d/VT) - 1)
        except:
            V_d, I_d = 0.6, (Vcc-0.6)/R
        return {"V_d": V_d, "I_d": I_d, "P_diode": V_d*I_d,
                "P_resistor": I_d**2 * R}

    # --- Transistor BJT ---
    def point_repos_BJT(self, Vcc: float, Rb: float, Rc: float,
                         beta: float = 100) -> dict:
        VBE = 0.7
        IB = (Vcc - VBE) / Rb if Rb > 0 else 0
        IC = beta * IB
        VCE = Vcc - IC * Rc
        if VCE < 0.2:
            VCE = 0.2
            IC = (Vcc - VCE) / Rc
            IB = IC / beta
        return {"IB_uA": IB*1e6, "IC_mA": IC*1e3, "VCE": VCE,
                "regime": "Saturation" if VCE < 0.3 else
                          "Actif" if VCE < Vcc else "Blocage"}

    def droite_charge(self, Vcc: float, Rc: float,
                       n: int = 200) -> tuple:
        V = np.linspace(0, Vcc, n)
        I = (Vcc - V) / Rc
        return V, I

    # --- Bruit ---
    def bruit_thermique(self, R: float, T: float, BW: float) -> float:
        k = 1.3806e-23
        return np.sqrt(4 * k * T * R * BW)

    def SNR_db(self, V_signal: float, V_bruit: float) -> float:
        return 20 * np.log10(V_signal / (V_bruit + 1e-15))

    # --- Oscillateur ---
    def oscillateur_colpitts(self, L: float,
                              C1: float, C2: float) -> dict:
        C_eq = C1 * C2 / (C1 + C2)
        f0 = 1 / (2 * np.pi * np.sqrt(L * C_eq))
        rapport = C1 / C2
        return {"f0_MHz": f0/1e6, "C_eq_nF": C_eq*1e9,
                "rapport_C1_C2": rapport}


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def electronique_analogique_page():
    st.markdown("## 🔌 Électronique Analogique Avancée")
    st.markdown("*Filtres RC/RLC, AOP, diodes, transistors, bruit, oscillateurs*")
    st.markdown("---")

    engine = ElecEngine()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📡 Filtres RC/RLC",
        "🔧 AOP & Amplificateurs",
        "💡 Diodes",
        "🔦 Transistors BJT",
        "📻 Oscillateurs & Bruit",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : FILTRES
    # ============================================================
    with tab1:
        st.markdown("### 📡 Filtres analogiques")
        col1, col2 = st.columns([1, 2])

        with col1:
            type_filtre = st.selectbox("Topologie", ["RC", "RLC"])
            if type_filtre == "RC":
                R_f = st.slider("R (Ω)", 1.0, 1e6, 1000.0, 1.0)
                C_f = st.slider("C (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
                fc = engine.fc_RC(R_f, C_f)
                st.metric("f_c (Hz)", f"{fc:.2f}")
                st.metric("τ = RC (ms)", f"{R_f*C_f*1e3:.4f}")
                ftype_RC = st.radio("Type", ["LP", "HP"], horizontal=True)
            else:
                R_f = st.slider("R (Ω)", 1.0, 1000.0, 10.0, 0.1)
                L_f = st.slider("L (μH)", 0.1, 10000.0, 100.0, 0.1) * 1e-6
                C_f = st.slider("C (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
                f0 = engine.freq_resonance(L_f, C_f)
                Q = engine.facteur_Q(R_f, L_f, C_f)
                st.metric("f₀ (kHz)", f"{f0/1e3:.3f}")
                st.metric("Q", f"{Q:.3f}")
                st.metric("BW (Hz)", f"{f0/Q:.1f}")
                ftype_RLC = st.radio("Type", ["LP","HP","BP","BR"], horizontal=True)

        with col2:
            # Réduction de la résolution pour plus de rapidité
            f_arr = np.logspace(1, 7, 500)  # Réduit de 1000 à 500 points
            if type_filtre == "RC":
                H = engine.filtre_RC(R_f, C_f, f_arr, ftype_RC)
            else:
                H = engine.filtre_RLC(R_f, L_f, C_f, f_arr, ftype_RLC)

            mag_dB = 20 * np.log10(np.abs(H) + 1e-12)
            phase_deg = np.angle(H, deg=True)

            fig_filt = make_subplots(rows=2, cols=1,
                subplot_titles=["Gain (dB)", "Phase (°)"])
            fig_filt.add_trace(go.Scatter(x=f_arr, y=mag_dB, mode='lines',
                name='|H(f)|', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
            fig_filt.add_trace(go.Scatter(x=f_arr, y=phase_deg, mode='lines',
                name='∠H(f)', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
            fig_filt.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                               annotation_text="-3dB", row=1, col=1)
            fig_filt.add_hline(y=-45, line_color='#ffcc00', line_dash='dash',
                               annotation_text="-45°", row=2, col=1)

            fc_marker = engine.fc_RC(R_f, C_f) if type_filtre == "RC" \
                        else engine.freq_resonance(L_f, C_f)
            fig_filt.add_vline(x=fc_marker, line_color='#ff00cc', line_dash='dot',
                              annotation_text=f"f_c={fc_marker:.1f}Hz")

            fig_filt.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                                   color='#c0d0ff', title_text="f (Hz)")
            fig_filt.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_filt.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=480,  # Réduit de 520 à 480
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                showlegend=False  # Désactive la légende pour plus de rapidité
            )
            st.plotly_chart(fig_filt, use_container_width=True)

            # Réponse indicielle optimisée
            st.markdown("#### ⏱️ Réponse indicielle")
            t_ind = np.linspace(0, 5*R_f*C_f if type_filtre=="RC"
                                else 5/fc_marker*2*np.pi, 300)  # Réduit de 500 à 300 points
            y_ind = engine.reponse_impulsionnelle_filtre(R_f, C_f, t_ind)
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=t_ind*1e3, y=y_ind, mode='lines',
                line=dict(color='#00ccff', width=2.5), name='Vout/Vin'))
            fig_ind.add_hline(y=0.632, line_color='#ffcc00', line_dash='dash',
                              annotation_text="63.2% (t=τ)")
            fig_ind.update_layout(
                title="Réponse indicielle RC", xaxis_title="t (ms)",
                yaxis_title="Vout/Vin",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=240,  # Réduit de 280 à 240
                showlegend=False
            )
            st.plotly_chart(fig_ind, use_container_width=True)

    # ============================================================
    # TAB 2 : AOP
    # ============================================================
    with tab2:
        st.markdown("### 🔧 Amplificateurs Opérationnels")
        col1, col2 = st.columns([1, 2])

        with col1:
            config_aop = st.selectbox("Configuration", [
                "Inverseur", "Non-inverseur", "Différentiel",
                "Intégrateur", "Dérivateur", "Comparateur"
            ])
            R1_aop = st.slider("R₁ (kΩ)", 0.1, 1000.0, 10.0, 0.1) * 1e3
            Rf_aop = st.slider("Rf (kΩ)", 0.1, 1000.0, 100.0, 0.1) * 1e3
            GBW = st.slider("GBW (MHz)", 0.1, 100.0, 1.0, 0.1) * 1e6
            Vin_aop = st.slider("Vin (V)", -10.0, 10.0, 1.0, 0.1)
            Vsat = st.slider("Saturation ±Vsat (V)", 1.0, 15.0, 12.0, 0.5)

            if config_aop == "Inverseur":
                gain = engine.gain_inverseur(Rf_aop, R1_aop)
            elif config_aop == "Non-inverseur":
                gain = engine.gain_non_inverseur(Rf_aop, R1_aop)
            elif config_aop == "Différentiel":
                R3 = st.slider("R₃ (kΩ)", 0.1, 1000.0, 10.0, 0.1) * 1e3
                R4 = st.slider("R₄ (kΩ)", 0.1, 1000.0, 100.0, 0.1) * 1e3
                gain = engine.gain_differentiel(R1_aop, Rf_aop, R3, R4)
            else:
                gain = engine.gain_inverseur(Rf_aop, R1_aop)

            Vout_ideal = np.clip(gain * Vin_aop, -Vsat, Vsat)
            f_bande = GBW / max(abs(gain), 1)

            st.metric("Gain A_v", f"{gain:.3f}")
            st.metric("Gain (dB)", f"{20*np.log10(abs(gain)):.2f}" if gain != 0 else "0")
            st.metric("Vout (V)", f"{Vout_ideal:.3f}")
            st.metric("f_-3dB (kHz)", f"{f_bande/1e3:.2f}")
            st.metric("Saturation", "✅ NON" if abs(Vout_ideal) < Vsat else "⚠️ OUI")

        with col2:
            # Réduction de la résolution pour plus de rapidité
            f_bode = np.logspace(1, 7, 400)  # Réduit de 500 à 400 points
            H_aop = engine.bode_aop(gain, GBW, f_bode)
            mag_aop = 20 * np.log10(np.abs(H_aop) + 1e-12)
            phase_aop = np.angle(H_aop, deg=True)

            fig_aop = make_subplots(rows=2, cols=1,
                subplot_titles=["Gain (dB)", "Phase (°)"])
            fig_aop.add_trace(go.Scatter(x=f_bode, y=mag_aop, mode='lines',
                name='|A(f)|', line=dict(color='#00ccff', width=2.5)), row=1, col=1)
            fig_aop.add_trace(go.Scatter(x=f_bode, y=phase_aop, mode='lines',
                name='∠A(f)', line=dict(color='#7700ff', width=2.5)), row=2, col=1)
            fig_aop.add_hline(y=20*np.log10(abs(gain))-3,
                              line_color='#ffcc00', line_dash='dash',
                              annotation_text="-3dB", row=1, col=1)
            fig_aop.add_vline(x=f_bande, line_color='#ff00cc', line_dash='dot',
                              annotation_text=f"f_c={f_bande:.0f}Hz")

            fig_aop.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                                  color='#c0d0ff', title_text="f (Hz)")
            fig_aop.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_aop.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=450,  # Réduit de 500 à 450
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                showlegend=False
            )
            st.plotly_chart(fig_aop, use_container_width=True)

            # Transfert Vin → Vout optimisé
            Vin_sweep = np.linspace(-Vsat, Vsat, 200)  # Réduit de 500 à 200 points
            Vout_sweep = np.clip(gain * Vin_sweep, -Vsat, Vsat)
            fig_vt = go.Figure()
            fig_vt.add_trace(go.Scatter(x=Vin_sweep, y=Vout_sweep, mode='lines',
                line=dict(color='#00ccff', width=3), name='Vout(Vin)'))
            fig_vt.add_hline(y=Vsat, line_color='#ff4444', line_dash='dot',
                             annotation_text=f"+Vsat={Vsat}V")
            fig_vt.add_hline(y=-Vsat, line_color='#ff4444', line_dash='dot',
                             annotation_text=f"-Vsat={-Vsat}V")
            fig_vt.update_layout(
                title="Transfert Vout = f(Vin)",
                xaxis_title="Vin (V)", yaxis_title="Vout (V)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=240,  # Réduit de 300 à 240
                showlegend=False
            )
            st.plotly_chart(fig_vt, use_container_width=True)

    # ============================================================
    # TAB 3 : DIODES
    # ============================================================
    with tab3:
        st.markdown("### 💡 Diodes — Modèle de Shockley")
        col1, col2 = st.columns([1, 2])

        with col1:
            Is = st.slider("Is (pA)", 0.001, 100.0, 1.0, 0.01) * 1e-12
            n_id = st.slider("Idéalité n", 1.0, 2.0, 1.0, 0.01)
            T_diode = st.slider("T (K)", 250.0, 400.0, 300.0, 1.0)
            Vcc_d = st.slider("Vcc (V)", 0.5, 20.0, 5.0, 0.1)
            R_diode = st.slider("R série (Ω)", 1.0, 10000.0, 1000.0, 1.0)

            VT = 1.3806e-23 * T_diode / 1.6022e-19
            st.metric("VT (mV)", f"{VT*1000:.2f}")

            pf = engine.point_fonctionnement_diode(Vcc_d, R_diode, Is)
            st.metric("Vd (V)", f"{pf['V_d']:.4f}")
            st.metric("Id (mA)", f"{pf['I_d']*1000:.4f}")
            st.metric("P_diode (mW)", f"{pf['P_diode']*1000:.4f}")
            st.metric("P_résistance (mW)", f"{pf['P_resistor']*1000:.4f}")

        with col2:
            V_range = np.linspace(-1.0, 1.0, 1000)
            I_diode = engine.courant_diode(V_range, Is, n_id, T_diode)
            I_circuit = (Vcc_d - V_range) / R_diode

            fig_diode = go.Figure()
            fig_diode.add_trace(go.Scatter(x=V_range, y=I_diode*1000, mode='lines',
                name='I_diode (mA)', line=dict(color='#00ccff', width=3)))
            fig_diode.add_trace(go.Scatter(x=V_range, y=I_circuit*1000, mode='lines',
                name='Droite de charge', line=dict(color='#ffcc00', width=2.5,
                dash='dash')))
            fig_diode.add_trace(go.Scatter(x=[pf['V_d']], y=[pf['I_d']*1000],
                mode='markers', name='Point Q',
                marker=dict(color='#ff00cc', size=14, symbol='star')))

            fig_diode.update_layout(
                title=f"Caractéristique I-V diode + droite de charge",
                xaxis_title="V (V)", yaxis_title="I (mA)",
                yaxis=dict(range=[-1, Vcc_d/R_diode*1000*1.1]),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=430,
            )
            st.plotly_chart(fig_diode, use_container_width=True)

            # Effet température
            st.markdown("#### 🌡️ Effet de la température")
            T_arr = [250, 300, 350, 400]
            V_range2 = np.linspace(0, 0.9, 500)
            fig_T = go.Figure()
            colors_T = ['#00ccff','#7700ff','#ff00cc','#ffcc00']
            for Ti, ci in zip(T_arr, colors_T):
                Ii = engine.courant_diode(V_range2, Is, n_id, Ti)
                fig_T.add_trace(go.Scatter(x=V_range2, y=np.log10(Ii+1e-15),
                    mode='lines', name=f'T={Ti}K',
                    line=dict(color=ci, width=2)))
            fig_T.update_layout(
                title="log₁₀(I) vs V pour différentes T",
                xaxis_title="V (V)", yaxis_title="log₁₀(I) (A)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
            )
            st.plotly_chart(fig_T, use_container_width=True)

    # ============================================================
    # TAB 4 : TRANSISTORS BJT
    # ============================================================
    with tab4:
        st.markdown("### 🔦 Transistor BJT")
        col1, col2 = st.columns([1, 2])

        with col1:
            Vcc_bjt = st.slider("Vcc (V)", 1.0, 30.0, 12.0, 0.5)
            Rb_bjt = st.slider("Rb (kΩ)", 1.0, 10000.0, 470.0, 1.0) * 1e3
            Rc_bjt = st.slider("Rc (kΩ)", 0.1, 10.0, 1.0, 0.1) * 1e3
            beta = st.slider("β (hFE)", 10, 500, 100)

            pq = engine.point_repos_BJT(Vcc_bjt, Rb_bjt, Rc_bjt, beta)
            st.metric("IB (μA)", f"{pq['IB_uA']:.3f}")
            st.metric("IC (mA)", f"{pq['IC_mA']:.3f}")
            st.metric("VCE (V)", f"{pq['VCE']:.3f}")
            st.metric("Régime", pq["regime"])
            st.metric("Gain Av ≈", f"{-beta*Rc_bjt/((1+beta)*26e-3):.1f}")

        with col2:
            # Famille de caractéristiques IC = f(VCE)
            VCE_arr = np.linspace(0, Vcc_bjt, 300)
            IB_list = np.linspace(1, 100, 6) * 1e-6
            V_dc, I_dc = engine.droite_charge(Vcc_bjt, Rc_bjt)

            fig_bjt = go.Figure()
            colors_b = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00','#ff4400']

            for i, IB in enumerate(IB_list):
                IC = beta * IB * np.ones_like(VCE_arr)
                IC_sat = np.minimum(IC, (Vcc_bjt - VCE_arr) / Rc_bjt)
                IC_sat = np.where(VCE_arr < 0.2, VCE_arr/0.2*beta*IB, IC_sat)
                fig_bjt.add_trace(go.Scatter(
                    x=VCE_arr, y=IC_sat*1000, mode='lines',
                    name=f'IB={IB*1e6:.0f}μA',
                    line=dict(color=colors_b[i], width=2)
                ))

            fig_bjt.add_trace(go.Scatter(x=V_dc, y=I_dc*1000, mode='lines',
                name='Droite de charge',
                line=dict(color='#ffffff', width=2.5, dash='dash')))
            fig_bjt.add_trace(go.Scatter(
                x=[pq['VCE']], y=[pq['IC_mA']], mode='markers',
                name='Point Q',
                marker=dict(color='#ff00cc', size=14, symbol='star')
            ))
            fig_bjt.update_layout(
                title="Famille de caractéristiques BJT",
                xaxis_title="VCE (V)", yaxis_title="IC (mA)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=430,
            )
            st.plotly_chart(fig_bjt, use_container_width=True)

    # ============================================================
    # TAB 5 : OSCILLATEURS & BRUIT
    # ============================================================
    with tab5:
        st.markdown("### 📻 Oscillateurs & Bruit thermique")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 🔁 Oscillateur Colpitts")
            L_col = st.slider("L (μH)", 0.1, 1000.0, 10.0, 0.1) * 1e-6
            C1_col = st.slider("C₁ (nF)", 0.1, 10000.0, 100.0, 0.1) * 1e-9
            C2_col = st.slider("C₂ (nF)", 0.1, 10000.0, 47.0, 0.1) * 1e-9

            col_res = engine.oscillateur_colpitts(L_col, C1_col, C2_col)
            st.metric("f₀ (MHz)", f"{col_res['f0_MHz']:.4f}")
            st.metric("C_eq (nF)", f"{col_res['C_eq_nF']:.4f}")
            st.metric("Rapport C₁/C₂", f"{col_res['rapport_C1_C2']:.3f}")

            st.markdown("#### 🔊 Bruit thermique")
            R_bruit = st.slider("R (kΩ)", 0.01, 1000.0, 10.0, 0.1) * 1e3
            T_bruit = st.slider("T (K)", 77.0, 500.0, 300.0, 1.0)
            BW_bruit = st.slider("BW (kHz)", 0.1, 10000.0, 100.0, 1.0) * 1e3

            Vn = engine.bruit_thermique(R_bruit, T_bruit, BW_bruit)
            V_sig = st.slider("V_signal (μV)", 0.1, 1000.0, 100.0, 0.1) * 1e-6
            SNR = engine.SNR_db(V_sig, Vn)

            st.metric("Vn_rms (nV/√Hz)", f"{engine.bruit_thermique(R_bruit, T_bruit, 1)*1e9:.3f}")
            st.metric("Vn total (μV)", f"{Vn*1e6:.3f}")
            st.metric("SNR (dB)", f"{SNR:.2f}")

        with col2:
            # Densité spectrale de bruit
            f_bruit = np.logspace(1, 8, 500)
            Sn_V = 4 * 1.3806e-23 * T_bruit * R_bruit * np.ones_like(f_bruit)

            fig_bruit = go.Figure()
            fig_bruit.add_trace(go.Scatter(
                x=f_bruit, y=np.sqrt(Sn_V)*1e9, mode='lines',
                name=f'DSP bruit ({R_bruit/1e3:.0f}kΩ)',
                line=dict(color='#00ccff', width=2.5),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
            ))
            fig_bruit.add_vline(x=BW_bruit, line_color='#ffcc00', line_dash='dash',
                                annotation_text=f"BW={BW_bruit/1e3:.0f}kHz")
            fig_bruit.update_layout(
                title="Densité spectrale de bruit thermique",
                xaxis_title="f (Hz)", yaxis_title="Vn (nV/√Hz)",
                xaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=360,
            )
            st.plotly_chart(fig_bruit, use_container_width=True)

            # Signal oscillateur simulé
            t_osc = np.linspace(0, 5/col_res["f0_MHz"]/1e6, 2000)
            f0 = col_res["f0_MHz"] * 1e6
            y_osc = np.cos(2*np.pi*f0*t_osc)
            y_noisy = y_osc + np.random.normal(0, 0.05, len(t_osc))
            fig_osc = go.Figure()
            fig_osc.add_trace(go.Scatter(x=t_osc*1e6, y=y_osc, mode='lines',
                name='Signal pur', line=dict(color='#00ccff', width=2)))
            fig_osc.add_trace(go.Scatter(x=t_osc*1e6, y=y_noisy, mode='lines',
                name='Avec bruit', line=dict(color='rgba(119,0,255,0.5)', width=1)))
            fig_osc.update_layout(
                title=f"Oscillateur Colpitts — f₀={col_res['f0_MHz']:.4f} MHz",
                xaxis_title="t (μs)", yaxis_title="V (u.a.)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=300,
            )
            st.plotly_chart(fig_osc, use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire électronique analogique")
        for nom, formule in FORMULES.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)
        st.markdown("---")
        st.markdown("### 🔬 Composants standards")
        df_comp = pd.DataFrame([
            {"Type": "Résistances (E24)", "Valeurs": "10, 11, 12, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91"},
            {"Type": "Condensateurs (pF)", "Valeurs": "10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100, 120, 150, 180, 220, 270, 330, 390, 470"},
            {"Type": "Inductances (μH)", "Valeurs": "0.1, 0.22, 0.47, 1, 2.2, 4.7, 10, 22, 47, 100, 220, 470, 1000"},
        ])
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("---")
        for r in ["Razavi — *Design of Analog CMOS Integrated Circuits* (McGraw-Hill, 2017)",
                  "Sedra & Smith — *Microelectronic Circuits* (Oxford, 2020)",
                  "Horowitz & Hill — *The Art of Electronics* (Cambridge, 2015)"]:
            st.markdown(f"- {r}")