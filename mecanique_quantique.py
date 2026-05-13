import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import eigh, expm
from scipy.special import hermite, eval_hermite, factorial
from scipy.optimize import brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES PHYSIQUES
# ============================================================
CONSTANTES = {
    "ħ (J·s)":          1.0546e-34,
    "h (J·s)":          6.6261e-34,
    "m_e (kg)":         9.1094e-31,
    "e (C)":            1.6022e-19,
    "ε₀ (F/m)":         8.8542e-12,
    "a₀ (m)":           5.2918e-11,
    "E_Hartree (eV)":   27.211,
    "E_Rydberg (eV)":   13.606,
    "k_B (J/K)":        1.3806e-23,
    "c (m/s)":          2.9979e8,
}

FORMULES = {
    "Équation de Schrödinger":   r"i\hbar\frac{\partial\psi}{\partial t}=\hat{H}\psi=\left[-\frac{\hbar^2}{2m}\nabla^2+V\right]\psi",
    "Hamiltonien":               r"\hat{H}=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x)",
    "Puits infini (E_n)":        r"E_n=\frac{n^2\pi^2\hbar^2}{2mL^2},\quad n=1,2,3\ldots",
    "Puits infini (ψ_n)":        r"\psi_n(x)=\sqrt{\frac{2}{L}}\sin\!\left(\frac{n\pi x}{L}\right)",
    "Oscillateur harmonique":    r"E_n=\hbar\omega\!\left(n+\frac{1}{2}\right)",
    "Atome d'hydrogène":         r"E_n=-\frac{13.6\text{ eV}}{n^2},\quad n=1,2,\ldots",
    "Effet tunnel":              r"T\approx e^{-2\kappa L},\quad\kappa=\sqrt{\frac{2m(V_0-E)}{\hbar^2}}",
    "Règle de Born":             r"P(x)=|\psi(x)|^2,\quad\int_{-\infty}^{\infty}|\psi|^2dx=1",
    "Incertitude Heisenberg":    r"\Delta x\cdot\Delta p\geq\frac{\hbar}{2}",
    "Opérateur quantité de mvt": r"\hat{p}=-i\hbar\frac{\partial}{\partial x}",
    "Valeur moyenne":            r"\langle A\rangle=\int\psi^*\hat{A}\psi\,dx",
    "Fonction d'onde H":         r"\psi_{nlm}=R_{nl}(r)Y_l^m(\theta,\phi)",
}

def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    dx = np.diff(x)
    return np.sum((y[:-1] + y[1:]) / 2 * dx)


# ============================================================
# MOTEUR MÉCANIQUE QUANTIQUE
# ============================================================
class QuantumEngine:
    """Moteur de calcul en mécanique quantique."""

    def __init__(self, m: float = 9.1094e-31, hbar: float = 1.0546e-34):
        self.m = m
        self.hbar = hbar

    # --- Puits infini ---
    def puits_infini_energie(self, n: int, L: float) -> float:
        return n**2 * np.pi**2 * self.hbar**2 / (2 * self.m * L**2)

    def puits_infini_psi(self, n: int, x: np.ndarray, L: float) -> np.ndarray:
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L) * (x >= 0) * (x <= L)

    @st.cache_data
    def puits_infini_multi(_self, x: np.ndarray, L: float,
                            n_max: int = 5) -> tuple:
        energies = []
        psis = []
        for n in range(1, n_max+1):
            En = _self.puits_infini_energie(n, L)
            psi = _self.puits_infini_psi(n, x, L)
            energies.append(En)
            psis.append(psi)
        return np.array(energies), np.array(psis)

    # --- Oscillateur harmonique ---
    def osc_harm_energie(self, n: int, omega: float) -> float:
        return self.hbar * omega * (n + 0.5)

    @st.cache_data
    def osc_harm_psi(_self, n: int, x: np.ndarray, omega: float) -> np.ndarray:
        alpha = np.sqrt(_self.m * omega / _self.hbar)
        xi = alpha * x
        Hn = eval_hermite(n, xi)
        N = (1/(np.sqrt(2**n * float(factorial(n)))) *
             (alpha/np.pi)**0.25)
        return N * Hn * np.exp(-xi**2/2)

    def osc_harm_potentiel(self, x: np.ndarray, omega: float) -> np.ndarray:
        return 0.5 * self.m * omega**2 * x**2

    # --- Barrière rectangulaire (effet tunnel) ---
    @st.cache_data
    def transmittance_tunnel(_self, E_arr: np.ndarray,
                              V0: float, L: np.ndarray | float) -> np.ndarray:
        E_arr = np.asarray(E_arr, dtype=float)
        L_arr = np.asarray(L, dtype=float)

        if L_arr.shape == ():
            L_arr = np.full_like(E_arr, L_arr)
        elif L_arr.shape != E_arr.shape:
            raise ValueError("L doit être scalaire ou avoir la même forme que E_arr")

        T = np.zeros_like(E_arr, dtype=float)
        mask = E_arr >= V0

        if np.any(mask):
            E_pos = E_arr[mask]
            L_pos = L_arr[mask]
            k1 = np.sqrt(2*_self.m*E_pos) / _self.hbar
            k2 = np.sqrt(2*_self.m*(E_pos - V0)) / _self.hbar
            sin_term = np.sin(k2 * L_pos)
            safe_mask = (k1 != 0) & (k2 != 0)
            denom_pos = 1 + (k1**2 - k2**2)**2 / (4 * k1**2 * k2**2 + 1e-30) * sin_term**2
            T[mask] = np.where(
                safe_mask,
                1.0 / denom_pos,
                np.where(k1 != 0, 1.0 / (1 + (V0 / (2 * E_pos))**2 * k1**2 * L_pos**2), 0.0)
            )

        if np.any(~mask):
            E_neg = E_arr[~mask]
            L_neg = L_arr[~mask]
            k1 = np.sqrt(2*_self.m*E_neg) / _self.hbar
            kappa = np.sqrt(2*_self.m*(V0 - E_neg)) / _self.hbar
            kL = kappa * L_neg
            sinh2 = np.where(kL < 500, np.sinh(kL)**2, np.exp(2*kL) / 4)
            denom_neg = 1 + (k1**2 + kappa**2)**2 / (4 * k1**2 * kappa**2 + 1e-30) * sinh2
            T[~mask] = np.where((k1 == 0) | (kappa == 0), 0.0, 1.0 / denom_neg)

        return np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

    def paquet_onde(self, x: np.ndarray, x0: float, sigma: float,
                    k0: float, t: float, omega_k: float = None) -> np.ndarray:
        """Paquet d'onde gaussien en propagation libre."""
        hbar = self.hbar
        m = self.m
        sigma_t = np.sqrt(sigma**2 + (hbar*t/(2*m*sigma))**2) if t != 0 else sigma
        phase = k0*x - (hbar*k0**2/(2*m))*t
        envelope = np.exp(-(x-x0-hbar*k0*t/m)**2/(4*sigma_t**2))
        norm = (2*np.pi*sigma_t**2)**0.25
        psi = (1/norm) * envelope * np.exp(1j*phase)
        return psi

    # --- Atome d'hydrogène ---
    def H_energie(self, n: int) -> float:
        """Énergie en Joules."""
        E_R = 13.6 * 1.6022e-19  # Rydberg en J
        return -E_R / n**2

    def H_energie_eV(self, n: int) -> float:
        return -13.6 / n**2

    def H_rayon_moyen(self, n: int, l: int) -> float:
        """Rayon moyen <r> en unités de a₀."""
        a0 = 5.2918e-11
        return a0 * (3*n**2 - l*(l+1)) / 2

    def H_psi_1s(self, r: np.ndarray) -> np.ndarray:
        """Fonction d'onde 1s normalisée."""
        a0 = 5.2918e-11
        return (1/np.sqrt(np.pi)) * (1/a0)**1.5 * np.exp(-r/a0)

    def H_densite_radiale(self, n: int, l: int,
                           r: np.ndarray) -> np.ndarray:
        """Densité de probabilité radiale P(r) = r²|R_nl|²."""
        a0 = 5.2918e-11
        rho = 2 * r / (n * a0)

        # Éviter les valeurs négatives ou nulles pour rho
        rho = np.maximum(rho, 1e-10)

        if n == 1 and l == 0:
            R = 2 * (1/a0)**1.5 * np.exp(-r/a0)
        elif n == 2 and l == 0:
            R = (1/(2*np.sqrt(2))) * (1/a0)**1.5 * (2-rho) * np.exp(-rho/2)
        elif n == 2 and l == 1:
            R = (1/(2*np.sqrt(6))) * (1/a0)**1.5 * rho * np.exp(-rho/2)
        elif n == 3 and l == 0:
            R = (2/(81*np.sqrt(3))) * (1/a0)**1.5 * (27-18*rho+2*rho**2) * np.exp(-rho/3)
        else:
            R = np.exp(-rho/n) / n

        # Éviter les NaN et inf
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        return r**2 * np.abs(R)**2

    # --- Quantification numérique (FDM) ---
    def resoudre_schrodinger_1d(self, x: np.ndarray,
                                  V: np.ndarray, n_states: int = 6) -> tuple:
        """Résolution numérique par différences finies."""
        N = len(x)
        dx = x[1] - x[0]
        diag = self.hbar**2 / (self.m * dx**2) + V
        off = -self.hbar**2 / (2 * self.m * dx**2) * np.ones(N-1)
        H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        eigenvalues, eigenvectors = eigh(H, subset_by_index=[0, min(n_states-1, N-2)])
        return eigenvalues, eigenvectors.T

    # --- Evolution temporelle ---
    def evolution_temporelle(self, psi0: np.ndarray,
                              x: np.ndarray, V: np.ndarray,
                              t_arr: np.ndarray) -> np.ndarray:
        """Evolution par exponentielle matricielle."""
        N = len(x)
        dx = x[1] - x[0]
        diag = self.hbar**2 / (self.m * dx**2) + V
        off = -self.hbar**2 / (2 * self.m * dx**2) * np.ones(N-1)
        H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        psi_t = []
        for t in t_arr:
            U_t = expm(-1j * H * t / self.hbar)
            psi_t.append(U_t @ psi0)
        return np.array(psi_t)

    # --- Valeurs moyennes ---
    def valeur_moyenne_x(self, psi: np.ndarray, x: np.ndarray) -> float:
        dx = x[1] - x[0]
        return np.real(np.sum(np.conj(psi) * x * psi) * dx)

    def incertitude_x(self, psi: np.ndarray, x: np.ndarray) -> float:
        dx = x[1] - x[0]
        x_moy = self.valeur_moyenne_x(psi, x)
        x2_moy = np.real(np.sum(np.conj(psi) * x**2 * psi) * dx)
        return np.sqrt(x2_moy - x_moy**2)

    def incertitude_p(self, psi: np.ndarray, x: np.ndarray) -> float:
        dx = x[1] - x[0]
        dpsi = np.gradient(psi, dx)
        # Éviter les NaN et inf dans les calculs
        dpsi = np.nan_to_num(dpsi, nan=0.0, posinf=0.0, neginf=0.0)
        p_moy = np.real(np.sum(np.conj(psi) * (-1j*self.hbar) * dpsi) * dx)
        p2_moy = np.real(np.sum(np.conj(psi) * (-self.hbar**2) *
                                np.gradient(dpsi, dx)) * dx)
        return np.sqrt(max(p2_moy - p_moy**2, 0))


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def mecanique_quantique_page():
    st.markdown("## ⚛️ Mécanique Quantique Avancée")
    st.markdown("*Puits quantiques, oscillateur, effet tunnel, atome H, evolution temporelle*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📦 Puits quantique",
        "〰️ Oscillateur harmonique",
        "🚧 Effet tunnel",
        "🔬 Atome d'hydrogène",
        "🕐 Évolution temporelle",
        "📖 Théorie"
    ])

    engine = QuantumEngine()

    # ============================================================
    # TAB 1 : PUITS INFINI
    # ============================================================
    with tab1:
        st.markdown("### 📦 Puits de potentiel infini")
        col1, col2 = st.columns([1, 2])

        with col1:
            L_nm = st.slider("Largeur L (nm)", 0.1, 10.0, 1.0, 0.1)
            L = L_nm * 1e-9
            n_max = st.slider("États (n_max)", 1, 8, 4)
            show_prob = st.checkbox("Afficher |ψ|²", True)
            show_energie = st.checkbox("Niveaux d'énergie", True)

            x = np.linspace(0, L, 300)  # Réduit de 500 à 300 points
            energies, psis = engine.puits_infini_multi(x, L, n_max)

            st.markdown("### 📐 Niveaux d'énergie")
            for n in range(1, n_max+1):
                E_J = energies[n-1]
                E_eV = E_J / 1.6022e-19
                st.metric(f"E_{n} (eV)", f"{E_eV:.4f}")

        with col2:
            fig_pw = go.Figure()
            colors_q = ['#00ccff','#7700ff','#ff00cc','#00ff88',
                        '#ffcc00','#ff4400','#88ccff','#cc88ff']

            for n in range(n_max):
                En_eV = energies[n] / 1.6022e-19
                y_plot = psis[n] if not show_prob else psis[n]**2
                scale = 0.3 * En_eV
                fig_pw.add_trace(go.Scatter(
                    x=x*1e9, y=y_plot*scale + En_eV,
                    mode='lines', name=f'n={n+1} ({En_eV:.3f}eV)',
                    line=dict(color=colors_q[n], width=2.5)
                ))
                if show_energie:
                    fig_pw.add_hline(y=En_eV,
                        line_color=f'rgba({int(colors_q[n][1:3],16)},'
                                   f'{int(colors_q[n][3:5],16)},'
                                   f'{int(colors_q[n][5:7],16)},0.3)',
                        line_dash='dot')

            # Parois
            fig_pw.add_vrect(x0=-0.5, x1=0, fillcolor='rgba(119,0,255,0.3)',
                             line_width=0)
            fig_pw.add_vrect(x0=L*1e9, x1=L*1e9+0.5,
                             fillcolor='rgba(119,0,255,0.3)', line_width=0)

            fig_pw.update_layout(
                title=f"Puits infini L={L_nm} nm — {n_max} états",
                xaxis_title="x (nm)",
                yaxis_title="|ψ|² + E (eV)" if show_prob else "ψ + E (eV)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=450,  # Réduit de 500 à 450
                showlegend=False  # Désactive la légende pour plus de rapidité
            )
            st.plotly_chart(fig_pw, use_container_width=True)

            # Résolution numérique avec potentiel personnalisé
            st.markdown("#### 🔢 Résolution numérique (FDM)")
            pot_type = st.selectbox("Potentiel", [
                "Puits infini", "Puits fini", "Double puits", "Harmonique"
            ])
            V0_num = st.slider("V₀ (eV)", 0.1, 10.0, 2.0, 0.1) * 1.6022e-19

            x_num = np.linspace(-L, 2*L, 150)  # Réduit de 200 à 150 points
            if pot_type == "Puits infini":
                V_num = np.where((x_num >= 0) & (x_num <= L), 0, 1e10*1.6022e-19)
            elif pot_type == "Puits fini":
                V_num = np.where((x_num >= 0) & (x_num <= L), 0, V0_num)
            elif pot_type == "Double puits":
                w = L/4
                V_num = V0_num * np.ones_like(x_num)
                V_num[(x_num > 0) & (x_num < w)] = 0
                V_num[(x_num > 3*w) & (x_num < 4*w)] = 0
            else:
                V_num = 0.5 * engine.m * (1e14)**2 * x_num**2

            evals, evecs = engine.resoudre_schrodinger_1d(x_num, V_num,
                                                            n_states=min(n_max,4))
            fig_fdm = go.Figure()
            V_eV = np.clip(V_num/1.6022e-19, 0, 15)
            fig_fdm.add_trace(go.Scatter(x=x_num*1e9, y=V_eV, mode='lines',
                name='V(x)', line=dict(color='rgba(255,255,255,0.5)', width=2)))
            for i, (E, psi) in enumerate(zip(evals, evecs)):
                E_eV = E/1.6022e-19
                if 0 < E_eV < 20:
                    y = np.real(psi)**2 * 2 + E_eV
                    fig_fdm.add_trace(go.Scatter(x=x_num*1e9, y=y, mode='lines',
                        name=f'E{i+1}={E_eV:.3f}eV',
                        line=dict(color=colors_q[i], width=2)))
            fig_fdm.update_layout(
                title=f"FDM — {pot_type}",
                xaxis_title="x (nm)", yaxis_title="E (eV)",
                yaxis=dict(range=[0, 12], color='#c0d0ff',
                          gridcolor='rgba(100,0,255,0.2)'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
            )
            st.plotly_chart(fig_fdm, use_container_width=True)

    # ============================================================
    # TAB 2 : OSCILLATEUR HARMONIQUE
    # ============================================================
    with tab2:
        st.markdown("### 〰️ Oscillateur harmonique quantique")
        col1, col2 = st.columns([1, 2])

        with col1:
            omega_hz = st.slider("ω (×10¹³ rad/s)", 0.1, 10.0, 1.0, 0.1)
            omega = omega_hz * 1e13
            n_osc = st.slider("États n_max", 1, 7, 4)
            x_max_osc = st.slider("x_max (pm)", 10, 1000, 200)

            x_osc = np.linspace(-x_max_osc*1e-12, x_max_osc*1e-12, 600)
            V_osc = engine.osc_harm_potentiel(x_osc, omega)

            st.markdown("### 📐 Niveaux d'énergie")
            for n in range(n_osc):
                En = engine.osc_harm_energie(n, omega) / 1.6022e-19
                st.metric(f"E_{n} (eV)", f"{En:.4f}")

        with col2:
            fig_osc = go.Figure()
            V_eV = V_osc / 1.6022e-19
            fig_osc.add_trace(go.Scatter(
                x=x_osc*1e12, y=np.clip(V_eV, 0, 5), mode='lines',
                name='V(x)', line=dict(color='rgba(255,255,255,0.5)', width=2)
            ))

            for n in range(n_osc):
                psi_n = engine.osc_harm_psi(n, x_osc, omega)
                En_eV = engine.osc_harm_energie(n, omega) / 1.6022e-19
                y_plot = np.real(psi_n)**2 * 0.4 + En_eV
                fig_osc.add_trace(go.Scatter(
                    x=x_osc*1e12, y=y_plot, mode='lines',
                    name=f'n={n} (E={En_eV:.3f}eV)',
                    line=dict(color=colors_q[n % len(colors_q)], width=2.5)
                ))
                fig_osc.add_hline(y=En_eV,
                    line_color=f'rgba({int(colors_q[n%8][1:3],16)},'
                               f'{int(colors_q[n%8][3:5],16)},'
                               f'{int(colors_q[n%8][5:7],16)},0.3)',
                    line_dash='dot')

            fig_osc.update_layout(
                title="Oscillateur harmonique quantique",
                xaxis_title="x (pm)", yaxis_title="E (eV)",
                yaxis=dict(range=[0, 5]),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig_osc, use_container_width=True)

    # ============================================================
    # TAB 3 : EFFET TUNNEL
    # ============================================================
    with tab3:
        st.markdown("### 🚧 Effet tunnel quantique")
        col1, col2 = st.columns([1, 2])

        with col1:
            V0_eV = st.slider("Hauteur V₀ (eV)", 0.1, 20.0, 5.0, 0.1)
            L_bar = st.slider("Épaisseur L (nm)", 0.01, 5.0, 0.5, 0.01)
            V0 = V0_eV * 1.6022e-19
            L_b = L_bar * 1e-9

            E_arr = np.linspace(0.01, V0_eV * 1.5, 500) * 1.6022e-19
            T_arr = engine.transmittance_tunnel(E_arr, V0, L_b)

            st.metric("T(E=V₀/2)", f"{engine.transmittance_tunnel(np.array([V0/2]), V0, L_b)[0]:.4e}")
            st.metric("T(E=V₀)", f"{engine.transmittance_tunnel(np.array([V0]), V0, L_b)[0]:.4f}")

        with col2:
            fig_tun = go.Figure()
            E_eV = E_arr / 1.6022e-19
            fig_tun.add_trace(go.Scatter(x=E_eV, y=T_arr, mode='lines',
                name='T(E)', line=dict(color='#00ccff', width=3)))
            fig_tun.add_vline(x=V0_eV, line_color='#ffcc00', line_dash='dash',
                              annotation_text=f"V₀={V0_eV} eV")
            fig_tun.add_hline(y=1, line_color='rgba(255,255,255,0.3)', line_dash='dot')
            fig_tun.update_layout(
                title=f"Transmittance tunnel — V₀={V0_eV}eV, L={L_bar}nm",
                xaxis_title="E (eV)", yaxis_title="T(E)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)',
                          type='log'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
            )
            st.plotly_chart(fig_tun, use_container_width=True)

            # T vs L
            L_arr = np.linspace(0.01, 5, 300) * 1e-9
            E_fixed = V0 * 0.5
            T_L = engine.transmittance_tunnel(np.full(len(L_arr), E_fixed), V0, L_arr)
            fig_tL = go.Figure()
            fig_tL.add_trace(go.Scatter(x=L_arr*1e9, y=T_L, mode='lines',
                name='T(L) pour E=V₀/2',
                line=dict(color='#7700ff', width=2.5)))
            fig_tL.update_layout(
                title="Transmittance vs épaisseur (E=V₀/2)",
                xaxis_title="L (nm)", yaxis_title="T", yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=300,
            )
            st.plotly_chart(fig_tL, use_container_width=True)

    # ============================================================
    # TAB 4 : ATOME D'HYDROGÈNE
    # ============================================================
    with tab4:
        st.markdown("### 🔬 Atome d'hydrogène")
        col1, col2 = st.columns([1, 2])

        with col1:
            n_H = st.slider("Nombre quantique n", 1, 5, 2)
            l_H = st.slider("Nombre quantique l", 0, n_H-1, min(1, n_H-1))

            E_eV = engine.H_energie_eV(n_H)
            r_moy = engine.H_rayon_moyen(n_H, l_H)
            a0 = 5.2918e-11

            st.metric(f"E_{n_H} (eV)", f"{E_eV:.4f}")
            st.metric("r moyen (nm)", f"{r_moy*1e9:.4f}")
            st.metric("r moyen (a₀)", f"{r_moy/a0:.2f}")
            st.metric("λ_Lyman (nm)",
                      f"{1240/(13.6*(1-1/n_H**2)):.3f}" if n_H > 1 else "N/A")

            st.markdown("### 📊 Série de Rydberg")
            for n in range(1, 6):
                En = engine.H_energie_eV(n)
                delta = 13.6*(1-1/n**2) if n > 1 else 0
                st.markdown(f"- n={n}: **{En:.3f} eV** | Lyman: {1240/delta:.1f}nm" if n>1
                            else f"- n={n}: **{En:.3f} eV** (état fondamental)")

        with col2:
            r_max = 30 * a0
            r_arr = np.linspace(0.001*a0, r_max, 1000)
            P_rad = engine.H_densite_radiale(n_H, l_H, r_arr)
            P_rad_norm = P_rad / (trapezoid_integral(P_rad, r_arr) + 1e-30)

            fig_H = make_subplots(rows=2, cols=1,
                subplot_titles=[f"Densité radiale P(r) — n={n_H}, l={l_H}",
                                 "Niveaux d'énergie (eV)"])

            fig_H.add_trace(go.Scatter(
                x=r_arr/a0, y=P_rad_norm*a0, mode='lines',
                name=f'P_{n_H}{l_H}(r)',
                line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
            ), row=1, col=1)
            fig_H.add_vline(x=r_moy/a0, line_color='#ffcc00', line_dash='dash',
                            annotation_text=f"<r>={r_moy/a0:.1f}a₀", row=1, col=1)

            # Niveaux d'énergie
            for n in range(1, 6):
                En = engine.H_energie_eV(n)
                fig_H.add_trace(go.Scatter(
                    x=[0, 1], y=[En, En], mode='lines',
                    name=f'n={n} ({En:.2f}eV)',
                    line=dict(color=colors_q[n-1], width=2.5)
                ), row=2, col=1)

            fig_H.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=580,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_H.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_H.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_H.update_xaxes(title_text="r/a₀", row=1, col=1)
            fig_H.update_xaxes(title_text="", row=2, col=1)
            fig_H.update_yaxes(title_text="P(r)·a₀", row=1, col=1)
            fig_H.update_yaxes(title_text="E (eV)", row=2, col=1)
            st.plotly_chart(fig_H, use_container_width=True)

    # ============================================================
    # TAB 5 : ÉVOLUTION TEMPORELLE
    # ============================================================
    with tab5:
        st.markdown("### 🕐 Paquet d'onde & Évolution temporelle")
        col1, col2 = st.columns([1, 2])

        with col1:
            x0_pm = st.slider("x₀ (pm)", -500, 0, -200)
            sigma_pm = st.slider("σ (pm)", 10, 200, 50)
            k0_inv = st.slider("k₀ (nm⁻¹)", 1.0, 50.0, 10.0, 0.5)
            t_fs = st.slider("Temps t (fs)", 0.0, 100.0, 0.0, 0.5)

            x0 = x0_pm * 1e-12
            sigma = sigma_pm * 1e-12
            k0 = k0_inv * 1e9
            t = t_fs * 1e-15

            x_wp = np.linspace(-1000e-12, 1000e-12, 1000)
            psi_wp = engine.paquet_onde(x_wp, x0, sigma, k0, t)

            # Heisenberg
            dx_H = sigma
            dp_H = engine.hbar / (2 * sigma)
            prod = dx_H * dp_H

            st.metric("Δx (pm)", f"{dx_H*1e12:.2f}")
            st.metric("Δp (kg·m/s)", f"{dp_H:.3e}")
            st.metric("ΔxΔp / (ħ/2)", f"{prod/(engine.hbar/2):.4f}")
            st.metric("E_cin (eV)",
                      f"{(engine.hbar*k0)**2/(2*engine.m*1.6022e-19):.4f}")

        with col2:
            prob = np.abs(psi_wp)**2
            re_psi = np.real(psi_wp)
            im_psi = np.imag(psi_wp)

            fig_wp = make_subplots(rows=2, cols=1,
                subplot_titles=[f"Paquet d'onde ψ(x,t={t_fs}fs)",
                                 "Densité de probabilité |ψ|²"])

            fig_wp.add_trace(go.Scatter(x=x_wp*1e12, y=re_psi, mode='lines',
                name='Re(ψ)', line=dict(color='#00ccff', width=2)), row=1, col=1)
            fig_wp.add_trace(go.Scatter(x=x_wp*1e12, y=im_psi, mode='lines',
                name='Im(ψ)', line=dict(color='#7700ff', width=2,
                dash='dash')), row=1, col=1)

            fig_wp.add_trace(go.Scatter(x=x_wp*1e12, y=prob, mode='lines',
                name='|ψ|²', line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.15)'), row=2, col=1)

            x_centre = (x0 + engine.hbar*k0/engine.m * t)*1e12
            fig_wp.add_vline(x=x_centre, line_color='#ffcc00', line_dash='dash',
                             annotation_text=f"x_c={x_centre:.1f}pm", row=2, col=1)

            fig_wp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=520,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_wp.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                 title_text="x (pm)")
            fig_wp.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_wp, use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire Mécanique Quantique")
        for nom, formule in FORMULES.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("---")
        st.markdown("### 🔬 Constantes quantiques")
        df_c = pd.DataFrame([{"Constante": k, "Valeur": v}
                              for k, v in CONSTANTES.items()])
        st.dataframe(df_c, use_container_width=True)

        st.markdown("---")
        for r in ["Cohen-Tannoudji et al. — *Mécanique Quantique* (EDP Sciences, 2018)",
                  "Griffiths — *Introduction to Quantum Mechanics* (Cambridge, 2018)",
                  "Shankar — *Principles of Quantum Mechanics* (Springer, 2012)"]:
            st.markdown(f"- {r}")