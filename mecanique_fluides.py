import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, brentq
from scipy import signal, stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
CONSTANTES = {
    "ρ_eau (kg/m³)":      1000.0,
    "ρ_air (kg/m³)":      1.225,
    "μ_eau (Pa·s)":       1.002e-3,
    "μ_air (Pa·s)":       1.81e-5,
    "ν_eau (m²/s)":       1.004e-6,
    "ν_air (m²/s)":       1.48e-5,
    "g (m/s²)":           9.81,
    "P_atm (Pa)":         101325.0,
    "γ_eau (N/m)":        0.0728,
    "κ_air (J/(kg·K))":   1.4,
    "R_air (J/(kg·K))":   287.0,
}

FORMULES = {
    "Navier-Stokes":         r"\rho\!\left(\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}\right)=-\nabla p+\mu\nabla^2\mathbf{u}+\rho\mathbf{g}",
    "Continuité":            r"\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0",
    "Reynolds":              r"Re=\frac{\rho U L}{\mu}=\frac{UL}{\nu}",
    "Bernoulli":             r"p+\frac{1}{2}\rho u^2+\rho g z=\text{const}",
    "Darcy-Weisbach":        r"\Delta p = f\frac{L}{D}\frac{\rho u^2}{2}",
    "Moody (turbulent)":     r"\frac{1}{\sqrt{f}}=-2\log\!\left(\frac{\varepsilon/D}{3.7}+\frac{2.51}{Re\sqrt{f}}\right)",
    "Couche limite Blasius":  r"\delta(x)=5\sqrt{\frac{\nu x}{U_\infty}}=\frac{5x}{\sqrt{Re_x}}",
    "Traînée":               r"F_D=\frac{1}{2}C_D\rho U^2 A",
    "Portance":              r"F_L=\frac{1}{2}C_L\rho U^2 A",
    "Strouhal":              r"St=\frac{fD}{U}",
    "Weber":                 r"We=\frac{\rho U^2 L}{\gamma}",
    "Froude":                r"Fr=\frac{U}{\sqrt{gL}}",
    "Mach":                  r"Ma=\frac{U}{a}=\frac{U}{\sqrt{\gamma RT}}",
    "Poiseuille (débit)":    r"Q=\frac{\pi R^4}{8\mu}\left(-\frac{dp}{dx}\right)=\frac{\pi R^4\Delta p}{8\mu L}",
    "Torricelli":            r"U_{sortie}=\sqrt{2gh}",
    "Coefficient de perte":  r"\Delta p=K\frac{\rho U^2}{2}",
}

FLUIDES_PREDEF = {
    "Eau (20°C)":      {"rho": 998.2,  "mu": 1.002e-3, "nu": 1.004e-6, "gamma": 0.0728},
    "Air (20°C)":      {"rho": 1.204,  "mu": 1.81e-5,  "nu": 1.506e-5, "gamma": None},
    "Huile moteur":    {"rho": 870.0,  "mu": 0.1,      "nu": 1.15e-4,  "gamma": 0.030},
    "Glycérol":        {"rho": 1261.0, "mu": 1.412,    "nu": 1.12e-3,  "gamma": 0.064},
    "Mercure":         {"rho": 13534.0,"mu": 1.526e-3, "nu": 1.13e-7,  "gamma": 0.485},
    "Personnalisé":    {"rho": 1000.0, "mu": 1e-3,     "nu": 1e-6,     "gamma": 0.07},
}


# ============================================================
# MOTEUR MÉCANIQUE DES FLUIDES
# ============================================================
class FluidEngine:
    """Moteur scientifique complet en mécanique des fluides."""

    def __init__(self, rho: float, mu: float):
        self.rho = rho
        self.mu = mu
        self.nu = mu / rho

    # --- Nombres adimensionnels ---
    def reynolds(self, U: float, L: float) -> float:
        return U * L / self.nu

    def froude(self, U: float, L: float) -> float:
        return U / np.sqrt(9.81 * L)

    def mach(self, U: float, T: float = 293.15,
             gamma: float = 1.4, R: float = 287.0) -> float:
        a = np.sqrt(gamma * R * T)
        return U / a

    def weber(self, U: float, L: float, gamma: float) -> float:
        return self.rho * U**2 * L / gamma

    def strouhal(self, f: float, D: float, U: float) -> float:
        return f * D / U

    # --- Écoulements en conduite ---
    @st.cache_data
    def profil_poiseuille(_self, R: float, n: int = 200) -> tuple:
        """Profil parabolique de Poiseuille."""
        r = np.linspace(-R, R, n)
        u_max = 1.0  # normalisé
        u = u_max * (1 - (r/R)**2)
        return r, u

    def debit_poiseuille(self, R: float, dpdx: float) -> float:
        """Débit volumique en conduite circulaire."""
        return np.pi * R**4 * abs(dpdx) / (8 * self.mu)

    def vitesse_moyenne(self, R: float, dpdx: float) -> float:
        return self.debit_poiseuille(R, dpdx) / (np.pi * R**2)

    def pertes_charge_darcy(self, U: float, L: float, D: float,
                             eps: float = 0.0) -> dict:
        """Pertes de charge par Darcy-Weisbach + Moody."""
        Re = self.reynolds(U, D)
        if Re < 2300:
            f = 64 / Re
            regime = "Laminaire"
        elif Re < 4000:
            f = 64 / Re * 0.5 + 0.316 * Re**(-0.25) * 0.5
            regime = "Transitoire"
        else:
            # Colebrook-White (résolution numérique)
            def colebrook(f_val):
                if f_val <= 0:
                    return 1e10
                return (1/np.sqrt(f_val) +
                        2*np.log10(eps/(3.7*D) + 2.51/(Re*np.sqrt(f_val))))
            try:
                f = brentq(lambda fv: colebrook(fv), 1e-6, 1.0)
            except:
                f = 0.316 * Re**(-0.25)
            regime = "Turbulent"

        dP = f * L/D * 0.5 * self.rho * U**2
        return {"f": f, "Re": Re, "dP": dP, "regime": regime,
                "dP_par_m": dP/L}

    # --- Couche limite ---
    @st.cache_data
    def couche_limite_blasius(_self, x_arr: np.ndarray,
                               U_inf: float) -> dict:
        """Couche limite laminaire de Blasius."""
        Re_x = U_inf * x_arr / _self.nu
        delta = 5 * x_arr / np.sqrt(np.maximum(Re_x, 1e-10))
        delta_star = 1.7208 * x_arr / np.sqrt(np.maximum(Re_x, 1e-10))
        theta = 0.664 * x_arr / np.sqrt(np.maximum(Re_x, 1e-10))
        Cf = 0.664 / np.sqrt(np.maximum(Re_x, 1e-10))
        return {"delta": delta, "delta_star": delta_star,
                "theta": theta, "Cf": Cf, "Re_x": Re_x}

    @st.cache_data
    def profil_blasius(_self, eta_max: float = 8.0, n: int = 200) -> tuple:
        """Profil de vitesse de Blasius par intégration numérique."""
        def blasius_ode(y, eta):
            f, fp, fpp = y
            return [fp, fpp, -0.5*f*fpp]

        eta = np.linspace(0, eta_max, n)
        y0 = [0, 0, 0.332]
        sol = odeint(blasius_ode, y0, eta)
        return eta, sol[:, 1]  # η, f'(η) = u/U_inf

    # --- Traînée et portance ---
    def force_trainee(self, CD: float, U: float, A: float) -> float:
        return 0.5 * CD * self.rho * U**2 * A

    def force_portance(self, CL: float, U: float, A: float) -> float:
        return 0.5 * CL * self.rho * U**2 * A

    def finesse_aerodyn(self, CL: float, CD: float) -> float:
        return CL / CD if CD > 0 else np.inf

    # --- Torricelli / orifice ---
    def vitesse_torricelli(self, h: float, Cd: float = 0.61) -> float:
        return Cd * np.sqrt(2 * 9.81 * h)

    @st.cache_data
    def vidange_reservoir(_self, A_res: float, A_or: float,
                           h0: float, Cd: float = 0.61) -> tuple:
        """Temps de vidange d'un réservoir."""
        def dhdt(h, t):
            if h[0] <= 0:
                return [0]
            v = Cd * np.sqrt(2 * 9.81 * h[0])
            return [-(A_or/A_res) * v]

        t_end = 2 * A_res * np.sqrt(2*h0/9.81) / (Cd * A_or * 1.5)
        t = np.linspace(0, t_end, 500)  # Réduit de 1000 à 500 points
        sol = odeint(dhdt, [h0], t)
        h = np.maximum(sol[:, 0], 0)
        return t, h

    # --- Choc hydraulique ---
    def coup_de_belier(self, U: float, L: float, D: float,
                        e: float, E: float) -> dict:
        """Surpression du coup de bélier (Joukowski)."""
        c = 1 / np.sqrt(self.rho * (1/(E) + D/(e*2e11)))
        delta_p = self.rho * c * U
        t_retour = 2 * L / c
        return {"c_onde": c, "delta_p": delta_p,
                "t_retour": t_retour, "surpression_%": delta_p/101325*100}

    # --- Similitude ---
    def similitude(self, echelle_L: float, U_maquette: float) -> dict:
        """Conditions de similitude Reynolds."""
        Re_mod = U_maquette * echelle_L / self.nu
        U_proto = U_maquette / echelle_L
        Re_proto = U_proto / self.nu
        return {"Re_modele": Re_mod, "U_prototype": U_proto,
                "Re_prototype": Re_proto,
                "Rapport_forces": echelle_L**3}

    # --- Diagnostics ---
    def diagnostiquer(self, U: float, L: float, D: float = None) -> list:
        Re = self.reynolds(U, L)
        D = D or L
        diag = [
            {"Test": "Régime (Re)", "Valeur": f"{Re:.1f}",
             "Statut": "🟢 Laminaire" if Re < 2300 else "🔴 Turbulent" if Re > 4000 else "🟡 Transitoire",
             "Note": f"Re={'<2300' if Re<2300 else '>4000' if Re>4000 else '2300-4000'}"},
            {"Test": "CFL (dt=0.01)", "Valeur": f"{U*0.01/D:.4f}",
             "Statut": "✅ OK" if U*0.01/D < 1 else "⚠️ Instable",
             "Note": "Condition CFL = U·Δt/Δx < 1"},
            {"Test": "Froude", "Valeur": f"{self.froude(U, L):.3f}",
             "Statut": "Fluvial" if self.froude(U,L)<1 else "Torrentiel",
             "Note": "Fr < 1 : fluvial"},
        ]
        return diag


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def mecanique_fluides_page():
    st.markdown("## 💧 Mécanique des Fluides Avancée")
    st.markdown("*Écoulements internes/externes, couche limite, pertes de charge, aérodynamique*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔢 Nombres adimensionnels",
        "🚿 Écoulements internes",
        "🌬️ Couche limite",
        "✈️ Aérodynamique",
        "🌊 Hydraulique",
        "📖 Théorie"
    ])

    with st.sidebar.expander("⚙️ Fluide", expanded=True):
        fluide_sel = st.selectbox("Fluide", list(FLUIDES_PREDEF.keys()))
        if fluide_sel == "Personnalisé":
            rho = st.slider("ρ (kg/m³)", 1.0, 15000.0, 1000.0)
            mu  = st.slider("μ (Pa·s)", 1e-6, 10.0, 1e-3, format="%.2e")
        else:
            fp = FLUIDES_PREDEF[fluide_sel]
            rho, mu = fp["rho"], fp["mu"]
            st.info(f"ρ={rho} kg/m³ | μ={mu:.2e} Pa·s | ν={fp['nu']:.2e} m²/s")

    engine = FluidEngine(rho, mu)

    # ============================================================
    # TAB 1 : NOMBRES ADIMENSIONNELS
    # ============================================================
    with tab1:
        st.markdown("### 🔢 Nombres adimensionnels")
        col1, col2 = st.columns([1, 2])

        with col1:
            U = st.slider("Vitesse U (m/s)", 0.01, 100.0, 1.0, 0.01)
            L = st.slider("Longueur L (m)", 0.001, 10.0, 0.1, 0.001)
            T = st.slider("Température T (K)", 200.0, 400.0, 293.0, 1.0)
            gamma_surf = st.slider("Tension superficielle γ (N/m)", 0.001, 0.5, 0.072, 0.001)

            Re = engine.reynolds(U, L)
            Fr = engine.froude(U, L)
            Ma = engine.mach(U, T)
            We = engine.weber(U, L, gamma_surf)

            nombres = {
                "Reynolds Re": Re,
                "Froude Fr": Fr,
                "Mach Ma": Ma,
                "Weber We": We,
                "Strouhal St (f=1Hz)": engine.strouhal(1.0, L, U),
                "Euler Eu": 1.0/(0.5*rho*U**2) if U > 0 else 0,
            }

            for k, v in nombres.items():
                st.metric(k, f"{v:.4f}")

        with col2:
            # Carte Re vs régimes
            U_arr = np.logspace(-2, 3, 200)
            Re_arr = engine.reynolds(U_arr, L)

            fig_re = go.Figure()
            fig_re.add_trace(go.Scatter(
                x=U_arr, y=Re_arr, mode='lines',
                name='Re(U)', line=dict(color='#00ccff', width=3)
            ))
            fig_re.add_hline(y=2300, line_color='#00ff88', line_dash='dash',
                             annotation_text="Laminaire→Transitoire (Re=2300)")
            fig_re.add_hline(y=4000, line_color='#ff4444', line_dash='dash',
                             annotation_text="Transitoire→Turbulent (Re=4000)")
            fig_re.add_vline(x=U, line_color='#ffcc00', line_dash='dot',
                             annotation_text=f"U={U} m/s")
            fig_re.update_layout(
                title=f"Reynolds vs Vitesse — {fluide_sel}",
                xaxis_title="U (m/s)", yaxis_title="Re",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
            )
            st.plotly_chart(fig_re, use_container_width=True)

            # Diagramme de Moody simplifié
            st.markdown("#### 📊 Diagramme de Moody")
            Re_moody = np.logspace(3, 8, 200)  # Réduit de 300 à 200 points
            eps_D_vals = [0, 1e-4, 1e-3, 5e-3, 1e-2]
            colors_m = ['#00ccff','#7700ff','#ff00cc','#00ff88','#ffcc00']
            fig_moody = go.Figure()

            for i, eps_D in enumerate(eps_D_vals):
                f_vals = []
                for Re_i in Re_moody:
                    if Re_i < 2300:
                        f_vals.append(64/Re_i)
                    else:
                        try:
                            def col(fv):
                                return 1/np.sqrt(fv)+2*np.log10(eps_D/3.7+2.51/(Re_i*np.sqrt(fv)))
                            fv = brentq(col, 1e-6, 1.0)
                            f_vals.append(fv)
                        except:
                            f_vals.append(0.316*Re_i**(-0.25))
                label = f"ε/D={eps_D}" if eps_D > 0 else "Lisse"
                fig_moody.add_trace(go.Scatter(
                    x=Re_moody, y=f_vals, mode='lines', name=label,
                    line=dict(color=colors_m[i], width=2)
                ))
            fig_moody.update_layout(
                title="Diagramme de Moody",
                xaxis_title="Re", yaxis_title="f (Darcy)",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=350,  # Réduit de 400 à 350
                showlegend=False  # Désactive la légende pour plus de rapidité
            )
            st.plotly_chart(fig_moody, use_container_width=True)

    # ============================================================
    # TAB 2 : ÉCOULEMENTS INTERNES
    # ============================================================
    with tab2:
        st.markdown("### 🚿 Écoulements en conduite")
        col1, col2 = st.columns([1, 2])

        with col1:
            R_pipe = st.slider("Rayon R (m)", 0.001, 0.5, 0.05, 0.001)
            L_pipe = st.slider("Longueur L (m)", 0.1, 100.0, 10.0, 0.1)
            dpdx   = st.slider("-dp/dx (Pa/m)", 0.01, 1000.0, 10.0, 0.1)
            eps    = st.slider("Rugosité ε (m)", 0.0, 0.01, 0.0001, 0.0001)

            U_moy = engine.vitesse_moyenne(R_pipe, dpdx)
            Q_vol = engine.debit_poiseuille(R_pipe, dpdx)
            pertes = engine.pertes_charge_darcy(U_moy, L_pipe, 2*R_pipe, eps)

            st.metric("U_moy (m/s)", f"{U_moy:.4f}")
            st.metric("Q (m³/s)", f"{Q_vol:.4e}")
            st.metric("Re", f"{pertes['Re']:.1f}")
            st.metric("Régime", pertes["regime"])
            st.metric("f (Darcy)", f"{pertes['f']:.5f}")
            st.metric("ΔP total (Pa)", f"{pertes['dP']:.2f}")
            st.metric("ΔP/m (Pa/m)", f"{pertes['dP_par_m']:.4f}")

        with col2:
            # Profil Poiseuille
            r, u = engine.profil_poiseuille(R_pipe)
            u_scale = u * U_moy / u.max() if u.max() > 0 else u

            fig_p = make_subplots(rows=1, cols=2,
                subplot_titles=["Profil u(r)", "Champ de vitesse 2D"])

            fig_p.add_trace(go.Scatter(
                x=u_scale, y=r, mode='lines', name='u(r)',
                line=dict(color='#00ccff', width=3),
                fill='tozerox', fillcolor='rgba(0,204,255,0.15)'
            ), row=1, col=1)
            fig_p.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=1, col=1)

            # Heatmap 2D
            theta = np.linspace(0, 2*np.pi, 40)  # Réduit de 60 à 40 points
            r_2d = np.linspace(0, R_pipe, 25)    # Réduit de 40 à 25 points
            R_g, Th_g = np.meshgrid(r_2d, theta)
            U_2d = U_moy * 2 * (1 - (R_g/R_pipe)**2)
            X_2d = R_g * np.cos(Th_g) * 1000
            Y_2d = R_g * np.sin(Th_g) * 1000

            fig_p.add_trace(go.Scatter(
                x=X_2d.flatten(), y=Y_2d.flatten(), mode='markers',
                marker=dict(color=U_2d.flatten(), size=4,
                           colorscale=[[0,'#020817'],[0.5,'#7700ff'],[1,'#00ccff']],
                           showscale=True),
                name='u(x,y)'
            ), row=1, col=2)

            fig_p.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=420,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_p.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_p.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_p, use_container_width=True)

            # Pertes sur longueur variable
            L_arr = np.linspace(0.1, L_pipe, 200)
            dP_arr = pertes['dP_par_m'] * L_arr
            fig_dP = go.Figure()
            fig_dP.add_trace(go.Scatter(x=L_arr, y=dP_arr, mode='lines',
                name='ΔP(L)', line=dict(color='#ff00cc', width=2.5),
                fill='tozeroy', fillcolor='rgba(255,0,204,0.1)'))
            fig_dP.update_layout(
                title="Pertes de charge vs longueur",
                xaxis_title="L (m)", yaxis_title="ΔP (Pa)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=300,
            )
            st.plotly_chart(fig_dP, use_container_width=True)

    # ============================================================
    # TAB 3 : COUCHE LIMITE
    # ============================================================
    with tab3:
        st.markdown("### 🌬️ Couche limite laminaire de Blasius")
        col1, col2 = st.columns([1, 2])

        with col1:
            U_inf = st.slider("U∞ (m/s)", 0.01, 50.0, 5.0, 0.1)
            x_max = st.slider("x_max (m)", 0.01, 5.0, 1.0, 0.01)

            x_arr = np.linspace(0.001, x_max, 300)
            bl = engine.couche_limite_blasius(x_arr, U_inf)

            Re_L = engine.reynolds(U_inf, x_max)
            st.metric("Re_L", f"{Re_L:.2e}")
            st.metric("δ à x_max (mm)", f"{bl['delta'][-1]*1000:.3f}")
            st.metric("δ* à x_max (mm)", f"{bl['delta_star'][-1]*1000:.3f}")
            st.metric("θ à x_max (mm)", f"{bl['theta'][-1]*1000:.3f}")
            st.metric("Cf à x_max", f"{bl['Cf'][-1]:.4f}")
            st.metric("Cf moyen", f"{2*bl['Cf'][-1]:.4f}")

        with col2:
            fig_bl = make_subplots(rows=2, cols=1,
                subplot_titles=["Épaisseurs de couche limite", "Profil de Blasius f'(η)"])

            fig_bl.add_trace(go.Scatter(
                x=x_arr, y=bl['delta']*1000, mode='lines', name='δ (mm)',
                line=dict(color='#00ccff', width=2.5)
            ), row=1, col=1)
            fig_bl.add_trace(go.Scatter(
                x=x_arr, y=bl['delta_star']*1000, mode='lines', name='δ* (mm)',
                line=dict(color='#7700ff', width=2, dash='dash')
            ), row=1, col=1)
            fig_bl.add_trace(go.Scatter(
                x=x_arr, y=bl['theta']*1000, mode='lines', name='θ (mm)',
                line=dict(color='#ff00cc', width=2, dash='dot')
            ), row=1, col=1)

            eta, fp_blasius = engine.profil_blasius()
            fig_bl.add_trace(go.Scatter(
                x=fp_blasius, y=eta, mode='lines', name="u/U∞",
                line=dict(color='#00ccff', width=3),
                fill='tozerox', fillcolor='rgba(0,204,255,0.1)'
            ), row=2, col=1)
            fig_bl.add_vline(x=0.99, line_color='#ffcc00', line_dash='dash',
                             annotation_text="u/U∞=0.99", row=2, col=1)

            fig_bl.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=550,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_bl.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_bl.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_bl, use_container_width=True)

    # ============================================================
    # TAB 4 : AÉRODYNAMIQUE
    # ============================================================
    with tab4:
        st.markdown("### ✈️ Aérodynamique & Forces")
        col1, col2 = st.columns([1, 2])

        with col1:
            U_aero = st.slider("Vitesse U (m/s)", 1.0, 300.0, 50.0, 1.0)
            S_aero = st.slider("Surface S (m²)", 0.01, 200.0, 20.0, 0.1)
            CD = st.slider("Cx (traînée)", 0.001, 2.0, 0.3, 0.001)
            CL = st.slider("Cz (portance)", 0.0, 3.0, 0.5, 0.01)
            masse = st.slider("Masse (kg)", 1.0, 500000.0, 5000.0, 100.0)

            FD = engine.force_trainee(CD, U_aero, S_aero)
            FL = engine.force_portance(CL, U_aero, S_aero)
            finesse = engine.finesse_aerodyn(CL, CD)
            poids = masse * 9.81

            st.metric("Traînée FD (N)", f"{FD:.2f}")
            st.metric("Portance FL (N)", f"{FL:.2f}")
            st.metric("Finesse CL/CD", f"{finesse:.2f}")
            st.metric("Charge utile (FL/FD)", f"{FL/FD:.2f}" if FD > 0 else "∞")
            st.metric("FL / Poids", f"{FL/poids:.4f}")
            st.metric("Vitesse décrochage (m/s)",
                      f"{np.sqrt(2*masse*9.81/(rho*S_aero*(CL+0.01))):.2f}")

        with col2:
            U_sweep = np.linspace(1, 300, 300)
            FD_sweep = engine.force_trainee(CD, U_sweep, S_aero)
            FL_sweep = engine.force_portance(CL, U_sweep, S_aero)

            fig_aero = go.Figure()
            fig_aero.add_trace(go.Scatter(x=U_sweep, y=FD_sweep, mode='lines',
                name='Traînée FD (N)', line=dict(color='#ff4444', width=2.5)))
            fig_aero.add_trace(go.Scatter(x=U_sweep, y=FL_sweep, mode='lines',
                name='Portance FL (N)', line=dict(color='#00ccff', width=2.5)))
            fig_aero.add_hline(y=poids, line_color='#ffcc00', line_dash='dash',
                               annotation_text=f"Poids={poids:.0f}N")
            fig_aero.add_vline(x=U_aero, line_color='rgba(255,255,255,0.4)',
                               line_dash='dot', annotation_text=f"U={U_aero}m/s")
            fig_aero.update_layout(
                title="Forces aérodynamiques vs vitesse",
                xaxis_title="U (m/s)", yaxis_title="Force (N)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
            )
            st.plotly_chart(fig_aero, use_container_width=True)

            # Polaire
            alpha_arr = np.linspace(-5, 20, 100)
            CL_arr = 0.1 * alpha_arr + CL
            CD_arr = CD + 0.01 * alpha_arr**2 / 100
            fig_pol = go.Figure()
            fig_pol.add_trace(go.Scatter(x=CD_arr, y=CL_arr, mode='lines',
                line=dict(color='#00ccff', width=2.5), name='Polaire'))
            fig_pol.update_layout(
                title="Polaire de portance CL=f(CD)",
                xaxis_title="CD (traînée)", yaxis_title="CL (portance)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=320,
            )
            st.plotly_chart(fig_pol, use_container_width=True)

    # ============================================================
    # TAB 5 : HYDRAULIQUE
    # ============================================================
    with tab5:
        st.markdown("### 🌊 Hydraulique & Torricelli")
        col1, col2 = st.columns([1, 2])

        with col1:
            h0 = st.slider("Hauteur initiale h₀ (m)", 0.1, 20.0, 5.0, 0.1)
            A_res = st.slider("Section réservoir A_res (m²)", 0.01, 100.0, 1.0, 0.01)
            A_or = st.slider("Section orifice A_or (m²)", 0.0001, 0.1, 0.005, 0.0001)
            Cd = st.slider("Coefficient de décharge Cd", 0.4, 1.0, 0.61, 0.01)

            v_tor = engine.vitesse_torricelli(h0, Cd)
            Q_or = Cd * A_or * np.sqrt(2 * 9.81 * h0)

            st.metric("Vitesse Torricelli (m/s)", f"{v_tor:.3f}")
            st.metric("Débit (m³/s)", f"{Q_or:.4f}")
            st.metric("Débit (L/s)", f"{Q_or*1000:.2f}")

        with col2:
            t_vid, h_vid = engine.vidange_reservoir(A_res, A_or, h0, Cd)
            Q_vid = Cd * A_or * np.sqrt(2 * 9.81 * np.maximum(h_vid, 0))

            fig_vid = make_subplots(rows=2, cols=1,
                subplot_titles=["Hauteur h(t)", "Débit Q(t)"])
            fig_vid.add_trace(go.Scatter(x=t_vid, y=h_vid, mode='lines',
                name='h(t)', line=dict(color='#00ccff', width=3),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'), row=1, col=1)
            fig_vid.add_trace(go.Scatter(x=t_vid, y=Q_vid, mode='lines',
                name='Q(t)', line=dict(color='#7700ff', width=2.5)), row=2, col=1)

            idx_vide = np.where(h_vid < 0.001)[0]
            if len(idx_vide) > 0:
                t_vide = t_vid[idx_vide[0]]
                fig_vid.add_vline(x=t_vide, line_color='#ffcc00', line_dash='dash',
                                  annotation_text=f"t_vidange={t_vide:.1f}s", row=1, col=1)

            fig_vid.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=500,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_vid.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                  title_text="Temps (s)")
            fig_vid.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_vid, use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire mécanique des fluides")
        for nom, formule in FORMULES.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("---")
        st.markdown("### 🔬 Propriétés des fluides")
        df_f = pd.DataFrame([{"Fluide": k, **{kk: vv for kk, vv in v.items() if vv is not None}}
                              for k, v in FLUIDES_PREDEF.items() if k != "Personnalisé"])
        st.dataframe(df_f, use_container_width=True)

        st.markdown("---")
        for r in ["Batchelor — *An Introduction to Fluid Dynamics* (Cambridge, 2000)",
                  "White — *Fluid Mechanics* (McGraw-Hill, 2015)",
                  "Schlichting — *Boundary Layer Theory* (Springer, 2017)"]:
            st.markdown(f"- {r}")