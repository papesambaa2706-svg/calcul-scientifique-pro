import streamlit as st
import numpy as np
from scipy.integrate import odeint, solve_ivp, quad
from scipy import linalg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORMULAIRE
# ============================================================
FORMULES = {
    "EDO ordre 1":           r"\frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0",
    "EDO ordre 2":           r"\frac{d^2y}{dt^2} + 2\zeta\omega_n\frac{dy}{dt} + \omega_n^2 y = 0",
    "Méthode d'Euler":       r"y_{n+1} = y_n + h\,f(t_n, y_n)",
    "Runge-Kutta 4":         r"y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)",
    "Lorenz":                r"\dot{x}=\sigma(y-x),\quad\dot{y}=x(\rho-z)-y,\quad\dot{z}=xy-\beta z",
    "Lotka-Volterra":        r"\dot{x}=\alpha x - \beta xy,\quad \dot{y}=-\gamma y + \delta xy",
    "Logistique":            r"\frac{dy}{dt} = ky\left(1 - \frac{y}{M}\right)",
    "Oscillateur Van der Pol":r"\ddot{x} - \mu(1-x^2)\dot{x} + x = 0",
    "Pendule non-linéaire":  r"\ddot{\theta} + \frac{g}{L}\sin\theta + b\dot{\theta} = 0",
    "Équation de la chaleur":r"\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2}",
    "SIR (épidémiologie)":   r"\dot{S}=-\beta SI,\quad\dot{I}=\beta SI-\gamma I,\quad\dot{R}=\gamma I",
    "Erreur RK4":            r"|e_n| \leq C h^4 \quad \text{(ordre 4)}",
}

SYSTEMES = {
    "Décroissance exponentielle": "dy/dt = -k·y",
    "Croissance exponentielle":   "dy/dt = +k·y",
    "Oscillateur harmonique":     "ÿ + ω²y = 0",
    "Oscillateur amorti":         "ÿ + 2ζω·ẏ + ω²y = 0",
    "Oscillateur Van der Pol":    "ÿ - μ(1-y²)ẏ + y = 0",
    "Logistique":                 "dy/dt = k·y(1-y/M)",
    "Lotka-Volterra":             "ẋ=αx-βxy, ẏ=-γy+δxy",
    "Lorenz (chaos)":             "système 3D chaotique",
    "Pendule non-linéaire":       "ÿ + (g/L)sinθ + bθ̇ = 0",
    "SIR épidémique":             "S,I,R couplés",
    "Double pendule":             "système chaotique à 4D",
    "Équation de Duffing":        "ÿ + δẏ + αy + βy³ = γcos(ωt)",
}

METHODES_NUMERIQUES = {
    "Euler explicite":  {"ordre": 1, "stabilité": "Conditionnelle", "coût": "O(n)"},
    "Euler implicite":  {"ordre": 1, "stabilité": "Inconditionnelle","coût": "O(n²)"},
    "RK4":              {"ordre": 4, "stabilité": "Conditionnelle", "coût": "O(4n)"},
    "RK45 (adaptatif)": {"ordre": 5, "stabilité": "Adaptative",    "coût": "Variable"},
    "LSODA (scipy)":    {"ordre": 12,"stabilité": "Adaptative",    "coût": "Variable"},
}


# ============================================================
# SOLVEURS NUMÉRIQUES
# ============================================================
class SolveurODE:
    """Solveurs numériques pour EDO avec comparaison de méthodes."""

    @staticmethod
    def euler_explicite(f, y0, t):
        y = np.zeros((len(t), len(np.atleast_1d(y0))))
        y[0] = np.atleast_1d(y0)
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            y[i+1] = y[i] + h * np.atleast_1d(f(t[i], y[i]))
        return y

    @staticmethod
    def euler_implicite(f, y0, t, tol=1e-8):
        """Euler implicite via point fixe."""
        from scipy.optimize import fsolve
        y = np.zeros((len(t), len(np.atleast_1d(y0))))
        y[0] = np.atleast_1d(y0)
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            y_pred = y[i] + h * np.atleast_1d(f(t[i], y[i]))
            try:
                y[i+1] = fsolve(lambda yn: yn - y[i] - h*np.atleast_1d(f(t[i+1], yn)),
                                 y_pred, full_output=False)
            except:
                y[i+1] = y_pred
        return y

    @staticmethod
    def rk4(f, y0, t):
        y = np.zeros((len(t), len(np.atleast_1d(y0))))
        y[0] = np.atleast_1d(y0)
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            yi = y[i]
            k1 = np.atleast_1d(f(t[i],         yi))
            k2 = np.atleast_1d(f(t[i]+h/2,     yi+h/2*k1))
            k3 = np.atleast_1d(f(t[i]+h/2,     yi+h/2*k2))
            k4 = np.atleast_1d(f(t[i]+h,       yi+h*k3))
            y[i+1] = yi + h/6*(k1 + 2*k2 + 2*k3 + k4)
        return y

    @staticmethod
    def rk45_scipy(f_ivp, y0, t):
        sol = solve_ivp(f_ivp, [t[0], t[-1]], np.atleast_1d(y0),
                        t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)
        return sol.y.T

    @staticmethod
    def lsoda_scipy(f_odeint, y0, t):
        return odeint(f_odeint, np.atleast_1d(y0), t, rtol=1e-8, atol=1e-10)


# ============================================================
# MOTEUR EDO
# ============================================================
class EquDiffEngine:
    """Moteur complet de résolution et analyse d'EDO."""

    def __init__(self, eq_type: str, params: dict):
        self.eq_type = eq_type
        self.params = params
        self.solveur = SolveurODE()

    def definir_systeme(self):
        """Retourne (f_odeint, f_ivp, y0, n_dim, labels)."""
        p = self.params

        if self.eq_type == "Décroissance exponentielle":
            f = lambda y, t: -p["k"] * y
            f_ivp = lambda t, y: [-p["k"] * y[0]]
            y0 = [p["y0"]]
            labels = ["y(t)"]

        elif self.eq_type == "Croissance exponentielle":
            f = lambda y, t: p["k"] * y
            f_ivp = lambda t, y: [p["k"] * y[0]]
            y0 = [p["y0"]]
            labels = ["y(t)"]

        elif self.eq_type == "Oscillateur harmonique":
            omega = p["k"]
            f = lambda Y, t: [Y[1], -omega**2 * Y[0]]
            f_ivp = lambda t, Y: [Y[1], -omega**2 * Y[0]]
            y0 = [p["y0"], 0]
            labels = ["y(t)", "ẏ(t)"]

        elif self.eq_type == "Oscillateur amorti":
            omega, zeta = p["k"], p.get("zeta", 0.3)
            f = lambda Y, t: [Y[1], -2*zeta*omega*Y[1] - omega**2*Y[0]]
            f_ivp = lambda t, Y: [Y[1], -2*zeta*omega*Y[1] - omega**2*Y[0]]
            y0 = [p["y0"], 0]
            labels = ["y(t)", "ẏ(t)"]

        elif self.eq_type == "Oscillateur Van der Pol":
            mu = p.get("mu", 1.0)
            f = lambda Y, t: [Y[1], mu*(1-Y[0]**2)*Y[1] - Y[0]]
            f_ivp = lambda t, Y: [Y[1], mu*(1-Y[0]**2)*Y[1] - Y[0]]
            y0 = [p["y0"], 0]
            labels = ["y(t)", "ẏ(t)"]

        elif self.eq_type == "Logistique":
            k, M = p["k"], p.get("M", 10.0)
            f = lambda y, t: k * y * (1 - y/M)
            f_ivp = lambda t, y: [k * y[0] * (1 - y[0]/M)]
            y0 = [p["y0"]]
            labels = ["N(t)"]

        elif self.eq_type == "Lotka-Volterra":
            alpha, beta = p.get("alpha", 1.0), p.get("beta", 0.1)
            gamma, delta = p.get("gamma", 0.075), p.get("delta", 1.5)
            f = lambda Y, t: [alpha*Y[0] - beta*Y[0]*Y[1],
                               -gamma*Y[1] + delta*beta*Y[0]*Y[1]]
            f_ivp = lambda t, Y: [alpha*Y[0] - beta*Y[0]*Y[1],
                                   -gamma*Y[1] + delta*beta*Y[0]*Y[1]]
            y0 = [p["y0"], p.get("y0_pred", 1.0)]
            labels = ["Proies", "Prédateurs"]

        elif self.eq_type == "Lorenz (chaos)":
            sigma = p.get("sigma", 10.0)
            rho = p.get("rho", 28.0)
            beta = p.get("beta_l", 8/3)
            f = lambda Y, t: [sigma*(Y[1]-Y[0]),
                               Y[0]*(rho-Y[2])-Y[1],
                               Y[0]*Y[1]-beta*Y[2]]
            f_ivp = lambda t, Y: [sigma*(Y[1]-Y[0]),
                                   Y[0]*(rho-Y[2])-Y[1],
                                   Y[0]*Y[1]-beta*Y[2]]
            y0 = [1.0, 1.0, 1.0]
            labels = ["x", "y", "z"]

        elif self.eq_type == "Pendule non-linéaire":
            g, L, b = 9.81, p.get("L", 1.0), p.get("b", 0.1)
            f = lambda Y, t: [Y[1], -(g/L)*np.sin(Y[0]) - b*Y[1]]
            f_ivp = lambda t, Y: [Y[1], -(g/L)*np.sin(Y[0]) - b*Y[1]]
            y0 = [p["y0"], 0]
            labels = ["θ(t)", "ω(t)"]

        elif self.eq_type == "SIR épidémique":
            N = p.get("N_pop", 1000)
            beta_s = p.get("beta_sir", 0.3)
            gamma_s = p.get("gamma_sir", 0.05)
            f = lambda Y, t: [-beta_s*Y[0]*Y[1]/N,
                               beta_s*Y[0]*Y[1]/N - gamma_s*Y[1],
                               gamma_s*Y[1]]
            f_ivp = lambda t, Y: [-beta_s*Y[0]*Y[1]/N,
                                   beta_s*Y[0]*Y[1]/N - gamma_s*Y[1],
                                   gamma_s*Y[1]]
            y0 = [N - 1, 1.0, 0.0]
            labels = ["S (Susceptibles)", "I (Infectés)", "R (Guéris)"]

        elif self.eq_type == "Équation de Duffing":
            delta = p.get("delta", 0.2)
            alpha_d = p.get("alpha_d", -1.0)
            beta_d = p.get("beta_d", 1.0)
            gamma_d = p.get("gamma_d", 0.3)
            omega_d = p.get("omega_d", 1.2)
            f = lambda Y, t: [Y[1], gamma_d*np.cos(omega_d*t) - delta*Y[1] - alpha_d*Y[0] - beta_d*Y[0]**3]
            f_ivp = lambda t, Y: [Y[1], gamma_d*np.cos(omega_d*t) - delta*Y[1] - alpha_d*Y[0] - beta_d*Y[0]**3]
            y0 = [0.1, 0.0]
            labels = ["x(t)", "ẋ(t)"]

        else:
            f = lambda y, t: -p["k"] * y
            f_ivp = lambda t, y: [-p["k"] * y[0]]
            y0 = [p["y0"]]
            labels = ["y(t)"]

        return f, f_ivp, y0, labels

    def resoudre(self, methode: str, t: np.ndarray) -> np.ndarray:
        f, f_ivp, y0, labels = self.definir_systeme()
        try:
            if methode == "Euler explicite":
                return self.solveur.euler_explicite(lambda t, y: np.array(f(y, t)), y0, t), labels
            elif methode == "Euler implicite":
                return self.solveur.euler_implicite(lambda t, y: np.array(f(y, t)), y0, t), labels
            elif methode == "RK4":
                return self.solveur.rk4(lambda t, y: np.array(f(y, t)), y0, t), labels
            elif methode == "RK45 (adaptatif)":
                return self.solveur.rk45_scipy(f_ivp, y0, t), labels
            else:  # LSODA
                return self.solveur.lsoda_scipy(f, y0, t), labels
        except Exception as e:
            return np.zeros((len(t), len(y0))), labels

    def solution_analytique(self, t: np.ndarray):
        p = self.params
        if self.eq_type == "Décroissance exponentielle":
            return p["y0"] * np.exp(-p["k"] * t)
        elif self.eq_type == "Croissance exponentielle":
            return p["y0"] * np.exp(p["k"] * t)
        elif self.eq_type == "Oscillateur harmonique":
            return p["y0"] * np.cos(p["k"] * t)
        return None

    def erreur_methodes(self, t: np.ndarray) -> pd.DataFrame:
        """Erreur de chaque méthode vs solution analytique."""
        y_exact = self.solution_analytique(t)
        if y_exact is None:
            return pd.DataFrame()

        methodes = ["Euler explicite", "RK4", "RK45 (adaptatif)", "LSODA (scipy)"]
        resultats = []
        for m in methodes:
            sol, _ = self.resoudre(m, t)
            err = np.abs(sol[:, 0] - y_exact)
            resultats.append({
                "Méthode": m,
                "Erreur max": f"{err.max():.2e}",
                "Erreur moy.": f"{err.mean():.2e}",
                "Erreur finale": f"{err[-1]:.2e}",
            })
        return pd.DataFrame(resultats)


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def equ_diff_page():
    st.markdown("## 🧮 Équations Différentielles Avancées")
    st.markdown("*Solveurs numériques, systèmes dynamiques, chaos, benchmark*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Résolution",
        "🌐 Portrait de phase",
        "⚔️ Benchmark méthodes",
        "🔬 Systèmes spéciaux",
        "📖 Théorie"
    ])

    # Config partagée
    with st.sidebar.expander("⚙️ Équation différentielle", expanded=True):
        eq_type = st.selectbox("Système", list(SYSTEMES.keys()))
        st.caption(SYSTEMES[eq_type])

        y0_val = st.slider("y(0) / x(0)", 0.1, 20.0, 1.0, 0.1)
        k_val = st.slider("k / ω", 0.01, 5.0, 0.5, 0.01)
        t_max = st.slider("Temps max (s)", 5.0, 200.0, 30.0, 1.0)
        n_pts = st.slider("Points", 200, 5000, 1000, 100)

        params = {"y0": y0_val, "k": k_val}

        # Paramètres spécifiques
        if eq_type == "Oscillateur amorti":
            params["zeta"] = st.slider("ζ (amortissement)", 0.0, 3.0, 0.3, 0.05)
        elif eq_type == "Oscillateur Van der Pol":
            params["mu"] = st.slider("μ", 0.1, 5.0, 1.0, 0.1)
        elif eq_type == "Logistique":
            params["M"] = st.slider("Capacité M", 1.0, 100.0, 10.0, 1.0)
        elif eq_type == "Lotka-Volterra":
            params["alpha"] = st.slider("α (croissance proies)", 0.1, 3.0, 1.0, 0.1)
            params["beta"] = st.slider("β (prédation)", 0.01, 1.0, 0.1, 0.01)
            params["gamma"] = st.slider("γ (mort prédateurs)", 0.01, 1.0, 0.075, 0.001)
            params["delta"] = st.slider("δ (assimilation)", 0.1, 3.0, 1.5, 0.1)
            params["y0_pred"] = st.slider("Prédateurs(0)", 0.1, 20.0, 1.0, 0.1)
        elif eq_type == "Lorenz (chaos)":
            params["sigma"] = st.slider("σ", 1.0, 20.0, 10.0, 0.5)
            params["rho"] = st.slider("ρ", 1.0, 50.0, 28.0, 0.5)
            params["beta_l"] = st.slider("β", 0.1, 5.0, 8/3, 0.1)
        elif eq_type == "Pendule non-linéaire":
            params["L"] = st.slider("Longueur L (m)", 0.1, 5.0, 1.0, 0.1)
            params["b"] = st.slider("Amortissement b", 0.0, 2.0, 0.1, 0.01)
        elif eq_type == "SIR épidémique":
            params["N_pop"] = st.slider("Population N", 100, 10000, 1000, 100)
            params["beta_sir"] = st.slider("β (contagion)", 0.01, 1.0, 0.3, 0.01)
            params["gamma_sir"] = st.slider("γ (guérison)", 0.01, 0.5, 0.05, 0.01)
        elif eq_type == "Équation de Duffing":
            params["delta"] = st.slider("δ (amortissement)", 0.0, 2.0, 0.2, 0.01)
            params["alpha_d"] = st.slider("α", -2.0, 2.0, -1.0, 0.1)
            params["beta_d"] = st.slider("β", 0.1, 5.0, 1.0, 0.1)
            params["gamma_d"] = st.slider("γ (force)", 0.0, 1.0, 0.3, 0.01)
            params["omega_d"] = st.slider("ω_d", 0.1, 5.0, 1.2, 0.05)

    engine = EquDiffEngine(eq_type, params)
    t = np.linspace(0, t_max, n_pts)

    # ============================================================
    # TAB 1 : RÉSOLUTION
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Solveur")
            methode = st.selectbox("Méthode numérique", list(METHODES_NUMERIQUES.keys()))
            info_m = METHODES_NUMERIQUES[methode]
            st.markdown(f"""
            - **Ordre :** {info_m['ordre']}
            - **Stabilité :** {info_m['stabilité']}
            - **Coût :** {info_m['coût']}
            """)
            show_exact = st.checkbox("Afficher solution exacte (si disponible)", True)
            show_energie = st.checkbox("Afficher énergie", False)

        with col2:
            sol, labels = engine.resoudre(methode, t)

            colors_sol = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88']
            fig = go.Figure()

            # Solution numérique
            for i, lab in enumerate(labels):
                fig.add_trace(go.Scatter(
                    x=t, y=sol[:, i], mode='lines', name=f'{lab} ({methode})',
                    line=dict(color=colors_sol[i % len(colors_sol)], width=2.5)
                ))

            # Solution analytique
            if show_exact:
                y_exact = engine.solution_analytique(t)
                if y_exact is not None:
                    fig.add_trace(go.Scatter(
                        x=t, y=y_exact, mode='lines', name='Solution exacte',
                        line=dict(color='#ffffff', width=2, dash='dash')
                    ))

            fig.update_layout(
                title=f"{eq_type} — Méthode : {methode}",
                xaxis_title="Temps (s)", yaxis_title="État",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Métriques
            c1, c2, c3 = st.columns(3)
            with c1: st.metric(f"{labels[0]} final", f"{sol[-1, 0]:.4f}")
            with c2: st.metric(f"Max {labels[0]}", f"{sol[:, 0].max():.4f}")
            with c3: st.metric(f"Min {labels[0]}", f"{sol[:, 0].min():.4f}")

            # Énergie oscillateur
            if show_energie and sol.shape[1] >= 2 and eq_type in [
                "Oscillateur harmonique", "Oscillateur amorti", "Oscillateur Van der Pol"
            ]:
                E_cin = 0.5 * sol[:, 1]**2
                E_pot = 0.5 * params["k"]**2 * sol[:, 0]**2
                E_tot = E_cin + E_pot
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=t, y=E_cin, name='E cinétique',
                                           line=dict(color='#00ccff', width=2)))
                fig_e.add_trace(go.Scatter(x=t, y=E_pot, name='E potentielle',
                                           line=dict(color='#7700ff', width=2)))
                fig_e.add_trace(go.Scatter(x=t, y=E_tot, name='E totale',
                                           line=dict(color='#ffffff', width=2.5, dash='dash')))
                fig_e.update_layout(
                    title="Énergie du système",
                    xaxis_title="t (s)", yaxis_title="Énergie (u.a.)",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=320,
                )
                st.plotly_chart(fig_e, use_container_width=True)

            # Export
            df_exp = pd.DataFrame({"t": t, **{lab: sol[:, i]
                                               for i, lab in enumerate(labels)}})
            st.download_button("💾 Export CSV", df_exp.to_csv(index=False).encode(),
                               "ode_solution.csv", "text/csv")

    # ============================================================
    # TAB 2 : PORTRAIT DE PHASE
    # ============================================================
    with tab2:
        st.markdown("### 🌐 Portrait de phase")

        sol_ph, labels_ph = engine.resoudre("RK45 (adaptatif)", t)

        if sol_ph.shape[1] >= 2:
            if sol_ph.shape[1] >= 3:
                fig_ph = go.Figure(data=go.Scatter3d(
                    x=sol_ph[:, 0], y=sol_ph[:, 1], z=sol_ph[:, 2],
                    mode='lines',
                    line=dict(color=t, colorscale=[[0,'#7700ff'],[0.5,'#00ccff'],[1,'#ffffff']],
                             width=2),
                    name='Trajectoire 3D'
                ))
                fig_title = "Attracteur de Lorenz" if eq_type == "Lorenz (chaos)" else f"Portrait de phase 3D — {eq_type}"
                fig_ph.update_layout(
                    title=fig_title,
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff'), yaxis=dict(color='#c0d0ff'),
                        zaxis=dict(color='#c0d0ff')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c0d0ff'),
                    height=550, margin=dict(l=0, r=0, t=40, b=0)
                )
            else:
                fig_ph = go.Figure()
                fig_ph.add_trace(go.Scatter(
                    x=sol_ph[:, 0], y=sol_ph[:, 1], mode='lines',
                    name=f'{labels_ph[0]} vs {labels_ph[1]}',
                    line=dict(color='#00ccff', width=2),
                ))
                # Point initial
                fig_ph.add_trace(go.Scatter(
                    x=[sol_ph[0, 0]], y=[sol_ph[0, 1]], mode='markers',
                    name='Départ', marker=dict(color='#ffcc00', size=14, symbol='star')
                ))
                fig_ph.update_layout(
                    title=f"Portrait de phase — {eq_type}",
                    xaxis_title=labels_ph[0], yaxis_title=labels_ph[1],
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=500,
                )
            st.plotly_chart(fig_ph, use_container_width=True)

            # Champ de vecteurs
            if sol_ph.shape[1] == 2 and eq_type not in ["Lorenz (chaos)"]:
                st.markdown("#### 🧭 Champ de vecteurs")
                x_range = np.linspace(sol_ph[:, 0].min()*1.2, sol_ph[:, 0].max()*1.2, 15)
                y_range = np.linspace(sol_ph[:, 1].min()*1.2, sol_ph[:, 1].max()*1.2, 15)
                X_g, Y_g = np.meshgrid(x_range, y_range)
                _, f_ivp, _, _ = engine.definir_systeme()

                U_g = np.zeros_like(X_g)
                V_g = np.zeros_like(Y_g)
                for i in range(X_g.shape[0]):
                    for j in range(X_g.shape[1]):
                        try:
                            dxy = f_ivp(0, [X_g[i,j], Y_g[i,j]])
                            norm = np.sqrt(dxy[0]**2 + dxy[1]**2) + 1e-8
                            U_g[i,j] = dxy[0] / norm
                            V_g[i,j] = dxy[1] / norm
                        except:
                            pass

                fig_vf = go.Figure()
                fig_vf.add_trace(go.Scatter(
                    x=sol_ph[:, 0], y=sol_ph[:, 1], mode='lines', name='Trajectoire',
                    line=dict(color='#00ccff', width=2.5)
                ))
                # Flèches
                for i in range(0, X_g.shape[0], 2):
                    for j in range(0, X_g.shape[1], 2):
                        fig_vf.add_annotation(
                            x=X_g[i,j] + U_g[i,j]*0.2,
                            y=Y_g[i,j] + V_g[i,j]*0.2,
                            ax=X_g[i,j], ay=Y_g[i,j],
                            xref='x', yref='y', axref='x', ayref='y',
                            arrowhead=2, arrowsize=1,
                            arrowcolor='rgba(119,0,255,0.5)',
                            arrowwidth=1.5, showarrow=True
                        )
                fig_vf.update_layout(
                    title="Champ de vecteurs",
                    xaxis_title=labels_ph[0], yaxis_title=labels_ph[1],
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    height=430,
                )
                st.plotly_chart(fig_vf, use_container_width=True)
        else:
            st.info("Portrait de phase disponible pour systèmes d'ordre ≥ 2.")

    # ============================================================
    # TAB 3 : BENCHMARK
    # ============================================================
    with tab3:
        st.markdown("### ⚔️ Benchmark des méthodes numériques")

        if eq_type in ["Décroissance exponentielle", "Croissance exponentielle",
                        "Oscillateur harmonique"]:
            df_err = engine.erreur_methodes(t)
            if not df_err.empty:
                st.dataframe(df_err, use_container_width=True)

                # Courbes d'erreur
                fig_bench = go.Figure()
                methodes_b = ["Euler explicite", "RK4", "RK45 (adaptatif)"]
                colors_b = ['#ff4444', '#7700ff', '#00ccff']
                y_exact = engine.solution_analytique(t)

                for m, col in zip(methodes_b, colors_b):
                    sol_b, _ = engine.resoudre(m, t)
                    err = np.abs(sol_b[:, 0] - y_exact)
                    fig_bench.add_trace(go.Scatter(
                        x=t, y=err, mode='lines', name=m,
                        line=dict(color=col, width=2)
                    ))
                fig_bench.update_layout(
                    title="Erreur |numérique - exact| par méthode",
                    xaxis_title="t (s)", yaxis_title="Erreur absolue",
                    yaxis_type='log',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
                )
                st.plotly_chart(fig_bench, use_container_width=True)

            # Effet du pas h
            st.markdown("#### 📉 Convergence vs pas h")
            y_exact_fin = engine.solution_analytique(np.array([t_max]))[0]
            h_vals = np.logspace(-3, 0, 20)
            errs_euler, errs_rk4 = [], []
            for h in h_vals:
                t_h = np.arange(0, t_max + h, h)
                sol_e, _ = engine.resoudre("Euler explicite", t_h)
                sol_r, _ = engine.resoudre("RK4", t_h)
                errs_euler.append(abs(sol_e[-1, 0] - y_exact_fin))
                errs_rk4.append(abs(sol_r[-1, 0] - y_exact_fin))

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=h_vals, y=errs_euler, mode='lines+markers',
                name='Euler (ordre 1)', line=dict(color='#ff4444', width=2)))
            fig_conv.add_trace(go.Scatter(x=h_vals, y=errs_rk4, mode='lines+markers',
                name='RK4 (ordre 4)', line=dict(color='#00ccff', width=2)))
            # Lignes de référence
            fig_conv.add_trace(go.Scatter(x=h_vals, y=h_vals**1 * errs_euler[0]/h_vals[0],
                mode='lines', name='O(h)', line=dict(color='rgba(255,100,0,0.5)', dash='dot')))
            fig_conv.add_trace(go.Scatter(x=h_vals, y=h_vals**4 * errs_rk4[0]/h_vals[0]**4,
                mode='lines', name='O(h⁴)', line=dict(color='rgba(0,200,255,0.5)', dash='dot')))

            fig_conv.update_layout(
                title="Ordre de convergence des méthodes",
                xaxis_title="Pas h", yaxis_title="Erreur |y(T)-y_exact(T)|",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("Le benchmark avec solution exacte est disponible pour les systèmes à solution analytique connue (décroissance, croissance, oscillateur harmonique).")

        st.markdown("#### 📊 Comparaison des méthodes")
        df_meth = pd.DataFrame([
            {"Méthode": k, **v} for k, v in METHODES_NUMERIQUES.items()
        ])
        st.dataframe(df_meth, use_container_width=True)

    # ============================================================
    # TAB 4 : SYSTÈMES SPÉCIAUX
    # ============================================================
    with tab4:
        st.markdown("### 🔬 Systèmes spéciaux")

        special = st.radio("Système", [
            "Sensibilité aux conditions initiales (Lorenz)",
            "Équation de la chaleur 1D (EDP)",
            "SIR — R₀ et seuil épidémique",
        ], horizontal=False)

        if special == "Sensibilité aux conditions initiales (Lorenz)":
            st.markdown("*Effet papillon : deux trajectoires proches divergent exponentiellement*")
            perturbation = st.slider("Perturbation ε", 1e-10, 1e-2, 1e-5, format="%.0e")
            t_lor = np.linspace(0, 40, 5000)

            def lorenz_sys(Y, t):
                return [10*(Y[1]-Y[0]), Y[0]*(28-Y[2])-Y[1], Y[0]*Y[1]-(8/3)*Y[2]]

            sol1 = odeint(lorenz_sys, [1.0, 1.0, 1.0], t_lor)
            sol2 = odeint(lorenz_sys, [1.0+perturbation, 1.0, 1.0], t_lor)
            dist = np.sqrt(np.sum((sol1 - sol2)**2, axis=1))

            fig_s = make_subplots(rows=2, cols=1,
                subplot_titles=["x(t) — deux trajectoires", "Distance ||Δy(t)||"])
            fig_s.add_trace(go.Scatter(x=t_lor, y=sol1[:,0], name='Trajectoire 1',
                line=dict(color='#00ccff', width=1.5)), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=t_lor, y=sol2[:,0], name=f'Trajectoire 2 (ε={perturbation:.0e})',
                line=dict(color='#ff00cc', width=1.5, dash='dot')), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=t_lor, y=dist, name='Distance',
                line=dict(color='#ffcc00', width=2)), row=2, col=1)
            fig_s.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=520,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_s.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_s.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_s, use_container_width=True)

        elif special == "Équation de la chaleur 1D (EDP)":
            st.markdown("*Discrétisation par différences finies*")
            alpha_ch = st.slider("Diffusivité α", 0.01, 1.0, 0.1, 0.01)
            Nx = st.slider("Nx (points spatiaux)", 20, 100, 40)
            Nt = st.slider("Nt (pas temporels)", 100, 2000, 500)

            L_ch = 1.0
            T_max_ch = 1.0
            dx = L_ch / (Nx - 1)
            dt = T_max_ch / Nt
            r = alpha_ch * dt / dx**2

            if r > 0.5:
                st.warning(f"⚠️ r = {r:.3f} > 0.5 → schéma instable! Réduire α ou dt.")

            x_ch = np.linspace(0, L_ch, Nx)
            u = np.sin(np.pi * x_ch)  # CI
            u_exact = lambda t: np.sin(np.pi * x_ch) * np.exp(-alpha_ch * np.pi**2 * t)

            snapshots = {0.0: u.copy()}
            for n in range(Nt):
                u_new = u.copy()
                u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
                u_new[0] = u_new[-1] = 0
                u = u_new
                if n in [Nt//4, Nt//2, 3*Nt//4, Nt-1]:
                    snapshots[n*dt] = u.copy()

            fig_ch = go.Figure()
            colors_ch = ['#00ccff', '#7700ff', '#ff00cc', '#00ff88', '#ffcc00']
            for i, (t_snap, u_snap) in enumerate(snapshots.items()):
                fig_ch.add_trace(go.Scatter(
                    x=x_ch, y=u_snap, mode='lines',
                    name=f't={t_snap:.2f}s (num.)',
                    line=dict(color=colors_ch[i%len(colors_ch)], width=2)
                ))
                u_ex = u_exact(t_snap)
                fig_ch.add_trace(go.Scatter(
                    x=x_ch, y=u_ex, mode='lines',
                    name=f't={t_snap:.2f}s (exact)',
                    line=dict(color=colors_ch[i%len(colors_ch)], width=1.5, dash='dot')
                ))
            fig_ch.update_layout(
                title=f"Équation de la chaleur 1D — α={alpha_ch}, r={r:.3f}",
                xaxis_title="x", yaxis_title="u(x,t)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig_ch, use_container_width=True)

        elif special == "SIR — R₀ et seuil épidémique":
            st.markdown("*Nombre de reproduction de base R₀ et dynamique épidémique*")
            beta_sir = st.slider("β (taux de contact)", 0.05, 1.0, 0.3, 0.01)
            gamma_sir = st.slider("γ (taux de guérison)", 0.01, 0.5, 0.1, 0.01)
            N_sir = st.slider("Population N", 100, 100000, 10000, 100)
            I0_sir = st.slider("Infectés initiaux I₀", 1, 100, 10)

            R0 = beta_sir / gamma_sir
            st.metric("Nombre de reproduction R₀", f"{R0:.3f}")
            st.metric("Statut", "🔴 Épidémie" if R0 > 1 else "🟢 Extinction")
            st.metric("Seuil immunité collective (%)", f"{(1 - 1/R0)*100:.1f}" if R0 > 1 else "N/A")

            t_sir = np.linspace(0, 200, 5000)
            def sir_model(Y, t):
                S, I, R = Y
                dS = -beta_sir * S * I / N_sir
                dI = beta_sir * S * I / N_sir - gamma_sir * I
                dR = gamma_sir * I
                return [dS, dI, dR]

            sol_sir = odeint(sir_model, [N_sir - I0_sir, I0_sir, 0], t_sir)
            fig_sir = go.Figure()
            for i, (lab, col) in enumerate([("S", '#00ccff'), ("I", '#ff4444'), ("R", '#00ff88')]):
                fig_sir.add_trace(go.Scatter(x=t_sir, y=sol_sir[:,i], mode='lines',
                    name=lab, line=dict(color=col, width=2.5)))
            fig_sir.update_layout(
                title=f"Modèle SIR — R₀={R0:.2f}",
                xaxis_title="Jours", yaxis_title="Population",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=450,
            )
            st.plotly_chart(fig_sir, use_container_width=True)

    # ============================================================
    # TAB 5 : THÉORIE
    # ============================================================
    with tab5:
        st.markdown("### 📖 Formulaire scientifique ODE/EDP")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📊 Tableau des méthodes numériques")
        df_meth = pd.DataFrame([{"Méthode": k, **v} for k, v in METHODES_NUMERIQUES.items()])
        st.dataframe(df_meth, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Hairer & Wanner — *Solving ODEs I & II* (Springer, 2008)",
            "Strogatz — *Nonlinear Dynamics and Chaos* (Westview, 2014)",
            "Press et al. — *Numerical Recipes* (Cambridge, 2007)",
            "Murray — *Mathematical Biology I & II* (Springer, 2002)",
        ]:
            st.markdown(f"- {r}")