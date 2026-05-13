import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo, basinhopping
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE SCIENTIFIQUE
# ============================================================
FORMULES = {
    "Gradient": r"\nabla f(x) = 0 \quad \text{(condition nécessaire)}",
    "Hessienne": r"H = \nabla^2 f(x) \succ 0 \quad \text{(minimum local)}",
    "Newton": r"x_{k+1} = x_k - H^{-1}\nabla f(x_k)",
    "Gradient conjugué": r"x_{k+1} = x_k - \alpha_k \nabla f(x_k)",
    "Lagrangien": r"\mathcal{L}(x,\lambda) = f(x) - \lambda g(x)",
    "KKT": r"\nabla f(x^*) = \lambda \nabla g(x^*)",
    "Rastrigin": r"f(x) = 10n + \sum_{i=1}^n [x_i^2 - 10\cos(2\pi x_i)]",
    "Rosenbrock": r"f(x,y) = (1-x)^2 + 100(y-x^2)^2",
    "Ackley": r"f(x) = -20e^{-0.2\sqrt{\frac{1}{n}\sum x_i^2}} - e^{\frac{1}{n}\sum\cos(2\pi x_i)} + e + 20",
}

METHODES_INFO = {
    "Nelder-Mead": {
        "type": "Sans gradient", "ordre": 0,
        "avantage": "Robuste, sans gradient", "inconvénient": "Lent en haute dimension",
        "convergence": "O(n²)", "use_case": "Fonctions bruitées"
    },
    "BFGS": {
        "type": "Quasi-Newton", "ordre": 2,
        "avantage": "Rapide, super-linéaire", "inconvénient": "Nécessite gradient",
        "convergence": "Super-linéaire", "use_case": "Fonctions lisses"
    },
    "L-BFGS-B": {
        "type": "Quasi-Newton limité", "ordre": 2,
        "avantage": "Économique en mémoire", "inconvénient": "Moins précis que BFGS",
        "convergence": "Super-linéaire", "use_case": "Grande dimension"
    },
    "Powell": {
        "type": "Directions conjuguées", "ordre": 0,
        "avantage": "Sans gradient, efficace", "inconvénient": "Peut stagner",
        "convergence": "Super-linéaire", "use_case": "Général"
    },
    "CG": {
        "type": "Gradient conjugué", "ordre": 1,
        "avantage": "Faible mémoire", "inconvénient": "Sensible au conditionnement",
        "convergence": "Linéaire", "use_case": "Problèmes quadratiques"
    },
    "Évolution différentielle": {
        "type": "Évolutionnaire global", "ordre": 0,
        "avantage": "Optimisation globale", "inconvénient": "Lent",
        "convergence": "Stochastique", "use_case": "Non-convexe multimodal"
    },
    "Recuit simulé": {
        "type": "Métaheuristique", "ordre": 0,
        "avantage": "Échappe aux minima locaux", "inconvénient": "Paramétrage complexe",
        "convergence": "Probabiliste", "use_case": "Optimisation globale"
    },
    "Basin-hopping": {
        "type": "Hybride global", "ordre": 1,
        "avantage": "Combine local + global", "inconvénient": "Coûteux",
        "convergence": "Stochastique", "use_case": "Paysages complexes"
    },
}

# ============================================================
# BIBLIOTHÈQUE DE FONCTIONS TEST
# ============================================================
class FunctionLibrary:
    """Bibliothèque de fonctions de test standard en optimisation."""

    @staticmethod
    def quadratique(x, a=1.0, b=0.0, c=0.0):
        return a * x[0]**2 + b * x[0] + c

    @staticmethod
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    @staticmethod
    def rastrigin(x):
        n = len(x)
        return 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)

    @staticmethod
    def ackley(x):
        n = len(x)
        sum1 = np.sqrt(0.5 * sum(xi**2 for xi in x))
        sum2 = 0.5 * sum(np.cos(2*np.pi*xi) for xi in x)
        return -20*np.exp(-0.2*sum1) - np.exp(sum2) + np.e + 20

    @staticmethod
    def sphere(x):
        return sum(xi**2 for xi in x)

    @staticmethod
    def himmelblau(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    @staticmethod
    def beale(x):
        return ((1.5 - x[0] + x[0]*x[1])**2 +
                (2.25 - x[0] + x[0]*x[1]**2)**2 +
                (2.625 - x[0] + x[0]*x[1]**3)**2)

    @staticmethod
    def booth(x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    @staticmethod
    def matyas(x):
        return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

    @staticmethod
    def sinusoide(x):
        return np.sin(x[0]) + 0.1*x[0]**2

    MINIMA_CONNUS = {
        "Rosenbrock": {"x*": [1.0, 1.0], "f*": 0.0},
        "Rastrigin": {"x*": [0.0, 0.0], "f*": 0.0},
        "Ackley": {"x*": [0.0, 0.0], "f*": 0.0},
        "Sphère": {"x*": [0.0, 0.0], "f*": 0.0},
        "Himmelblau": {"x*": [3.0, 2.0], "f*": 0.0},
        "Beale": {"x*": [3.0, 0.5], "f*": 0.0},
        "Booth": {"x*": [1.0, 3.0], "f*": 0.0},
        "Matyas": {"x*": [0.0, 0.0], "f*": 0.0},
    }


# ============================================================
# MOTEUR D'OPTIMISATION
# ============================================================
class OptimisationEngine:
    """Moteur d'optimisation avec sélection automatique de méthode."""

    def __init__(self):
        self.historique = []
        self.n_iterations = 0

    def callback(self, xk):
        self.historique.append(xk.copy() if hasattr(xk, 'copy') else xk)
        self.n_iterations += 1

    def resoudre(self, func, x0, methode, bounds=None):
        self.historique = []
        self.n_iterations = 0

        try:
            if methode == "Évolution différentielle":
                b = bounds or [(-5, 5)] * len(x0)
                res = differential_evolution(
                    func, b,
                    callback=lambda xk, convergence: self.historique.append(xk.copy()),
                    maxiter=1000, tol=1e-8, seed=42
                )
            elif methode == "Recuit simulé":
                b = bounds or [(-5, 5)] * len(x0)
                res = dual_annealing(
                    func, b,
                    callback=lambda x, f, ctx: self.historique.append(x.copy()),
                    maxiter=1000, seed=42
                )
            elif methode == "Basin-hopping":
                res = basinhopping(
                    func, x0,
                    callback=self.callback,
                    niter=200, stepsize=0.5,
                    minimizer_kwargs={"method": "L-BFGS-B"}
                )
            else:
                res = minimize(
                    func, x0,
                    method=methode,
                    callback=self.callback,
                    options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10}
                )
            return res
        except Exception as e:
            return None

    def gradient_numerique(self, func, x, h=1e-6):
        """Gradient numérique par différences finies centrées."""
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (func(xp) - func(xm)) / (2*h)
        return grad

    def hessienne_numerique(self, func, x, h=1e-5):
        """Hessienne numérique par différences finies centrées."""
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                xpp = x.copy(); xpp[i] += h; xpp[j] += h
                xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                H[i,j] = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4*h**2)
        return H

    def analyser_point(self, func, x_opt):
        """Analyse du point optimal : gradient, hessienne, conditionnement."""
        grad = self.gradient_numerique(func, x_opt)
        H = self.hessienne_numerique(func, x_opt)
        try:
            eigenvalues = np.linalg.eigvals(H)
            cond = np.linalg.cond(H)
            type_point = (
                "Minimum local" if np.all(eigenvalues > 0) else
                "Maximum local" if np.all(eigenvalues < 0) else
                "Point selle"
            )
        except:
            eigenvalues = np.array([np.nan])
            cond = np.nan
            type_point = "Indéterminé"
        return {
            "gradient_norme": float(np.linalg.norm(grad)),
            "hessienne": H,
            "valeurs_propres": eigenvalues,
            "conditionnement": cond,
            "type_point": type_point
        }

    def benchmark_methodes(self, func, x0, methodes):
        """Compare plusieurs méthodes sur la même fonction."""
        resultats = []
        for m in methodes:
            engine = OptimisationEngine()
            res = engine.resoudre(func, x0.copy(), m)
            if res is not None:
                resultats.append({
                    "Méthode": m,
                    "f(x*)": f"{res.fun:.6e}",
                    "Itérations": engine.n_iterations,
                    "Succès": "✅" if res.success else "❌",
                    "x*": str(np.round(res.x, 4))
                })
        return pd.DataFrame(resultats)


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def optimisation_page():
    st.markdown("## 🎯 Optimisation Scientifique Avancée")
    st.markdown("*Minimisation, analyse de paysage, benchmark de solveurs*")
    st.markdown("---")

    lib = FunctionLibrary()
    engine = OptimisationEngine()

    # Onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Optimisation 1D/2D",
        "🌐 Paysage 2D",
        "⚔️ Benchmark",
        "📐 Analyse mathématique",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : OPTIMISATION PRINCIPALE
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Configuration")

            mode = st.radio("Dimension", ["1D", "2D"], horizontal=True)

            fonctions_2d = {
                "Rosenbrock": lib.rosenbrock,
                "Rastrigin": lib.rastrigin,
                "Ackley": lib.ackley,
                "Sphère": lib.sphere,
                "Himmelblau": lib.himmelblau,
                "Beale": lib.beale,
                "Booth": lib.booth,
                "Matyas": lib.matyas,
            }
            fonctions_1d = {
                "Quadratique": None,
                "Sinusoïde": lib.sinusoide,
            }

            if mode == "1D":
                func_name = st.selectbox("Fonction", list(fonctions_1d.keys()))
                if func_name == "Quadratique":
                    a = st.slider("a", -5.0, 5.0, 1.0, 0.1)
                    b = st.slider("b", -5.0, 5.0, -2.0, 0.1)
                    c = st.slider("c", -5.0, 5.0, 3.0, 0.1)
                    func = lambda x: a*x[0]**2 + b*x[0] + c
                    x_min_ana = -b / (2*a) if a != 0 else 0
                    st.markdown(f"**Minimum analytique :** x* = {x_min_ana:.4f}")
                    st.markdown(f"**f(x*) = ** {func([x_min_ana]):.4f}")
                else:
                    func = fonctions_1d[func_name]

                x0_1d = st.slider("Point initial x₀", -8.0, 8.0, 2.0, 0.1)
                x0 = np.array([x0_1d])

            else:
                func_name = st.selectbox("Fonction 2D", list(fonctions_2d.keys()))
                func = fonctions_2d[func_name]

                if func_name in lib.MINIMA_CONNUS:
                    info = lib.MINIMA_CONNUS[func_name]
                    st.info(f"**Minimum global connu :** x*={info['x*']}, f*={info['f*']}")

                c1, c2 = st.columns(2)
                with c1:
                    x0_x = st.slider("x₀", -4.0, 4.0, 2.0, 0.1)
                with c2:
                    x0_y = st.slider("y₀", -4.0, 4.0, 2.0, 0.1)
                x0 = np.array([x0_x, x0_y])

            st.markdown("### 🔧 Solveur")
            methodes_locales = ["Nelder-Mead", "BFGS", "L-BFGS-B", "Powell", "CG"]
            methodes_globales = ["Évolution différentielle", "Recuit simulé", "Basin-hopping"]
            toutes = methodes_locales + methodes_globales
            methode = st.selectbox("Méthode", toutes)

            if methode in METHODES_INFO:
                info_m = METHODES_INFO[methode]
                st.markdown(f"""
                - **Type :** {info_m['type']}
                - **Convergence :** {info_m['convergence']}
                - **✅ Avantage :** {info_m['avantage']}
                - **⚠️ Limite :** {info_m['inconvénient']}
                """)

            lancer = st.button("🚀 Lancer l'optimisation", use_container_width=True)

        with col2:
            if lancer:
                with st.spinner("Optimisation en cours..."):
                    res = engine.resoudre(func, x0.copy(), methode)

                if res is not None:
                    st.session_state.res_opt = res
                    st.session_state.engine_opt = engine
                    st.session_state.func_opt = func
                    st.session_state.x0_opt = x0
                    st.session_state.mode_opt = mode

                    # --- Résultats ---
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("f(x*) minimum", f"{res.fun:.6e}")
                    with c2:
                        st.metric("Itérations", engine.n_iterations)
                    with c3:
                        st.metric("Statut", "✅ Convergé" if res.success else "⚠️ Non convergé")

                    if mode == "1D":
                        x_plot = np.linspace(-10, 10, 1000)
                        y_plot = np.array([func([xi]) for xi in x_plot])

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=x_plot, y=y_plot, mode='lines',
                            name='f(x)', line=dict(color='#00ccff', width=3)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[res.x[0]], y=[res.fun], mode='markers',
                            name='Minimum', marker=dict(color='#00ff88', size=16, symbol='star')
                        ))
                        fig.add_trace(go.Scatter(
                            x=[x0[0]], y=[func(x0)], mode='markers',
                            name='Départ', marker=dict(color='#ffcc00', size=12, symbol='circle')
                        ))

                        # Historique convergence
                        if len(engine.historique) > 1:
                            hist_x = [h[0] if hasattr(h, '__len__') else h for h in engine.historique]
                            hist_y = [func([h]) for h in hist_x]
                            fig.add_trace(go.Scatter(
                                x=hist_x, y=hist_y, mode='markers+lines',
                                name='Trajectoire', opacity=0.5,
                                line=dict(color='#ff00cc', width=1, dash='dot'),
                                marker=dict(size=5, color='#ff00cc')
                            ))

                        fig.update_layout(
                            title=f"Optimisation 1D — {func_name}",
                            xaxis_title="x", yaxis_title="f(x)",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(5,0,20,0.8)',
                            font=dict(color='#c0d0ff'),
                            xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                            yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                            legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                            height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # Paysage 2D + trajectoire
                        resolution = 80
                        x_r = np.linspace(-5, 5, resolution)
                        y_r = np.linspace(-5, 5, resolution)
                        Z = np.array([[func([xi, yi]) for xi in x_r] for yi in y_r])
                        Z_clipped = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 98))

                        fig = go.Figure()
                        fig.add_trace(go.Contour(
                            z=Z_clipped, x=x_r, y=y_r,
                            colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                            contours=dict(coloring='heatmap', showlabels=False),
                            showscale=True,
                            colorbar=dict(tickfont=dict(color='#c0d0ff'), title="f(x,y)")
                        ))

                        # Trajectoire
                        if len(engine.historique) > 1:
                            traj = np.array(engine.historique)
                            if traj.ndim == 2 and traj.shape[1] >= 2:
                                fig.add_trace(go.Scatter(
                                    x=traj[:,0], y=traj[:,1], mode='lines+markers',
                                    name='Trajectoire',
                                    line=dict(color='#ffcc00', width=2),
                                    marker=dict(size=4, color='#ffcc00')
                                ))

                        fig.add_trace(go.Scatter(
                            x=[x0[0]], y=[x0[1]], mode='markers',
                            name='Départ', marker=dict(color='#ffcc00', size=14, symbol='circle')
                        ))
                        fig.add_trace(go.Scatter(
                            x=[res.x[0]], y=[res.x[1]], mode='markers',
                            name='Minimum', marker=dict(color='#00ff88', size=16, symbol='star')
                        ))

                        if func_name in lib.MINIMA_CONNUS:
                            xstar = lib.MINIMA_CONNUS[func_name]["x*"]
                            fig.add_trace(go.Scatter(
                                x=[xstar[0]], y=[xstar[1]], mode='markers',
                                name='Vrai minimum', marker=dict(color='#ff0000', size=14, symbol='x')
                            ))

                        fig.update_layout(
                            title=f"Paysage 2D — {func_name}",
                            xaxis_title="x", yaxis_title="y",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(5,0,20,0.8)',
                            font=dict(color='#c0d0ff'),
                            xaxis=dict(color='#c0d0ff'),
                            yaxis=dict(color='#c0d0ff'),
                            legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                            height=520,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Courbe de convergence
                    if len(engine.historique) > 1:
                        st.markdown("### 📉 Courbe de convergence")
                        try:
                            conv_vals = []
                            for h in engine.historique:
                                hh = np.array(h).flatten()
                                if len(hh) >= len(x0):
                                    conv_vals.append(func(hh[:len(x0)]))
                            if conv_vals:
                                fig_conv = go.Figure(go.Scatter(
                                    y=conv_vals, mode='lines+markers',
                                    line=dict(color='#00ccff', width=2),
                                    marker=dict(size=3)
                                ))
                                fig_conv.update_layout(
                                    xaxis_title="Itération", yaxis_title="f(x)",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(5,0,20,0.8)',
                                    font=dict(color='#c0d0ff'),
                                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                              type='log'),
                                    height=300,
                                )
                                st.plotly_chart(fig_conv, use_container_width=True)
                        except:
                            pass

                    # Analyse du point optimal
                    st.markdown("### 🔬 Analyse du point optimal")
                    analyse = engine.analyser_point(func, res.x)
                    ca, cb, cc = st.columns(3)
                    with ca:
                        st.metric("‖∇f(x*)‖", f"{analyse['gradient_norme']:.2e}")
                    with cb:
                        st.metric("Type de point", analyse['type_point'])
                    with cc:
                        st.metric("Conditionnement H", f"{analyse['conditionnement']:.2e}"
                                  if not np.isnan(analyse['conditionnement']) else "N/A")

                    eigvals = analyse['valeurs_propres']
                    if not np.any(np.isnan(eigvals)):
                        st.markdown("**Valeurs propres de la Hessienne :**")
                        cols_eig = st.columns(min(len(eigvals), 4))
                        for i, (col, ev) in enumerate(zip(cols_eig, eigvals)):
                            with col:
                                st.metric(f"λ{i+1}", f"{ev.real:.4e}")

    # ============================================================
    # TAB 2 : PAYSAGE 3D
    # ============================================================
    with tab2:
        st.markdown("### 🌐 Visualisation du paysage 3D")
        col1, col2 = st.columns([1, 3])

        with col1:
            fonctions_viz = {
                "Rosenbrock": lib.rosenbrock,
                "Rastrigin": lib.rastrigin,
                "Ackley": lib.ackley,
                "Sphère": lib.sphere,
                "Himmelblau": lib.himmelblau,
                "Beale": lib.beale,
                "Booth": lib.booth,
                "Matyas": lib.matyas,
            }
            func_viz_name = st.selectbox("Fonction", list(fonctions_viz.keys()), key="viz3d")
            func_viz = fonctions_viz[func_viz_name]
            range_viz = st.slider("Plage [-R, R]", 1.0, 6.0, 3.0, 0.5, key="range3d")
            res_viz = st.slider("Résolution", 30, 120, 60, key="res3d")
            mode_viz = st.radio("Vue", ["Surface 3D", "Contour 2D"], horizontal=True)

        with col2:
            x_v = np.linspace(-range_viz, range_viz, res_viz)
            y_v = np.linspace(-range_viz, range_viz, res_viz)
            Z_v = np.array([[func_viz([xi, yi]) for xi in x_v] for yi in y_v])
            Z_clip = np.clip(Z_v, np.percentile(Z_v, 1), np.percentile(Z_v, 99))

            if mode_viz == "Surface 3D":
                fig3d = go.Figure(data=[go.Surface(
                    z=Z_clip, x=x_v, y=y_v,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5),
                )])
                fig3d.update_layout(
                    scene=dict(
                        bgcolor='rgba(5,0,20,0.9)',
                        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        zaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c0d0ff'),
                    height=550,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
            else:
                fig3d = go.Figure(data=[go.Contour(
                    z=Z_clip, x=x_v, y=y_v,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    contours=dict(coloring='heatmap', showlabels=True,
                                 labelfont=dict(size=9, color='white')),
                    colorbar=dict(tickfont=dict(color='#c0d0ff'))
                )])
                fig3d.update_layout(
                    xaxis_title="x", yaxis_title="y",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff'),
                    yaxis=dict(color='#c0d0ff'),
                    height=550,
                )

            if func_viz_name in lib.MINIMA_CONNUS:
                xstar = lib.MINIMA_CONNUS[func_viz_name]["x*"]
                if mode_viz == "Contour 2D":
                    fig3d.add_trace(go.Scatter(
                        x=[xstar[0]], y=[xstar[1]], mode='markers',
                        name='Minimum global',
                        marker=dict(color='#ff0000', size=14, symbol='x')
                    ))

            st.plotly_chart(fig3d, use_container_width=True)

    # ============================================================
    # TAB 3 : BENCHMARK
    # ============================================================
    with tab3:
        st.markdown("### ⚔️ Benchmark des méthodes")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_bench_name = st.selectbox("Fonction", list(fonctions_2d.keys()), key="bench_func")
            func_bench = fonctions_2d[func_bench_name]
            methodes_bench = st.multiselect(
                "Méthodes à comparer",
                ["Nelder-Mead", "BFGS", "L-BFGS-B", "Powell", "CG",
                 "Évolution différentielle", "Recuit simulé"],
                default=["Nelder-Mead", "BFGS", "L-BFGS-B", "Powell"]
            )
            bx0 = st.slider("x₀ benchmark", -4.0, 4.0, 2.0, 0.1, key="bench_x0")
            by0 = st.slider("y₀ benchmark", -4.0, 4.0, 2.0, 0.1, key="bench_y0")
            x0_bench = np.array([bx0, by0])

            lancer_bench = st.button("⚔️ Lancer le benchmark", use_container_width=True)

        with col2:
            if lancer_bench and methodes_bench:
                with st.spinner("Benchmark en cours..."):
                    df_bench = engine.benchmark_methodes(func_bench, x0_bench, methodes_bench)

                st.dataframe(
                    df_bench.style.applymap(
                        lambda v: 'color: #00ff88' if v == '✅' else 'color: #ff4444' if v == '❌' else '',
                    ),
                    use_container_width=True
                )

                # Barplot résultats
                fig_b = go.Figure()
                for _, row in df_bench.iterrows():
                    try:
                        val = float(row["f(x*)"])
                        fig_b.add_trace(go.Bar(
                            x=[row["Méthode"]], y=[abs(val) + 1e-10],
                            name=row["Méthode"],
                            marker_color='#7700ff' if row["Succès"] == "✅" else '#ff4444'
                        ))
                    except:
                        pass

                fig_b.update_layout(
                    title="Comparaison |f(x*)| par méthode",
                    yaxis_type='log', yaxis_title="|f(x*)| (log)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    showlegend=False,
                    height=380,
                )
                st.plotly_chart(fig_b, use_container_width=True)

    # ============================================================
    # TAB 4 : ANALYSE MATHÉMATIQUE
    # ============================================================
    with tab4:
        st.markdown("### 📐 Analyse mathématique")
        st.markdown("#### Tableau des méthodes")

        df_methodes = pd.DataFrame([
            {
                "Méthode": k,
                "Type": v["type"],
                "Ordre": v["ordre"],
                "Convergence": v["convergence"],
                "Avantage": v["avantage"],
                "Cas d'usage": v["use_case"]
            }
            for k, v in METHODES_INFO.items()
        ])
        st.dataframe(df_methodes, use_container_width=True)

        st.markdown("#### 🔢 Conditions d'optimalité")
        conditions = {
            "1er ordre (nécessaire)": r"\nabla f(x^*) = 0",
            "2ème ordre (suffisante)": r"H(x^*) \succ 0 \quad \text{(définie positive)}",
            "Contrainte égalité (KKT)": r"\nabla f = \lambda \nabla g",
            "Contrainte inégalité": r"\mu_i g_i(x^*) = 0, \quad \mu_i \geq 0",
        }
        for nom, formule in conditions.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("#### 📊 Diagnostic de convergence")
        diag_data = {
            "Problème": ["Non convergence", "Convergence lente", "Instabilité", "Minimum local", "Ill-conditioning"],
            "Symptôme": ["nit = maxiter", "‖∇f‖ décroît lentement", "f oscille", "f* > f_global*", "κ(H) >> 1"],
            "Cause": ["Pas trop grand", "Mauvais α", "Pas trop grand", "Départ local", "Échelle mal calibrée"],
            "Solution": ["Réduire tol/step", "Line search adaptif", "Réduire step", "Méthode globale", "Normaliser les variables"]
        }
        st.dataframe(pd.DataFrame(diag_data), use_container_width=True)

    # ============================================================
    # TAB 5 : THÉORIE
    # ============================================================
    with tab5:
        st.markdown("### 📖 Formulaire Scientifique")
        for nom, formule in FORMULES.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("---")
        st.markdown("### 📚 Références")
        refs = [
            "Nocedal & Wright — *Numerical Optimization* (Springer, 2006)",
            "Boyd & Vandenberghe — *Convex Optimization* (Cambridge, 2004)",
            "Storn & Price — *Differential Evolution* (J. Global Optimization, 1997)",
            "Kirkpatrick et al. — *Simulated Annealing* (Science, 1983)",
        ]
        for r in refs:
            st.markdown(f"- {r}")