import streamlit as st
import numpy as np
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import quad, dblquad, tplquad, fixed_quad
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORMULAIRE
# ============================================================
FORMULES = {
    "Trapèzes":          r"I \approx \frac{h}{2}\left[f(a)+2\sum_{i=1}^{n-1}f(x_i)+f(b)\right],\quad h=\frac{b-a}{n}",
    "Simpson 1/3":       r"I \approx \frac{h}{3}\left[f(a)+4f(x_1)+2f(x_2)+\cdots+4f(x_{n-1})+f(b)\right]",
    "Simpson 3/8":       r"I \approx \frac{3h}{8}\left[f(a)+3f(x_1)+3f(x_2)+2f(x_3)+\cdots\right]",
    "Gauss-Legendre":    r"I \approx \frac{b-a}{2}\sum_{i=1}^n w_i\,f\!\left(\frac{b-a}{2}x_i+\frac{a+b}{2}\right)",
    "Monte Carlo":       r"I \approx (b-a)\cdot\frac{1}{N}\sum_{i=1}^N f(X_i),\quad X_i\sim\mathcal{U}(a,b)",
    "Romberg":           r"R_{n,m}=\frac{4^m R_{n,m-1}-R_{n-1,m-1}}{4^m-1}",
    "Erreur Trapèzes":   r"|E_T|\leq\frac{(b-a)^3}{12n^2}\max|f''|",
    "Erreur Simpson":    r"|E_S|\leq\frac{(b-a)^5}{180n^4}\max|f^{(4)}|",
    "Erreur Gauss(n)":   r"|E_G|\leq\frac{(b-a)^{2n+1}(n!)^4}{(2n+1)[(2n)!]^3}\max|f^{(2n)}|",
    "Intégrale double":  r"\iint_D f(x,y)\,dx\,dy\approx\sum_{i,j}w_i w_j f(x_i,y_j)",
}

FONCTIONS_PREDEF = {
    "x²":                   (lambda x: x**2,           "Polynôme",       True),
    "sin(x)":               (np.sin,                    "Trigonométrique",True),
    "exp(-x²)":             (lambda x: np.exp(-x**2),  "Gaussienne",     True),
    "cos(x)":               (np.cos,                    "Trigonométrique",True),
    "1/(1+x²)":             (lambda x: 1/(1+x**2),     "Rationnelle",    True),
    "x·sin(x)":             (lambda x: x*np.sin(x),    "Mixte",          True),
    "√(1-x²)":             (lambda x: np.sqrt(np.maximum(1-x**2, 0)),
                                                          "Semi-circulaire",True),
    "ln(x)":                (lambda x: np.where(x>0, np.log(x), 0),
                                                          "Logarithmique",  False),
    "1/√x":                 (lambda x: np.where(x>1e-10, 1/np.sqrt(x), 0),
                                                          "Singulière",     False),
    "sin(1/x)":             (lambda x: np.where(np.abs(x)>1e-8, np.sin(1/x), 0),
                                                          "Oscillante",     False),
    "exp(-x)·cos(10x)":    (lambda x: np.exp(-x)*np.cos(10*x),
                                                          "Oscillante amorties",True),
    "Heaviside(x-0.5)":    (lambda x: np.where(x>=0.5, 1.0, 0.0),
                                                          "Discontinue",    True),
}

METHODES_INFO = {
    "Trapèzes":        {"ordre": 2, "formule": "Open",   "complexité": "O(n)"},
    "Simpson 1/3":     {"ordre": 4, "formule": "Closed", "complexité": "O(n)"},
    "Simpson 3/8":     {"ordre": 4, "formule": "Closed", "complexité": "O(n)"},
    "Gauss-Legendre":  {"ordre": "2n-1","formule": "Gauss","complexité": "O(n)"},
    "Monte Carlo":     {"ordre": "½",  "formule": "Stoch.","complexité": "O(n)"},
    "Romberg":         {"ordre": "∞", "formule": "Adapt.","complexité": "O(n²)"},
    "SciPy (quad)":    {"ordre": "adapt.","formule": "QUADPACK","complexité": "adapt."},
}


# ============================================================
# SOLVEURS D'INTÉGRATION
# ============================================================
class IntegrationSolver:
    """Solveurs numériques d'intégration avancés."""

    @staticmethod
    def trapeze(f, a, b, n):
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

    @staticmethod
    def simpson_13(f, a, b, n):
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

    @staticmethod
    def simpson_38(f, a, b, n):
        if n % 3 != 0:
            n += 3 - n % 3
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        S = y[0] + y[-1]
        for i in range(1, n):
            S += (3 if i % 3 != 0 else 2) * y[i]
        return 3*h/8 * S

    @staticmethod
    def gauss_legendre(f, a, b, n=10):
        xi, wi = np.polynomial.legendre.leggauss(n)
        x = 0.5*(xi+1)*(b-a) + a
        return np.sum(wi * f(x)) * 0.5*(b-a)

    @staticmethod
    def monte_carlo(f, a, b, n, seed=42):
        np.random.seed(seed)
        x = np.random.uniform(a, b, n)
        return (b - a) * np.mean(f(x))

    @staticmethod
    def romberg_method(f, a, b, max_order=6):
        try:
            R = np.zeros((max_order, max_order), dtype=float)
            for k in range(max_order):
                n = 2**k
                x = np.linspace(a, b, n + 1)
                y = f(x)
                R[k, 0] = np.trapz(y, x)
                for j in range(1, k + 1):
                    R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
            return float(R[max_order - 1, max_order - 1])
        except Exception:
            return None

    @staticmethod
    def scipy_quad(f, a, b):
        try:
            val, err = quad(f, a, b, limit=200, epsabs=1e-12)
            return val, err
        except:
            return None, None

    @staticmethod
    def integrale_double(f2d, xa, xb, ya, yb, nx=20, ny=20):
        """Intégrale double par règle des trapèzes 2D."""
        x = np.linspace(xa, xb, nx)
        y = np.linspace(ya, yb, ny)
        X, Y = np.meshgrid(x, y)
        Z = f2d(X, Y)
        dx = (xb-xa)/(nx-1)
        dy = (yb-ya)/(ny-1)
        return np.trapz(np.trapz(Z, dx=dy), dx=dx)

    @staticmethod
    def convergence_etude(f, a, b, methode: str,
                          ref: float, n_range: np.ndarray) -> np.ndarray:
        """Erreur de convergence pour N variable."""
        solver = IntegrationSolver()
        erreurs = []
        for n in n_range:
            n = int(n)
            try:
                if methode == "Trapèzes":
                    val = solver.trapeze(f, a, b, n)
                elif methode == "Simpson 1/3":
                    val = solver.simpson_13(f, a, b, n)
                elif methode == "Gauss-Legendre":
                    val = solver.gauss_legendre(f, a, b, n)
                else:
                    val = solver.monte_carlo(f, a, b, n)
                erreurs.append(abs(val - ref))
            except:
                erreurs.append(np.nan)
        return np.array(erreurs)


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def integration_page():
    st.markdown("## ∫ Intégration Numérique Avancée")
    st.markdown("*Méthodes classiques, Gauss, Monte Carlo, Romberg, intégrale double, convergence*")
    st.markdown("---")

    solver = IntegrationSolver()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔢 Intégration 1D",
        "⚔️ Benchmark méthodes",
        "📉 Convergence",
        "🌐 Intégrale double",
        "🎲 Monte Carlo avancé",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : INTÉGRATION 1D
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")

            func_name = st.selectbox("Fonction", list(FONCTIONS_PREDEF.keys()))
            f_info = FONCTIONS_PREDEF[func_name]
            f = f_info[0]
            st.caption(f"Catégorie : *{f_info[1]}*")

            a = st.number_input("Borne a", value=0.0, step=0.1)
            b = st.number_input("Borne b", value=1.0, step=0.1)
            if b <= a:
                st.error("b doit être > a")
                st.stop()

            n = st.slider("Discrétisation n", 4, 2000, 50)

            methodes_sel = st.multiselect("Méthodes",
                list(METHODES_INFO.keys()),
                default=["Trapèzes", "Simpson 1/3", "Gauss-Legendre", "SciPy (quad)"]
            )

            show_fill  = st.checkbox("Afficher l'aire", True)
            show_pts   = st.checkbox("Afficher les points de calcul", False)

        with col2:
            # Valeur de référence
            ref_val, ref_err = solver.scipy_quad(f, a, b)
            if ref_val is None:
                st.warning("Référence SciPy indisponible.")
                ref_val = 0.0

            # Graphique
            x_plot = np.linspace(a, b, 2000)
            try:
                y_plot = f(x_plot)
            except:
                y_plot = np.zeros_like(x_plot)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_plot, y=y_plot, mode='lines', name='f(x)',
                line=dict(color='#00ccff', width=3)
            ))
            if show_fill:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[a], x_plot, [b]]),
                    y=np.concatenate([[0], y_plot, [0]]),
                    fill='toself', fillcolor='rgba(119,0,255,0.2)',
                    mode='none', name='Aire intégrée'
                ))
            if show_pts:
                x_pts = np.linspace(a, b, n+1)
                try:
                    y_pts = f(x_pts)
                    fig.add_trace(go.Scatter(
                        x=x_pts, y=y_pts, mode='markers', name='Points n',
                        marker=dict(color='#ff00cc', size=5)
                    ))
                except:
                    pass

            fig.update_layout(
                title=f"Intégration de {func_name} sur [{a:.2f}, {b:.2f}]",
                xaxis_title='x', yaxis_title='f(x)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Calcul et résultats
            resultats = []
            for m in methodes_sel:
                try:
                    if m == "Trapèzes":
                        val = solver.trapeze(f, a, b, n)
                    elif m == "Simpson 1/3":
                        val = solver.simpson_13(f, a, b, n)
                    elif m == "Simpson 3/8":
                        val = solver.simpson_38(f, a, b, n)
                    elif m == "Gauss-Legendre":
                        val = solver.gauss_legendre(f, a, b, min(n, 100))
                    elif m == "Monte Carlo":
                        val = solver.monte_carlo(f, a, b, n*100)
                    elif m == "Romberg":
                        val = solver.romberg_method(f, a, b)
                    else:
                        val, _ = solver.scipy_quad(f, a, b)

                    err = abs(val - ref_val) if val is not None else np.nan
                    resultats.append({
                        "Méthode": m,
                        "Valeur": f"{val:.8f}" if val else "N/A",
                        "Erreur abs.": f"{err:.2e}" if not np.isnan(err) else "N/A",
                        "Erreur rel. (%)": f"{err/abs(ref_val)*100:.4f}" if ref_val and not np.isnan(err) else "N/A",
                    })
                except Exception as e:
                    resultats.append({"Méthode": m, "Valeur": "❌",
                                      "Erreur abs.": str(e)[:30], "Erreur rel. (%)": "N/A"})

            df_res = pd.DataFrame(resultats)
            st.markdown("### 📊 Résultats")
            st.dataframe(df_res, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Référence SciPy", f"{ref_val:.8f}")
            with c2: st.metric("Erreur estimée", f"{ref_err:.2e}" if ref_err else "N/A")
            with c3:
                if resultats:
                    best_idx = np.argmin([float(r["Erreur abs."].replace("N/A","inf").replace("❌","inf"))
                                          if r["Erreur abs."] not in ["N/A","❌"] else np.inf
                                          for r in resultats])
                    st.metric("Meilleure méthode", resultats[best_idx]["Méthode"])

    # ============================================================
    # TAB 2 : BENCHMARK
    # ============================================================
    with tab2:
        st.markdown("### ⚔️ Benchmark toutes méthodes")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_b = st.selectbox("Fonction", list(FONCTIONS_PREDEF.keys()), key="fb")
            a_b = st.number_input("a", value=0.0, key="ab")
            b_b = st.number_input("b", value=1.0, key="bb")
            n_b = st.slider("n", 10, 1000, 100, key="nb")

        with col2:
            f_b = FONCTIONS_PREDEF[func_b][0]
            ref_b, _ = solver.scipy_quad(f_b, a_b, b_b)
            if ref_b is None:
                ref_b = 0.0

            methodes_all = ["Trapèzes", "Simpson 1/3", "Simpson 3/8",
                            "Gauss-Legendre", "Monte Carlo", "Romberg"]
            bench_results = []
            for m in methodes_all:
                try:
                    if m == "Trapèzes":
                        val = solver.trapeze(f_b, a_b, b_b, n_b)
                    elif m == "Simpson 1/3":
                        val = solver.simpson_13(f_b, a_b, b_b, n_b)
                    elif m == "Simpson 3/8":
                        val = solver.simpson_38(f_b, a_b, b_b, n_b)
                    elif m == "Gauss-Legendre":
                        val = solver.gauss_legendre(f_b, a_b, b_b, min(n_b, 100))
                    elif m == "Monte Carlo":
                        val = solver.monte_carlo(f_b, a_b, b_b, n_b*100)
                    else:
                        val = solver.romberg_method(f_b, a_b, b_b)

                    err = abs(val - ref_b) if val is not None else np.nan
                    bench_results.append({
                        "Méthode": m,
                        "Valeur": round(val, 8) if val else None,
                        "Erreur abs.": err,
                        "Ordre": METHODES_INFO[m]["ordre"],
                    })
                except:
                    bench_results.append({"Méthode": m, "Valeur": None,
                                          "Erreur abs.": np.nan, "Ordre": "?"})

            df_bench = pd.DataFrame(bench_results)
            st.dataframe(df_bench[["Méthode","Valeur","Erreur abs.","Ordre"]],
                         use_container_width=True)

            # Barplot erreurs
            fig_b = go.Figure()
            for row in bench_results:
                if not np.isnan(row["Erreur abs."]) and row["Erreur abs."] > 0:
                    fig_b.add_trace(go.Bar(
                        x=[row["Méthode"]], y=[row["Erreur abs."]+1e-16],
                        marker_color='#7700ff', showlegend=False
                    ))
            fig_b.update_layout(
                title="Erreur absolue |val - ref|",
                yaxis_type='log', yaxis_title="|E| (log)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=350,
            )
            st.plotly_chart(fig_b, use_container_width=True)

    # ============================================================
    # TAB 3 : CONVERGENCE
    # ============================================================
    with tab3:
        st.markdown("### 📉 Étude de convergence")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_c = st.selectbox("Fonction", list(FONCTIONS_PREDEF.keys()), key="fc")
            a_c = st.number_input("a", value=0.0, key="ac_conv")
            b_c = st.number_input("b", value=1.0, key="bc_conv")
            methodes_conv = st.multiselect("Méthodes",
                ["Trapèzes", "Simpson 1/3", "Gauss-Legendre", "Monte Carlo"],
                default=["Trapèzes", "Simpson 1/3", "Gauss-Legendre"])

        with col2:
            f_c = FONCTIONS_PREDEF[func_c][0]
            ref_c, _ = solver.scipy_quad(f_c, a_c, b_c)
            if ref_c is None:
                ref_c = 0.0

            n_range = np.logspace(1, 3, 30).astype(int)
            colors_conv = ['#00ccff','#7700ff','#ff00cc','#00ff88']

            fig_conv = go.Figure()
            for i, m in enumerate(methodes_conv):
                errs = solver.convergence_etude(f_c, a_c, b_c, m, ref_c, n_range)
                fig_conv.add_trace(go.Scatter(
                    x=n_range, y=errs+1e-16, mode='lines+markers', name=m,
                    line=dict(color=colors_conv[i%len(colors_conv)], width=2.5),
                    marker=dict(size=5)
                ))

            # Lignes de référence
            n_ref = np.array([n_range[0], n_range[-1]], dtype=float)
            fig_conv.add_trace(go.Scatter(
                x=n_ref, y=1.0/n_ref**2, mode='lines', name='O(h²)',
                line=dict(color='rgba(255,200,0,0.4)', width=2, dash='dot')
            ))
            fig_conv.add_trace(go.Scatter(
                x=n_ref, y=1.0/n_ref**4, mode='lines', name='O(h⁴)',
                line=dict(color='rgba(0,255,136,0.4)', width=2, dash='dot')
            ))
            fig_conv.add_trace(go.Scatter(
                x=n_ref, y=1.0/np.sqrt(n_ref), mode='lines', name='O(1/√N)',
                line=dict(color='rgba(255,100,0,0.4)', width=2, dash='dot')
            ))

            fig_conv.update_layout(
                title=f"Convergence — {func_c}",
                xaxis_title="n (nombre de points)", yaxis_title="|E| (log)",
                xaxis_type='log', yaxis_type='log',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=480,
            )
            st.plotly_chart(fig_conv, use_container_width=True)

    # ============================================================
    # TAB 4 : INTÉGRALE DOUBLE
    # ============================================================
    with tab4:
        st.markdown("### 🌐 Intégrale double")
        col1, col2 = st.columns([1, 2])

        with col1:
            func2d_name = st.selectbox("Fonction f(x,y)", [
                "x² + y²", "sin(x)·cos(y)", "exp(-(x²+y²))",
                "x·y", "sin(x+y)"
            ])
            xa = st.slider("xa", -5.0, 0.0, 0.0, 0.1)
            xb = st.slider("xb", 0.1, 5.0, 1.0, 0.1)
            ya = st.slider("ya", -5.0, 0.0, 0.0, 0.1, key="ya2")
            yb = st.slider("yb", 0.1, 5.0, 1.0, 0.1, key="yb2")
            nx2 = st.slider("nx", 10, 80, 30)
            ny2 = st.slider("ny", 10, 80, 30)

            f2d_map = {
                "x² + y²":         lambda x, y: x**2 + y**2,
                "sin(x)·cos(y)":   lambda x, y: np.sin(x)*np.cos(y),
                "exp(-(x²+y²))":   lambda x, y: np.exp(-(x**2+y**2)),
                "x·y":             lambda x, y: x*y,
                "sin(x+y)":        lambda x, y: np.sin(x+y),
            }
            f2d = f2d_map[func2d_name]

        with col2:
            val2d = solver.integrale_double(f2d, xa, xb, ya, yb, nx2, ny2)

            # Référence SciPy
            try:
                ref2d, _ = dblquad(lambda y, x: f2d(x, y), xa, xb,
                                    lambda x: ya, lambda x: yb)
            except:
                ref2d = None

            c1, c2 = st.columns(2)
            with c1: st.metric("Valeur numérique", f"{val2d:.6f}")
            with c2: st.metric("Référence SciPy", f"{ref2d:.6f}" if ref2d else "N/A")
            if ref2d:
                st.metric("Erreur abs.", f"{abs(val2d-ref2d):.2e}")

            # Surface 3D
            x2d = np.linspace(xa, xb, 60)
            y2d = np.linspace(ya, yb, 60)
            X2d, Y2d = np.meshgrid(x2d, y2d)
            Z2d = f2d(X2d, Y2d)

            fig_2d = go.Figure(data=[go.Surface(
                z=Z2d, x=x2d, y=y2d,
                colorscale=[[0,'#020817'],[0.4,'#7700ff'],[0.7,'#00ccff'],[1,'#ffffff']],
                showscale=True,
                lighting=dict(ambient=0.5, diffuse=0.8),
            )])
            fig_2d.update_layout(
                title=f"∬ {func2d_name} dxdy = {val2d:.4f}",
                scene=dict(
                    bgcolor='rgba(5,0,20,0.9)',
                    xaxis=dict(color='#c0d0ff', title='x'),
                    yaxis=dict(color='#c0d0ff', title='y'),
                    zaxis=dict(color='#c0d0ff', title='f(x,y)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0d0ff'),
                height=480, margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig_2d, use_container_width=True)

    # ============================================================
    # TAB 5 : MONTE CARLO AVANCÉ
    # ============================================================
    with tab5:
        st.markdown("### 🎲 Monte Carlo avancé")
        col1, col2 = st.columns([1, 2])

        with col1:
            func_mc = st.selectbox("Fonction", list(FONCTIONS_PREDEF.keys()), key="fmc")
            a_mc = st.number_input("a", value=0.0, key="amc")
            b_mc = st.number_input("b", value=1.0, key="bmc")
            N_mc = st.slider("N (points Monte Carlo)", 100, 100000, 5000, 100)
            n_runs = st.slider("Répétitions (variance)", 5, 50, 20)
            seed_mc = st.slider("Seed", 0, 999, 42)

        with col2:
            f_mc = FONCTIONS_PREDEF[func_mc][0]
            ref_mc, _ = solver.scipy_quad(f_mc, a_mc, b_mc)
            if ref_mc is None:
                ref_mc = 0.0

            # Convergence Monte Carlo
            N_range = np.geomspace(10, N_mc, 40).astype(int)
            mc_vals = []
            for n_iter in N_range:
                np.random.seed(seed_mc)
                x_s = np.random.uniform(a_mc, b_mc, n_iter)
                mc_vals.append((b_mc - a_mc) * np.mean(f_mc(x_s)))

            fig_mc = make_subplots(rows=2, cols=1,
                subplot_titles=[f"Convergence Monte Carlo — {func_mc}",
                                 "Histogramme des estimations (répétitions)"])

            fig_mc.add_trace(go.Scatter(
                x=N_range, y=mc_vals, mode='lines+markers', name='MC estimé',
                line=dict(color='#00ccff', width=2)
            ), row=1, col=1)
            fig_mc.add_hline(y=ref_mc, line_color='#ffcc00', line_dash='dash',
                             annotation_text=f"Exacte={ref_mc:.4f}", row=1, col=1)

            # Distribution des estimations sur n_runs tirages
            estimates = []
            for _ in range(n_runs):
                np.random.seed(np.random.randint(0, 9999))
                xs = np.random.uniform(a_mc, b_mc, N_mc)
                estimates.append((b_mc - a_mc) * np.mean(f_mc(xs)))

            fig_mc.add_trace(go.Histogram(
                x=estimates, nbinsx=15, name='Estimations',
                marker=dict(color='rgba(119,0,255,0.6)',
                           line=dict(color='rgba(0,204,255,0.8)', width=0.5))
            ), row=2, col=1)
            fig_mc.add_vline(x=ref_mc, line_color='#ffcc00', line_dash='dash',
                             annotation_text="Exacte", row=2, col=1)

            fig_mc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                height=550,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            fig_mc.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_mc.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_mc.update_xaxes(type='log', row=1, col=1, title_text="N")
            st.plotly_chart(fig_mc, use_container_width=True)

            # Stats Monte Carlo
            mc_mean = np.mean(estimates)
            mc_std  = np.std(estimates)
            mc_err  = abs(mc_mean - ref_mc)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Moyenne MC", f"{mc_mean:.6f}")
            with c2: st.metric("Écart-type", f"{mc_std:.6f}")
            with c3: st.metric("Erreur moy.", f"{mc_err:.2e}")
            with c4: st.metric("1/√N théorique", f"{(b_mc-a_mc)/np.sqrt(N_mc):.2e}")

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
        st.markdown("### 📊 Comparaison des méthodes")
        df_m = pd.DataFrame([
            {"Méthode": k, "Ordre erreur": v["ordre"],
             "Type": v["formule"], "Coût": v["complexité"]}
            for k, v in METHODES_INFO.items()
        ])
        st.dataframe(df_m, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚗️ Tableau de diagnostic")
        diag_int = {
            "Problème": ["Erreur grande", "Oscillation intégrale", "Singularité",
                         "Monte Carlo lent", "Simpson invalide"],
            "Cause": ["n trop petit", "f oscillante", "Pôle en [a,b]",
                      "N insuffisant", "n impair"],
            "Solution": ["Augmenter n", "Gauss ou subdivision", "Changer bornes",
                         "Augmenter N", "Prendre n pair"]
        }
        st.dataframe(pd.DataFrame(diag_int), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Davis & Rabinowitz — *Methods of Numerical Integration* (Academic Press, 1984)",
            "Press et al. — *Numerical Recipes* (Cambridge, 2007)",
            "Golub & Welsch — *Calculation of Gauss quadrature rules* (Math. Comp., 1969)",
        ]:
            st.markdown(f"- {r}")