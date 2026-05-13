import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import (interp1d, CubicSpline, BarycentricInterpolator,
                                make_interp_spline, RBFInterpolator,
                                RegularGridInterpolator)
from scipy.optimize import curve_fit
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial import chebyshev, legendre
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORMULAIRE SCIENTIFIQUE
# ============================================================
FORMULES = {
    "Polynôme de Lagrange": r"P_n(x) = \sum_{i=0}^{n} y_i \prod_{j\neq i} \frac{x-x_j}{x_i-x_j}",
    "Spline cubique": r"S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3",
    "Erreur interpolation": r"|f(x) - P_n(x)| \leq \frac{M_{n+1}}{(n+1)!}\prod_{i=0}^n|x-x_i|",
    "Phénomène de Runge": r"f(x) = \frac{1}{1+25x^2} \quad \text{(oscille aux bords)}",
    "Nœuds de Chebyshev": r"x_k = \cos\left(\frac{2k+1}{2n+2}\pi\right), \quad k=0,\ldots,n",
    "Interpolation RBF": r"s(x) = \sum_{i=1}^n \lambda_i \phi(\|x - x_i\|)",
    "Régression linéaire": r"\hat{y} = \beta_0 + \beta_1 x, \quad \beta_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}",
    "R² (coefficient)": r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}",
    "RMSE": r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}",
    "MAE": r"\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|",
}

METHODES_INFO = {
    "Linéaire":         {"ordre": 1,  "dérivées": "C⁰", "oscille": False, "complexité": "O(n)"},
    "Cubique (Spline)": {"ordre": 3,  "dérivées": "C²", "oscille": False, "complexité": "O(n)"},
    "Lagrange":         {"ordre": n,  "dérivées": "C∞", "oscille": True,  "complexité": "O(n²)"} if (n := "n-1") else {},
    "Nearest":          {"ordre": 0,  "dérivées": "C⁻¹","oscille": False, "complexité": "O(1)"},
    "Polynôme ordre N": {"ordre": "N","dérivées": "C∞", "oscille": True,  "complexité": "O(n²)"},
    "Chebyshev":        {"ordre": "N","dérivées": "C∞", "oscille": False, "complexité": "O(n²)"},
    "RBF (Gaussien)":   {"ordre": "∞","dérivées": "C∞", "oscille": False, "complexité": "O(n³)"},
    "B-Spline":         {"ordre": "k","dérivées": "Cᵏ⁻¹","oscille": False, "complexité": "O(n)"},
}

EXEMPLES_PREDIFINIS = {
    "Parabole":     ("0,1,2,3,4", "0,1,4,9,16",          "y = x²"),
    "Sinus":        ("0,0.5,1,1.5,2,2.5,3", "0,0.479,0.841,0.997,0.909,0.598,0.141", "y = sin(x)"),
    "Exponentielle":("0,0.5,1,1.5,2,2.5,3", "1,1.65,2.72,4.48,7.39,12.2,20.1",       "y = eˣ"),
    "Runge":        ("-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1",
                     "0.038,0.058,0.099,0.200,0.500,1.000,0.500,0.200,0.099,0.058,0.038",
                     "y = 1/(1+25x²)"),
    "Données bruitées": ("0,1,2,3,4,5,6", "1.1,2.9,4.2,8.8,15.9,25.1,36.4",         "données réelles"),
    "Oscillant":    ("0,1,2,3,4,5,6,7",   "0,1,-1,0,1,-1,0,1",                       "signal périodique"),
    "Chebyshev test":("-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1",
                      "0.038,0.066,0.138,0.390,1.0,0.390,0.138,0.066,0.038",
                      "nœuds de Chebyshev"),
}


# ============================================================
# MOTEUR INTERPOLATION
# ============================================================
class InterpolationEngine:
    """Moteur d'interpolation scientifique avancé."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self._valider(x_data, y_data)
        self.x = np.array(x_data, dtype=float)
        self.y = np.array(y_data, dtype=float)
        self.n = len(x_data)

    def _valider(self, x, y):
        if len(x) != len(y):
            raise ValueError("x et y doivent avoir la même longueur")
        if len(x) < 2:
            raise ValueError("Au moins 2 points requis")
        if len(np.unique(x)) != len(x):
            raise ValueError("Les valeurs de x doivent être distinctes")

    def interpoler(self, x_fin: np.ndarray, methode: str,
                   poly_order: int = 3, rbf_kernel: str = "gaussian") -> np.ndarray:
        try:
            x_sorted = self.x
            y_sorted = self.y
            idx = np.argsort(x_sorted)
            x_s = x_sorted[idx]
            y_s = y_sorted[idx]

            if methode == "Linéaire":
                f = interp1d(x_s, y_s, kind='linear', fill_value='extrapolate')
                return f(x_fin)

            elif methode == "Cubique (Spline)":
                cs = CubicSpline(x_s, y_s)
                return cs(x_fin)

            elif methode == "Lagrange":
                bi = BarycentricInterpolator(x_s, y_s)
                return bi(x_fin)

            elif methode == "Nearest":
                f = interp1d(x_s, y_s, kind='nearest', fill_value='extrapolate')
                return f(x_fin)

            elif methode == "Polynôme ordre N":
                p = Polynomial.fit(x_s, y_s, deg=poly_order)
                return p(x_fin)

            elif methode == "Chebyshev":
                c = chebyshev.chebfit(x_s, y_s, deg=min(poly_order, self.n-1))
                return chebyshev.chebval(x_fin, c)

            elif methode == "RBF (Gaussien)":
                rbf = RBFInterpolator(x_s.reshape(-1,1), y_s,
                                      kernel=rbf_kernel, epsilon=1.0)
                return rbf(x_fin.reshape(-1,1))

            elif methode == "B-Spline":
                k = min(poly_order, self.n - 1)
                bsp = make_interp_spline(x_s, y_s, k=k)
                return bsp(x_fin)

            else:
                return np.zeros_like(x_fin)

        except Exception as e:
            st.error(f"Erreur interpolation ({methode}): {e}")
            return np.zeros_like(x_fin)

    def metriques(self, x_fin: np.ndarray, y_fin: np.ndarray,
                  methode: str) -> dict:
        """Calcule les métriques d'erreur sur les points de données."""
        y_pred = np.interp(self.x, x_fin, y_fin)
        residus = self.y - y_pred
        ss_res = np.sum(residus**2)
        ss_tot = np.sum((self.y - self.y.mean())**2)
        rmse = np.sqrt(np.mean(residus**2))
        mae = np.mean(np.abs(residus))
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        max_err = np.max(np.abs(residus))
        return {
            "R²": r2, "RMSE": rmse, "MAE": mae,
            "Erreur max": max_err,
            "Erreur moy.": np.mean(np.abs(residus)),
            "Biais": np.mean(residus),
        }

    def nœuds_chebyshev(self, n: int, a: float, b: float) -> np.ndarray:
        """Génère n nœuds de Chebyshev sur [a, b]."""
        k = np.arange(n)
        x_cheb = np.cos((2*k + 1) * np.pi / (2*n))
        return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

    def regression_polynomiale(self, degre: int) -> dict:
        """Régression polynomiale avec intervalles de confiance."""
        try:
            p, cov = np.polyfit(self.x, self.y, degre, cov=True)
            poly = np.poly1d(p)
            y_pred = poly(self.x)
            residus = self.y - y_pred
            ss_res = np.sum(residus**2)
            ss_tot = np.sum((self.y - self.y.mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            n_dof = self.n - degre - 1
            sigma2 = ss_res / max(n_dof, 1)
            return {
                "coefficients": p,
                "r2": r2,
                "rmse": np.sqrt(sigma2),
                "poly": poly,
                "cov": cov
            }
        except:
            return {}

    def benchmark_methodes(self, x_fin: np.ndarray,
                           methodes: list) -> pd.DataFrame:
        """Compare toutes les méthodes sur les mêmes données."""
        resultats = []
        for m in methodes:
            try:
                y_fin = self.interpoler(x_fin, m)
                met = self.metriques(x_fin, y_fin, m)
                resultats.append({
                    "Méthode": m,
                    "R²": f"{met['R²']:.6f}",
                    "RMSE": f"{met['RMSE']:.2e}",
                    "MAE": f"{met['MAE']:.2e}",
                    "Erreur max": f"{met['Erreur max']:.2e}",
                })
            except:
                resultats.append({"Méthode": m, "R²": "❌", "RMSE": "❌",
                                   "MAE": "❌", "Erreur max": "❌"})
        return pd.DataFrame(resultats)

    def detecter_anomalies(self) -> list:
        """Détecte les problèmes potentiels dans les données."""
        issues = []
        if self.n < 3:
            issues.append("⚠️ Peu de points — interpolation peu fiable")
        if np.any(np.diff(self.x) <= 0):
            issues.append("⚠️ Points x non triés ou dupliqués")
        z_scores = np.abs(stats.zscore(self.y))
        outliers = np.where(z_scores > 2.5)[0]
        if len(outliers) > 0:
            issues.append(f"⚠️ Outliers détectés aux indices : {outliers.tolist()}")
        x_range = self.x.max() - self.x.min()
        gaps = np.diff(np.sort(self.x))
        if gaps.max() > 0.5 * x_range:
            issues.append("⚠️ Grand écart entre points — risque d'oscillation")
        if self.n > 10:
            issues.append("ℹ️ N > 10 — préférer Spline ou RBF à Lagrange")
        return issues if issues else ["✅ Données valides"]


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def interpolation_page():
    st.markdown("## 📈 Interpolation & Régression Scientifique")
    st.markdown("*Interpolation avancée, benchmark, analyse d'erreur, régression*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔢 Interpolation",
        "📊 Benchmark méthodes",
        "📉 Régression",
        "🔬 Analyse avancée",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : INTERPOLATION PRINCIPALE
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📊 Saisie des données")

            # Exemples
            ex_sel = st.selectbox("Exemple prédéfini", ["— Manuel —"] + list(EXEMPLES_PREDIFINIS.keys()))
            if ex_sel != "— Manuel —":
                x_def, y_def, desc = EXEMPLES_PREDIFINIS[ex_sel]
                st.caption(f"*{desc}*")
            else:
                x_def, y_def = "0,1,2,3,4", "0,1,4,9,16"

            x_str = st.text_input("**x** (séparés par virgule)", x_def)
            y_str = st.text_input("**y** (séparés par virgule)", y_def)

            try:
                x_data = np.array([float(v.strip()) for v in x_str.split(',')])
                y_data = np.array([float(v.strip()) for v in y_str.split(',')])
                engine = InterpolationEngine(x_data, y_data)
                st.success(f"✅ {len(x_data)} points chargés")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                st.stop()

            # Détection anomalies
            anomalies = engine.detecter_anomalies()
            for a in anomalies:
                if "✅" in a:
                    st.success(a)
                elif "⚠️" in a:
                    st.warning(a)
                else:
                    st.info(a)

            st.markdown("### ⚙️ Configuration")
            methode = st.selectbox("Méthode", [
                "Linéaire", "Cubique (Spline)", "Lagrange",
                "Nearest", "Polynôme ordre N",
                "Chebyshev", "RBF (Gaussien)", "B-Spline"
            ])

            n_points_fin = st.slider("Points interpolés", 50, 2000, 500)

            poly_order = 3
            rbf_kernel = "gaussian"
            if methode in ["Polynôme ordre N", "Chebyshev", "B-Spline"]:
                poly_order = st.slider("Ordre", 1, min(15, engine.n-1), 3)
            if methode == "RBF (Gaussien)":
                rbf_kernel = st.selectbox("Noyau RBF",
                    ["gaussian", "linear", "thin_plate_spline",
                     "cubic", "quintic", "multiquadric"])

            extrapoler = st.checkbox("Extrapolation (±20%)", False)

            # Nœuds de Chebyshev
            show_cheb = st.checkbox("Afficher nœuds de Chebyshev", False)

        with col2:
            x_min, x_max = x_data.min(), x_data.max()
            marge = 0.2 * (x_max - x_min) if extrapoler else 0.0
            x_fin = np.linspace(x_min - marge, x_max + marge, n_points_fin)

            y_fin = engine.interpoler(x_fin, methode, poly_order, rbf_kernel)

            # Graphique principal
            fig = go.Figure()

            # Courbe interpolée
            fig.add_trace(go.Scatter(
                x=x_fin, y=y_fin, mode='lines', name=methode,
                line=dict(color='#00ccff', width=3)
            ))

            # Points originaux
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                name='Points donnés',
                marker=dict(color='#ff00cc', size=12, symbol='circle',
                           line=dict(width=2, color='#ffffff'))
            ))

            # Nœuds de Chebyshev
            if show_cheb:
                x_cheb = engine.nœuds_chebyshev(engine.n, x_min, x_max)
                y_cheb = np.interp(x_cheb, x_fin, y_fin)
                fig.add_trace(go.Scatter(
                    x=x_cheb, y=y_cheb, mode='markers',
                    name='Nœuds Chebyshev',
                    marker=dict(color='#ffcc00', size=10, symbol='diamond')
                ))

            if extrapoler:
                fig.add_vrect(x0=x_min-marge, x1=x_min,
                              fillcolor='rgba(255,100,0,0.1)',
                              line_width=0, annotation_text="extrapol.")
                fig.add_vrect(x0=x_max, x1=x_max+marge,
                              fillcolor='rgba(255,100,0,0.1)',
                              line_width=0, annotation_text="extrapol.")

            fig.update_layout(
                title=f"Interpolation {methode} — {engine.n} points",
                xaxis_title="x", yaxis_title="y",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=450,
            )
            st.plotly_chart(fig, width='stretch')

            # Métriques
            met = engine.metriques(x_fin, y_fin, methode)
            cols_m = st.columns(3)
            with cols_m[0]: st.metric("R²", f"{met['R²']:.6f}")
            with cols_m[1]: st.metric("RMSE", f"{met['RMSE']:.2e}")
            with cols_m[2]: st.metric("Erreur max", f"{met['Erreur max']:.2e}")

            # Erreur aux points
            st.markdown("#### 📉 Erreur résiduelle aux points")
            y_pred_pts = np.interp(x_data, x_fin, y_fin)
            erreurs = np.abs(y_data - y_pred_pts)
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=list(range(len(x_data))), y=erreurs,
                marker=dict(
                    color=erreurs,
                    colorscale=[[0,'#7700ff'],[0.5,'#00ccff'],[1,'#ff0000']],
                    showscale=True
                ),
                name='|résidu|'
            ))
            fig_err.update_layout(
                xaxis_title="Index point", yaxis_title="|y - ŷ|",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                height=280,
            )
            st.plotly_chart(fig_err, width='stretch')

            # Export
            df_exp = pd.DataFrame({"x": x_fin, "y_interp": y_fin})
            st.download_button("💾 Export CSV", df_exp.to_csv(index=False).encode(),
                               "interpolation.csv", "text/csv")

    # ============================================================
    # TAB 2 : BENCHMARK
    # ============================================================
    with tab2:
        st.markdown("### 📊 Benchmark de toutes les méthodes")

        try:
            x_data_b = np.array([float(v.strip()) for v in x_str.split(',')])
            y_data_b = np.array([float(v.strip()) for v in y_str.split(',')])
            engine_b = InterpolationEngine(x_data_b, y_data_b)
            x_fin_b = np.linspace(x_data_b.min(), x_data_b.max(), 500)

            methodes_bench = ["Linéaire", "Cubique (Spline)", "Lagrange",
                              "Nearest", "Polynôme ordre N", "B-Spline"]

            if engine_b.n >= 3:
                methodes_bench.append("RBF (Gaussien)")

            with st.spinner("Calcul benchmark..."):
                df_bench = engine_b.benchmark_methodes(x_fin_b, methodes_bench)

            st.dataframe(df_bench, width='stretch')

            # Barplot RMSE
            fig_bench = go.Figure()
            for _, row in df_bench.iterrows():
                try:
                    rmse_val = float(row["RMSE"])
                    fig_bench.add_trace(go.Bar(
                        x=[row["Méthode"]], y=[rmse_val + 1e-16],
                        name=row["Méthode"],
                        marker_color='#7700ff'
                    ))
                except:
                    pass

            fig_bench.update_layout(
                title="RMSE par méthode",
                yaxis_type='log', yaxis_title="RMSE (log)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                showlegend=False, height=380,
            )
            st.plotly_chart(fig_bench, width='stretch')

            # Graphique comparatif toutes méthodes
            st.markdown("#### Comparaison visuelle")
            fig_comp = go.Figure()
            colors_c = ['#00ccff','#7700ff','#ff00cc','#00ff88',
                        '#ffcc00','#ff4400','#88ccff','#cc88ff']

            for i, m in enumerate(methodes_bench[:6]):
                try:
                    y_m = engine_b.interpoler(x_fin_b, m)
                    fig_comp.add_trace(go.Scatter(
                        x=x_fin_b, y=y_m, mode='lines', name=m,
                        line=dict(color=colors_c[i % len(colors_c)], width=2)
                    ))
                except:
                    pass

            fig_comp.add_trace(go.Scatter(
                x=x_data_b, y=y_data_b, mode='markers', name='Données',
                marker=dict(color='#ffffff', size=12, symbol='circle')
            ))
            fig_comp.update_layout(
                title="Comparaison visuelle des méthodes",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=420,
            )
            st.plotly_chart(fig_comp, width='stretch')

        except Exception as e:
            st.error(f"Erreur benchmark: {e}")

    # ============================================================
    # TAB 3 : RÉGRESSION
    # ============================================================
    with tab3:
        st.markdown("### 📉 Régression polynomiale & analyse")

        try:
            x_r = np.array([float(v.strip()) for v in x_str.split(',')])
            y_r = np.array([float(v.strip()) for v in y_str.split(',')])
            engine_r = InterpolationEngine(x_r, y_r)

            col1, col2 = st.columns([1, 2])
            with col1:
                deg_reg = st.slider("Degré de régression", 1, min(8, engine_r.n-1), 2)
                res_reg = engine_r.regression_polynomiale(deg_reg)

                if res_reg:
                    st.metric("R²", f"{res_reg['r2']:.6f}")
                    st.metric("RMSE", f"{res_reg['rmse']:.4e}")
                    st.markdown("**Coefficients :**")
                    for i, c in enumerate(res_reg["coefficients"]):
                        st.text(f"  a{deg_reg-i} = {c:.6e}")

            with col2:
                if res_reg:
                    x_fin_r = np.linspace(x_r.min(), x_r.max(), 500)
                    y_reg = res_reg["poly"](x_fin_r)
                    y_pred_r = res_reg["poly"](x_r)
                    residus_r = y_r - y_pred_r

                    fig_reg = make_subplots(rows=2, cols=1,
                        subplot_titles=["Régression polynomiale", "Résidus"])

                    fig_reg.add_trace(go.Scatter(
                        x=x_r, y=y_r, mode='markers', name='Données',
                        marker=dict(color='#ff00cc', size=10)
                    ), row=1, col=1)
                    fig_reg.add_trace(go.Scatter(
                        x=x_fin_r, y=y_reg, mode='lines', name=f'Deg {deg_reg}',
                        line=dict(color='#00ccff', width=3)
                    ), row=1, col=1)

                    fig_reg.add_trace(go.Bar(
                        x=list(range(len(x_r))), y=residus_r,
                        name='Résidus', marker_color='#7700ff'
                    ), row=2, col=1)
                    fig_reg.add_hline(y=0, line_color='white', line_dash='dash',
                                      row=2, col=1)

                    fig_reg.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(5,0,20,0.8)',
                        font=dict(color='#c0d0ff'),
                        height=520,
                        legend=dict(bgcolor='rgba(0,0,0,0.5)')
                    )
                    fig_reg.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                    fig_reg.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                    st.plotly_chart(fig_reg, width='stretch')

            # Comparaison degrés
            st.markdown("#### 📈 R² vs degré")
            degs = list(range(1, min(engine_r.n, 9)))
            r2_vals = []
            for d in degs:
                res_d = engine_r.regression_polynomiale(d)
                r2_vals.append(res_d.get("r2", 0) if res_d else 0)

            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(
                x=degs, y=r2_vals, mode='lines+markers',
                line=dict(color='#00ccff', width=2.5),
                marker=dict(size=8, color='#7700ff')
            ))
            fig_r2.add_hline(y=1.0, line_color='#ffcc00', line_dash='dash',
                             annotation_text="R²=1")
            fig_r2.update_layout(
                xaxis_title="Degré", yaxis_title="R²",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                          range=[0, 1.05]),
                height=300,
            )
            st.plotly_chart(fig_r2, width='stretch')

        except Exception as e:
            st.error(f"Erreur régression: {e}")

    # ============================================================
    # TAB 4 : ANALYSE AVANCÉE
    # ============================================================
    with tab4:
        st.markdown("### 🔬 Analyse avancée — Phénomène de Runge")

        col1, col2 = st.columns([1, 2])
        with col1:
            n_runge = st.slider("Nombre de points uniformes", 3, 20, 10)
            deg_runge = st.slider("Degré Lagrange", 2, 19, 9)

        with col2:
            x_true = np.linspace(-1, 1, 500)
            y_true = 1 / (1 + 25 * x_true**2)

            # Points uniformes
            x_unif = np.linspace(-1, 1, n_runge)
            y_unif = 1 / (1 + 25 * x_unif**2)

            # Points Chebyshev
            engine_r2 = InterpolationEngine(x_unif, y_unif)
            k = np.arange(n_runge)
            x_cheb_r = np.cos((2*k + 1) * np.pi / (2*n_runge))
            x_cheb_r = np.sort(x_cheb_r)
            y_cheb_r = 1 / (1 + 25 * x_cheb_r**2)

            engine_cheb = InterpolationEngine(x_cheb_r, y_cheb_r)
            y_lagr_unif = engine_r2.interpoler(x_true, "Lagrange")
            y_lagr_cheb = engine_cheb.interpoler(x_true, "Lagrange")
            y_spline = engine_r2.interpoler(x_true, "Cubique (Spline)")

            fig_runge = go.Figure()
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_true, mode='lines', name='f(x) vraie',
                line=dict(color='#ffffff', width=2)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_lagr_unif, mode='lines', name='Lagrange (uniforme)',
                line=dict(color='#ff4444', width=2.5)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_lagr_cheb, mode='lines', name='Lagrange (Chebyshev)',
                line=dict(color='#00ff88', width=2.5)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_true, y=y_spline, mode='lines', name='Spline cubique',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_unif, y=y_unif, mode='markers', name='Pts uniformes',
                marker=dict(color='#ff4444', size=8, symbol='circle')
            ))
            fig_runge.add_trace(go.Scatter(
                x=x_cheb_r, y=y_cheb_r, mode='markers', name='Pts Chebyshev',
                marker=dict(color='#00ff88', size=8, symbol='diamond')
            ))
            fig_runge.update_layout(
                title=f"Phénomène de Runge — f(x)=1/(1+25x²), n={n_runge}",
                xaxis_title="x", yaxis_title="y",
                yaxis=dict(range=[-2, 2]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                yaxis2=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=480,
            )
            st.plotly_chart(fig_runge, width='stretch')

            st.info("🔑 Les nœuds de Chebyshev éliminent le phénomène de Runge en concentrant les points aux bords de l'intervalle.")

    # ============================================================
    # TAB 5 : THÉORIE
    # ============================================================
    with tab5:
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
        df_comp = pd.DataFrame([
            {"Méthode": "Linéaire", "Continuité": "C⁰", "Oscillation": "Non",
             "Complexité": "O(n)", "Usage": "Données irrégulières"},
            {"Méthode": "Spline cubique", "Continuité": "C²", "Oscillation": "Rare",
             "Complexité": "O(n)", "Usage": "Données lisses ✅"},
            {"Méthode": "Lagrange", "Continuité": "C∞", "Oscillation": "Oui (Runge)",
             "Complexité": "O(n²)", "Usage": "Peu de points"},
            {"Méthode": "Chebyshev", "Continuité": "C∞", "Oscillation": "Réduite",
             "Complexité": "O(n²)", "Usage": "Nœuds optimaux ✅"},
            {"Méthode": "RBF", "Continuité": "C∞", "Oscillation": "Rare",
             "Complexité": "O(n³)", "Usage": "Multi-dimensionnel"},
            {"Méthode": "B-Spline", "Continuité": "Cᵏ⁻¹", "Oscillation": "Non",
             "Complexité": "O(n)", "Usage": "CAO, animation"},
        ])
        st.dataframe(df_comp, width='stretch')

        st.markdown("---")
        st.markdown("### 📚 Références")
        refs = [
            "Burden & Faires — *Numerical Analysis* (Cengage, 2010)",
            "Press et al. — *Numerical Recipes in Python* (Cambridge, 2007)",
            "Stoer & Bulirsch — *Introduction to Numerical Analysis* (Springer, 2002)",
            "De Boor — *A Practical Guide to Splines* (Springer, 2001)",
        ]
        for r in refs:
            st.markdown(f"- {r}")