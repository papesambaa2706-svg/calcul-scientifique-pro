import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, signal, integrate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
CONSTANTES_ENERGIE = {
    "Pouvoir calorifique gaz naturel": (10.55,  "kWh/m³"),
    "Pouvoir calorifique fioul":       (10.00,  "kWh/L"),
    "Pouvoir calorifique charbon":     (8.14,   "kWh/kg"),
    "Facteur émission électricité FR": (0.052,  "kgCO₂/kWh"),
    "Facteur émission gaz naturel":    (0.234,  "kgCO₂/kWh"),
    "1 tep (tonne équivalent pétrole)":(11628,  "kWh"),
    "1 kWh":                           (3.6e6,  "J"),
}

FORMULES_ENERGIE = {
    "Puissance électrique":    r"P = U \cdot I = \frac{U^2}{R} = R \cdot I^2 \quad \text{(W)}",
    "Énergie thermique":       r"Q = m \cdot c_p \cdot \Delta T \quad \text{(J)}",
    "Rendement":               r"\eta = \frac{E_{utile}}{E_{totale}} \times 100\%",
    "Loi de Joule":            r"P_{Joule} = R \cdot I^2 \quad \text{(W)}",
    "Énergie cinétique":       r"E_c = \frac{1}{2}mv^2 \quad \text{(J)}",
    "Énergie potentielle":     r"E_p = mgh \quad \text{(J)}",
    "Puissance mécanique":     r"P = F \cdot v = \tau \cdot \omega \quad \text{(W)}",
    "Bilan énergétique":       r"E_{in} = E_{utile} + E_{pertes}",
    "Facteur de puissance":    r"\cos\phi = \frac{P}{S} = \frac{P}{\sqrt{P^2+Q^2}}",
    "Coefficient performance": r"\text{COP} = \frac{E_{utile}}{W_{electrique}}",
    "Degré-jour":              r"DJ = \sum_{j} \max(T_{ref} - T_j, 0)",
    "Intensité énergétique":   r"IE = \frac{\text{Consommation (kWh)}}{\text{Surface (m}^2)}",
}

INDICATEURS_BATIMENT = {
    "Classe A": (0,    50),
    "Classe B": (51,   90),
    "Classe C": (91,  150),
    "Classe D": (151, 230),
    "Classe E": (231, 330),
    "Classe F": (331, 450),
    "Classe G": (451, 9999),
}


# ============================================================
# MOTEUR ÉNERGIE
# ============================================================
class EnergyEngine:
    """Moteur d'analyse énergétique scientifique."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = list(df.select_dtypes(include=np.number).columns)

    def profil_energetique(self, col: str) -> dict:
        """Analyse statistique complète d'une série énergétique."""
        s = self.df[col].dropna()
        return {
            "Moyenne": s.mean(),
            "Médiane": s.median(),
            "Std": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "Somme": s.sum(),
            "CV (%)": s.std() / s.mean() * 100 if s.mean() != 0 else 0,
            "Skewness": float(stats.skew(s)),
            "Kurtosis": float(stats.kurtosis(s)),
            "P10": s.quantile(0.10),
            "P90": s.quantile(0.90),
            "Outliers IQR": int(((s < s.quantile(0.25)-1.5*(s.quantile(0.75)-s.quantile(0.25))) |
                                 (s > s.quantile(0.75)+1.5*(s.quantile(0.75)-s.quantile(0.25)))).sum()),
        }

    def decomposition_tendance(self, col: str) -> dict:
        """Décomposition Tendance + Saisonnalité (Fourier) + Résidu."""
        s = self.df[col].dropna().values
        N = len(s)
        t = np.arange(N)

        # Tendance polynomiale
        p = np.polyfit(t, s, 2)
        tendance = np.polyval(p, t)
        residu = s - tendance

        # Composante cyclique par FFT
        fft_vals = np.fft.rfft(residu)
        freqs = np.fft.rfftfreq(N)
        magnitude = np.abs(fft_vals)

        # Fréquence dominante
        idx_dom = np.argmax(magnitude[1:]) + 1
        f_dom = freqs[idx_dom]

        return {
            "t": t, "original": s, "tendance": tendance,
            "residu": residu, "freqs": freqs, "magnitude": magnitude,
            "f_dom": f_dom, "periode_dom": 1/f_dom if f_dom > 0 else np.inf,
        }

    def bilan_energetique(self, E_in: float, eta: float) -> dict:
        """Calcul du bilan énergétique simple."""
        E_utile = E_in * eta / 100
        E_pertes = E_in - E_utile
        return {
            "E_in (kWh)": E_in,
            "E_utile (kWh)": E_utile,
            "E_pertes (kWh)": E_pertes,
            "Rendement (%)": eta,
            "CO₂ évité (kgCO₂)": E_utile * 0.052,
        }

    def intensite_energetique(self, col: str, surface: float) -> pd.Series:
        return self.df[col] / surface

    def modele_predictif(self, target: str, features: list,
                          modele: str = "Ridge", test_size: float = 0.2) -> dict:
        """Modèle prédictif de consommation énergétique."""
        X = self.df[features].dropna()
        y = self.df.loc[X.index, target]
        if len(X) < 10:
            return {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "Régression Linéaire": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        model = models.get(modele, Ridge())
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        residus = y_test.values - y_pred

        importance = None
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        return {
            "model": model, "scaler": scaler,
            "y_test": y_test, "y_pred": y_pred,
            "r2": r2, "rmse": rmse, "residus": residus,
            "importance": importance, "features": features
        }

    def classe_dpe(self, consommation_kwh_m2: float) -> str:
        """Classe DPE selon consommation en kWh/m²/an."""
        for classe, (mini, maxi) in INDICATEURS_BATIMENT.items():
            if mini <= consommation_kwh_m2 <= maxi:
                return classe
        return "G"

    def detecter_pointes(self, col: str, seuil_sigma: float = 2.0) -> np.ndarray:
        """Détecte les pics de consommation (pointes)."""
        s = self.df[col].dropna()
        z = np.abs(stats.zscore(s))
        return z > seuil_sigma


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def energy_page():
    st.markdown("## ⚡ Analyse Énergétique Avancée")
    st.markdown("*Profil énergétique, décomposition, prédiction, bilan, DPE*")
    st.markdown("---")

    uploaded = st.file_uploader("📁 Importer CSV ou Excel", type=["csv", "xlsx"])

    if uploaded is None:
        # Mode démo avec données synthétiques
        st.info("👆 Importez vos données ou utilisez le mode démo ci-dessous.")
        if st.button("🎲 Générer données démo"):
            np.random.seed(42)
            n = 365
            t_demo = np.arange(n)
            consommation = (
                200 + 100 * np.cos(2*np.pi*t_demo/365) +
                np.random.normal(0, 20, n) +
                0.1 * t_demo
            )
            temperature = 15 - 10 * np.cos(2*np.pi*t_demo/365) + np.random.normal(0, 3, n)
            df_demo = pd.DataFrame({
                "Consommation_kWh": np.clip(consommation, 50, 500),
                "Temperature_C": temperature,
                "Jour": t_demo,
                "Irradiation_Wh_m2": np.clip(
                    300 + 200*np.sin(2*np.pi*t_demo/365) + np.random.normal(0, 50, n), 0, 800)
            })
            st.session_state["df_energy"] = df_demo
            st.success("✅ Données démo générées (365 jours)")
            st.dataframe(df_demo.head(10), use_container_width=True)
        return

    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
             else pd.read_excel(uploaded)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        st.success(f"✅ {df.shape[0]} lignes × {df.shape[1]} colonnes")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return

    engine = EnergyEngine(df)
    numeric_cols = engine.numeric_cols

    if not numeric_cols:
        st.error("❌ Aucune colonne numérique.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Profil",
        "📊 Visualisation",
        "📈 Décomposition",
        "🤖 Prédiction",
        "⚡ Bilan & DPE",
        "🔍 Détection pointes",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : PROFIL
    # ============================================================
    with tab1:
        st.markdown("### 📋 Profil énergétique complet")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Lignes", df.shape[0])
        with c2: st.metric("Colonnes", df.shape[1])
        with c3: st.metric("Valeurs nulles", int(df.isnull().sum().sum()))
        with c4: st.metric("Variables num.", len(numeric_cols))

        col_prof = st.selectbox("Variable à analyser", numeric_cols)
        profil = engine.profil_energetique(col_prof)

        cols_p = st.columns(4)
        items = list(profil.items())
        for i, (k, v) in enumerate(items):
            with cols_p[i % 4]:
                st.metric(k, f"{v:.3f}" if isinstance(v, float) else str(v))

        # Distribution
        s_p = df[col_prof].dropna()
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(
            x=s_p, nbinsx=40, name='Distribution',
            marker=dict(color='rgba(119,0,255,0.6)',
                       line=dict(color='rgba(0,204,255,0.8)', width=0.5))
        ))
        kde = stats.gaussian_kde(s_p)
        x_kde = np.linspace(s_p.min(), s_p.max(), 300)
        scale = len(s_p) * (s_p.max()-s_p.min()) / 40
        fig_d.add_trace(go.Scatter(
            x=x_kde, y=kde(x_kde)*scale, mode='lines', name='KDE',
            line=dict(color='#00ccff', width=3)
        ))
        fig_d.add_vline(x=s_p.mean(), line_color='#ffcc00', line_dash='dash',
                        annotation_text=f"Moy={s_p.mean():.1f}")
        fig_d.add_vline(x=s_p.median(), line_color='#00ff88', line_dash='dot',
                        annotation_text=f"Méd={s_p.median():.1f}")
        fig_d.update_layout(
            title=f"Distribution — {col_prof}",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
            font=dict(color='#c0d0ff'),
            xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
            yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # ============================================================
    # TAB 2 : VISUALISATION
    # ============================================================
    with tab2:
        st.markdown("### 📊 Visualisation énergétique")

        col1, col2 = st.columns([1, 2])
        with col1:
            graph_type = st.selectbox("Type", [
                "Série temporelle", "Scatter + régression",
                "Boxplot comparatif", "Corrélation matricielle",
                "Violin plot"
            ])
            sel_col = st.selectbox("Variable principale", numeric_cols, key="viz_col")

        with col2:
            if graph_type == "Série temporelle":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(df))), y=df[sel_col],
                    mode='lines', name=sel_col,
                    line=dict(color='#00ccff', width=2)
                ))
                # Moyenne mobile
                ma = df[sel_col].rolling(7, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=list(range(len(df))), y=ma, mode='lines',
                    name='Moyenne mobile (7)',
                    line=dict(color='#ffcc00', width=2.5, dash='dash')
                ))

            elif graph_type == "Scatter + régression":
                x_col = st.selectbox("X", numeric_cols, key="sc_x")
                y_col = st.selectbox("Y", numeric_cols, key="sc_y",
                                      index=min(1, len(numeric_cols)-1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[y_col], mode='markers',
                    marker=dict(color='#7700ff', size=6, opacity=0.7)
                ))
                mask = df[[x_col, y_col]].dropna()
                if len(mask) > 2:
                    slope, intercept, r, *_ = stats.linregress(mask[x_col], mask[y_col])
                    x_r = np.linspace(mask[x_col].min(), mask[x_col].max(), 200)
                    fig.add_trace(go.Scatter(
                        x=x_r, y=slope*x_r+intercept, mode='lines',
                        name=f'y={slope:.3f}x+{intercept:.3f} (R²={r**2:.3f})',
                        line=dict(color='#ffcc00', width=2.5)
                    ))

            elif graph_type == "Boxplot comparatif":
                fig = go.Figure()
                for col in numeric_cols[:5]:
                    s_norm = (df[col] - df[col].mean()) / df[col].std()
                    fig.add_trace(go.Box(y=s_norm, name=col, boxmean='sd'))

            elif graph_type == "Corrélation matricielle":
                corr = df[numeric_cols].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale=[[0,'#020817'],[0.5,'#7700ff'],[1,'#00ccff']],
                    zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
                    colorbar=dict(tickfont=dict(color='#c0d0ff'))
                ))

            else:  # Violin
                fig = go.Figure()
                for col in numeric_cols[:4]:
                    fig.add_trace(go.Violin(y=df[col].dropna(), name=col,
                                           box_visible=True, meanline_visible=True))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 3 : DÉCOMPOSITION
    # ============================================================
    with tab3:
        st.markdown("### 📈 Décomposition de série temporelle")
        col_dec = st.selectbox("Variable", numeric_cols, key="dec_col")

        decomp = engine.decomposition_tendance(col_dec)
        fig_dec = make_subplots(rows=3, cols=1,
            subplot_titles=["Signal original + Tendance",
                            "Résidu (détrend)", "Spectre FFT"])

        fig_dec.add_trace(go.Scatter(
            x=decomp["t"], y=decomp["original"], mode='lines', name='Original',
            line=dict(color='rgba(0,204,255,0.5)', width=1.5)
        ), row=1, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp["t"], y=decomp["tendance"], mode='lines', name='Tendance',
            line=dict(color='#ffcc00', width=2.5)
        ), row=1, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp["t"], y=decomp["residu"], mode='lines', name='Résidu',
            line=dict(color='#7700ff', width=1.5)
        ), row=2, col=1)
        fig_dec.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=2, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp["freqs"][1:len(decomp["freqs"])//2],
            y=decomp["magnitude"][1:len(decomp["freqs"])//2],
            mode='lines', fill='tozeroy', fillcolor='rgba(119,0,255,0.2)',
            line=dict(color='#7700ff', width=2), name='|FFT|'
        ), row=3, col=1)

        f_dom = decomp.get("f_dom", 0)
        periode = decomp.get("periode_dom", 0)
        if f_dom > 0:
            fig_dec.add_vline(x=f_dom, line_color='#ffcc00', line_dash='dash',
                              annotation_text=f"f={f_dom:.3f}", row=3, col=1)

        fig_dec.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
            font=dict(color='#c0d0ff'), height=620,
            legend=dict(bgcolor='rgba(0,0,0,0.5)')
        )
        fig_dec.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
        fig_dec.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
        st.plotly_chart(fig_dec, use_container_width=True)

        if periode < np.inf:
            st.metric("Période dominante (points)", f"{periode:.1f}")

    # ============================================================
    # TAB 4 : PRÉDICTION
    # ============================================================
    with tab4:
        st.markdown("### 🤖 Modèle prédictif de consommation")

        col1, col2 = st.columns([1, 2])
        with col1:
            target_e = st.selectbox("Variable cible", numeric_cols)
            features_e = st.multiselect(
                "Variables explicatives",
                [c for c in numeric_cols if c != target_e],
                default=[c for c in numeric_cols if c != target_e][:2]
            )
            modele_e = st.selectbox("Modèle", ["Ridge", "Régression Linéaire", "Gradient Boosting"])
            test_sz = st.slider("Test (%)", 10, 40, 20) / 100

        with col2:
            if st.button("🚀 Entraîner", use_container_width=True) and features_e:
                res_e = engine.modele_predictif(target_e, features_e, modele_e, test_sz)
                if res_e:
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("R²", f"{res_e['r2']:.4f}")
                    with c2: st.metric("RMSE", f"{res_e['rmse']:.4f}")
                    with c3: st.metric("Qualité", "✅ Bon" if res_e['r2'] > 0.8
                                       else "⚠️ Moyen" if res_e['r2'] > 0.5 else "❌ Faible")

                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=res_e['y_test'], y=res_e['y_pred'], mode='markers',
                        marker=dict(color='#00ccff', size=7, opacity=0.7), name='Prédictions'
                    ))
                    lim = [min(res_e['y_test'].min(), res_e['y_pred'].min()),
                           max(res_e['y_test'].max(), res_e['y_pred'].max())]
                    fig_pr.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
                        name='Parfait', line=dict(color='#ffcc00', dash='dash')))
                    fig_pr.update_layout(
                        title="Réel vs Prédit",
                        xaxis_title="Consommation réelle",
                        yaxis_title="Consommation prédite",
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=400,
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)

                    if res_e['importance'] is not None:
                        imp_df = pd.DataFrame({
                            "Variable": features_e,
                            "Importance": res_e['importance']
                        }).sort_values("Importance", ascending=True)
                        fig_imp = go.Figure(go.Bar(
                            x=imp_df["Importance"], y=imp_df["Variable"],
                            orientation='h',
                            marker=dict(color=imp_df["Importance"],
                                       colorscale=[[0,'#7700ff'],[1,'#00ccff']])
                        ))
                        fig_imp.update_layout(
                            title="Importance des variables",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                            font=dict(color='#c0d0ff'),
                            xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                            yaxis=dict(color='#c0d0ff'), height=280,
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

    # ============================================================
    # TAB 5 : BILAN & DPE
    # ============================================================
    with tab5:
        st.markdown("### ⚡ Bilan énergétique & DPE")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔋 Bilan simplifié")
            E_in = st.slider("Énergie entrante (kWh)", 100.0, 100000.0, 10000.0, 100.0)
            eta = st.slider("Rendement η (%)", 1.0, 100.0, 80.0, 0.5)
            bilan = engine.bilan_energetique(E_in, eta)

            for k, v in bilan.items():
                st.metric(k, f"{v:.2f}")

            # Diagramme de Sankey simplifié
            fig_san = go.Figure(go.Funnel(
                y=["Énergie entrante", "Énergie utile", "Pertes"],
                x=[E_in, bilan["E_utile (kWh)"], bilan["E_pertes (kWh)"]],
                marker=dict(color=['#00ccff', '#00ff88', '#ff4444'])
            ))
            fig_san.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0d0ff'), height=300,
            )
            st.plotly_chart(fig_san, use_container_width=True)

        with col2:
            st.markdown("#### 🏠 Diagnostic de Performance Énergétique (DPE)")
            surface = st.slider("Surface habitable (m²)", 10, 1000, 100)
            col_conso = st.selectbox("Variable de consommation", numeric_cols, key="dpe_col")
            conso_totale = df[col_conso].sum()
            conso_m2 = conso_totale / surface

            classe = engine.classe_dpe(conso_m2)
            couleurs_dpe = {
                "A": "#00cc00", "B": "#66cc00", "C": "#cccc00",
                "D": "#ffaa00", "E": "#ff6600", "F": "#ff3300", "G": "#cc0000"
            }
            st.markdown(f"""
            <div style='text-align:center; padding:20px; border-radius:12px;
                        background:rgba(10,0,40,0.6); border:2px solid {couleurs_dpe.get(classe[6], '#ffffff')};'>
                <div style='font-size:3rem; color:{couleurs_dpe.get(classe[6], "#ffffff")}'>
                    {classe}
                </div>
                <div style='color:#c0d0ff; font-size:1.2rem'>
                    {conso_m2:.0f} kWh/m²/an
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Barème DPE")
            df_dpe = pd.DataFrame([
                {"Classe": k, "Min (kWh/m²)": v[0], "Max (kWh/m²)": v[1]}
                for k, v in INDICATEURS_BATIMENT.items()
            ])
            st.dataframe(df_dpe, use_container_width=True)

    # ============================================================
    # TAB 6 : DÉTECTION POINTES
    # ============================================================
    with tab6:
        st.markdown("### 🔍 Détection des pointes de consommation")

        col1, col2 = st.columns([1, 2])
        with col1:
            col_pointe = st.selectbox("Variable", numeric_cols, key="pointe_col")
            seuil = st.slider("Seuil (σ)", 1.0, 4.0, 2.0, 0.1)

        with col2:
            pointes_mask = engine.detecter_pointes(col_pointe, seuil)
            s_p = df[col_pointe].dropna()
            idx = list(range(len(s_p)))
            n_pointes = pointes_mask.sum()

            st.metric("Pointes détectées", n_pointes)
            st.metric("% du temps", f"{n_pointes/len(s_p)*100:.1f}%")
            st.metric("Valeur seuil", f"{s_p.mean() + seuil*s_p.std():.2f}")

            colors_pt = ['#ff4444' if p else '#00ccff' for p in pointes_mask]
            fig_pt = go.Figure()
            fig_pt.add_trace(go.Scatter(
                x=idx, y=s_p.values, mode='lines', name='Consommation',
                line=dict(color='rgba(0,204,255,0.4)', width=1.5)
            ))
            fig_pt.add_trace(go.Scatter(
                x=[i for i, m in enumerate(pointes_mask) if m],
                y=[s_p.iloc[i] for i, m in enumerate(pointes_mask) if m],
                mode='markers', name='Pointes',
                marker=dict(color='#ff4444', size=8, symbol='circle')
            ))
            fig_pt.add_hline(
                y=s_p.mean() + seuil*s_p.std(),
                line_color='#ffcc00', line_dash='dash',
                annotation_text=f"Seuil {seuil}σ"
            )
            fig_pt.update_layout(
                title=f"Détection de pointes — {col_pointe}",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420,
            )
            st.plotly_chart(fig_pt, use_container_width=True)

    # ============================================================
    # TAB 7 : THÉORIE
    # ============================================================
    with tab7:
        st.markdown("### 📖 Formulaire scientifique énergétique")
        for nom, formule in FORMULES_ENERGIE.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("---")
        st.markdown("### 🔬 Constantes énergétiques")
        df_cst = pd.DataFrame([
            {"Constante": k, "Valeur": v[0], "Unité": v[1]}
            for k, v in CONSTANTES_ENERGIE.items()
        ])
        st.dataframe(df_cst, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "IEA — *World Energy Outlook* (2023)",
            "ADEME — *Guide de l'efficacité énergétique* (2022)",
            "Pérez-Lombard et al. — *A review on buildings energy consumption* (Energy & Buildings, 2008)",
            "MacKay — *Sustainable Energy Without the Hot Air* (UIT Cambridge, 2009)",
        ]:
            st.markdown(f"- {r}")

    # Export
    st.markdown("---")
    st.download_button("💾 Export CSV", df.to_csv(index=False).encode(),
                       "energy_export.csv", "text/csv")