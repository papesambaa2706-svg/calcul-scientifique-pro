import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, silhouette_score)
from sklearn.pipeline import Pipeline
import pandas as pd

# ============================================================
# FORMULAIRE
# ============================================================
FORMULES_ML = {
    "R²":           r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}",
    "RMSE":         r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2}",
    "Ridge (L2)":   r"\hat{\beta} = \arg\min\left[\|y-X\beta\|^2 + \lambda\|\beta\|^2\right]",
    "Lasso (L1)":   r"\hat{\beta} = \arg\min\left[\|y-X\beta\|^2 + \lambda\|\beta\|_1\right]",
    "PCA":          r"\mathbf{X}_{réduit} = \mathbf{X}\mathbf{W}_k,\quad \mathbf{W}_k \in \mathbb{R}^{p\times k}",
    "K-Means":      r"J = \sum_{k=1}^K\sum_{x\in C_k}\|x-\mu_k\|^2",
    "Silhouette":   r"s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}",
    "Corrélation":  r"r_{xy} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2}}",
}

MODELES_INFO = {
    "Régression Linéaire": {"type": "Linéaire", "régularisation": "Non",   "interprétable": "✅"},
    "Ridge (L2)":          {"type": "Linéaire", "régularisation": "L2",    "interprétable": "✅"},
    "Lasso (L1)":          {"type": "Linéaire", "régularisation": "L1",    "interprétable": "✅"},
    "ElasticNet":          {"type": "Linéaire", "régularisation": "L1+L2", "interprétable": "✅"},
    "Random Forest":       {"type": "Ensemble", "régularisation": "Non",   "interprétable": "⚠️"},
    "Gradient Boosting":   {"type": "Ensemble", "régularisation": "Non",   "interprétable": "⚠️"},
    "SVR":                 {"type": "Kernel",   "régularisation": "C",     "interprétable": "❌"},
}


# ============================================================
# CLASSE MOTEUR DATA SCIENCE
# ============================================================
class DataScienceEngine:
    """Moteur d'analyse, ML et visualisation scientifique."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = list(df.select_dtypes(include=np.number).columns)
        self.cat_cols = list(df.select_dtypes(include='object').columns)

    def profil_complet(self) -> pd.DataFrame:
        """Profil statistique complet par colonne."""
        rows = []
        for col in self.numeric_cols:
            s = self.df[col].dropna()
            rows.append({
                "Colonne": col,
                "Dtype": str(self.df[col].dtype),
                "Count": len(s),
                "Missing (%)": f"{self.df[col].isnull().mean()*100:.1f}",
                "Moyenne": f"{s.mean():.4f}",
                "Médiane": f"{s.median():.4f}",
                "Std": f"{s.std():.4f}",
                "Min": f"{s.min():.4f}",
                "Max": f"{s.max():.4f}",
                "Skewness": f"{stats.skew(s):.3f}",
                "Kurtosis": f"{stats.kurtosis(s):.3f}",
                "Outliers (IQR)": int(((s < s.quantile(0.25) - 1.5*(s.quantile(0.75)-s.quantile(0.25))) |
                                       (s > s.quantile(0.75) + 1.5*(s.quantile(0.75)-s.quantile(0.25)))).sum()),
            })
        return pd.DataFrame(rows)

    def detecter_outliers_zscore(self, col: str, seuil: float = 3.0) -> np.ndarray:
        s = self.df[col].dropna()
        return np.abs(stats.zscore(s)) > seuil

    def entrainer_modele(self, X: pd.DataFrame, y: pd.Series,
                          modele_nom: str, test_size: float,
                          scaler_nom: str, **kwargs) -> dict:
        """Pipeline complet : prétraitement → entraînement → évaluation → CV."""
        # Scaler
        scalers = {
            "Aucun": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }
        scaler = scalers.get(scaler_nom)

        # Modèle
        modeles = {
            "Régression Linéaire": LinearRegression(),
            "Ridge (L2)": Ridge(alpha=kwargs.get("alpha", 1.0)),
            "Lasso (L1)": Lasso(alpha=kwargs.get("alpha", 0.1), max_iter=10000),
            "ElasticNet": ElasticNet(alpha=kwargs.get("alpha", 0.1),
                                     l1_ratio=kwargs.get("l1_ratio", 0.5),
                                     max_iter=10000),
            "Random Forest": RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100), random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100), random_state=42),
            "SVR": SVR(kernel=kwargs.get("kernel", "rbf"),
                       C=kwargs.get("C", 1.0)),
        }
        model = modeles[modele_nom]

        # Pipeline
        if scaler:
            pipe = Pipeline([("scaler", scaler), ("model", model)])
        else:
            pipe = Pipeline([("model", model)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Métriques
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        residus = y_test.values - y_pred

        # Cross-validation
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=5,
                                        scoring='r2', error_score='raise')
        except:
            cv_scores = np.array([r2])

        # Feature importance
        importance = None
        final_model = pipe.named_steps["model"]
        if hasattr(final_model, "feature_importances_"):
            importance = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            importance = np.abs(final_model.coef_)

        return {
            "pipe": pipe,
            "y_test": y_test, "y_pred": y_pred,
            "r2": r2, "rmse": rmse, "mae": mae,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "residus": residus,
            "importance": importance,
            "features": list(X.columns),
        }

    def pca_analyse(self, n_components: int = 2) -> dict:
        """Analyse en composantes principales."""
        X = self.df[self.numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(n_components, len(self.numeric_cols)))
        X_pca = pca.fit_transform(X_scaled)
        return {
            "X_pca": X_pca,
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
            "components": pca.components_,
            "loadings": pd.DataFrame(
                pca.components_.T,
                index=self.numeric_cols,
                columns=[f"PC{i+1}" for i in range(pca.n_components_)]
            )
        }

    def clustering_kmeans(self, n_clusters: int) -> dict:
        """K-Means avec score de silhouette."""
        X = self.df[self.numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
        inertia = km.inertia_
        return {"labels": labels, "centers": km.cluster_centers_,
                "silhouette": sil, "inertia": inertia,
                "X_scaled": X_scaled}

    def elbow_method(self, k_max: int = 10) -> tuple:
        """Méthode du coude pour choisir k optimal."""
        X = self.df[self.numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        inertias, silhouettes = [], []
        ks = range(2, min(k_max+1, len(X)))
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))
        return list(ks), inertias, silhouettes


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def data_science_page():
    st.markdown("## 📊 Data Science & Machine Learning")
    st.markdown("*Analyse complète, ML multi-modèles, PCA, clustering, diagnostic*")
    st.markdown("---")

    # Upload
    uploaded = st.file_uploader("📁 Importer CSV ou Excel", type=["csv", "xlsx"])

    if uploaded is None:
        st.info("👆 Importez un fichier CSV ou Excel pour commencer.")
        st.markdown("#### Exemple de datasets compatibles")
        ex_data = {
            "Advertising": "TV, Radio, Newspaper → Sales",
            "Boston Housing": "Caractéristiques maisons → Prix",
            "Iris": "Mesures fleurs → Classification",
            "Énergie": "Paramètres → Consommation",
        }
        for k, v in ex_data.items():
            st.markdown(f"- **{k}** : {v}")
        return

    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
             else pd.read_excel(uploaded)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        st.success(f"✅ {df.shape[0]} lignes × {df.shape[1]} colonnes chargées")
    except Exception as e:
        st.error(f"❌ Erreur lecture: {e}")
        return

    engine = DataScienceEngine(df)
    numeric_cols = engine.numeric_cols

    if not numeric_cols:
        st.error("❌ Aucune colonne numérique détectée.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Profil",
        "📊 Visualisation",
        "🔗 Corrélations",
        "🤖 Machine Learning",
        "🔬 PCA",
        "🧩 Clustering",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : PROFIL COMPLET
    # ============================================================
    with tab1:
        st.markdown("### 📋 Profil complet du dataset")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Lignes", df.shape[0])
        with c2: st.metric("Colonnes", df.shape[1])
        with c3: st.metric("Valeurs manquantes", int(df.isnull().sum().sum()))
        with c4: st.metric("Variables numériques", len(numeric_cols))

        st.markdown("#### 📊 Statistiques avancées")
        df_profil = engine.profil_complet()
        st.dataframe(df_profil, use_container_width=True)

        st.markdown("#### ⚠️ Valeurs manquantes")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            fig_miss = go.Figure(go.Bar(
                x=missing.index, y=missing.values,
                marker=dict(color='#ff4444'),
            ))
            fig_miss.update_layout(
                title="Valeurs manquantes par colonne",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=300,
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅ Aucune valeur manquante")

        st.markdown("#### 🔎 Détection d'outliers (Z-score > 3)")
        col_out = st.selectbox("Colonne", numeric_cols, key="out_col")
        outliers_mask = engine.detecter_outliers_zscore(col_out)
        n_out = outliers_mask.sum()
        st.metric(f"Outliers dans {col_out}", n_out)

        fig_out = go.Figure()
        s_col = df[col_out].dropna()
        colors_out = ['#ff4444' if o else '#00ccff' for o in outliers_mask]
        fig_out.add_trace(go.Scatter(
            x=list(range(len(s_col))), y=s_col.values, mode='markers',
            marker=dict(color=colors_out, size=5), name=col_out
        ))
        fig_out.update_layout(
            title=f"Outliers — {col_out}",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
            font=dict(color='#c0d0ff'),
            xaxis=dict(color='#c0d0ff'), yaxis=dict(color='#c0d0ff',
            gridcolor='rgba(100,0,255,0.2)'), height=320,
        )
        st.plotly_chart(fig_out, use_container_width=True)

    # ============================================================
    # TAB 2 : VISUALISATION
    # ============================================================
    with tab2:
        st.markdown("### 📊 Visualisation avancée")

        col1, col2 = st.columns([1, 2])
        with col1:
            graph_type = st.selectbox("Type", [
                "Scatter + tendance", "Histogramme + KDE",
                "Boxplot", "Violin", "Pairplot (3 var)", "Heatmap"
            ])
            x_col = st.selectbox("Variable X", numeric_cols, key="vx")
            y_col = st.selectbox("Variable Y", numeric_cols,
                                  index=min(1, len(numeric_cols)-1), key="vy")

        with col2:
            if graph_type == "Scatter + tendance":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[y_col], mode='markers',
                    marker=dict(color=df[y_col],
                               colorscale=[[0,'#7700ff'],[0.5,'#00ccff'],[1,'#ffffff']],
                               size=7, showscale=True),
                    name='Données'
                ))
                # Régression
                mask = df[[x_col, y_col]].dropna()
                if len(mask) > 2:
                    slope, intercept, r, p, se = stats.linregress(mask[x_col], mask[y_col])
                    x_r = np.linspace(mask[x_col].min(), mask[x_col].max(), 200)
                    fig.add_trace(go.Scatter(
                        x=x_r, y=slope*x_r + intercept, mode='lines',
                        name=f'y={slope:.3f}x+{intercept:.3f} (R²={r**2:.3f})',
                        line=dict(color='#ffcc00', width=2.5)
                    ))

            elif graph_type == "Histogramme + KDE":
                fig = go.Figure()
                data_h = df[x_col].dropna()
                fig.add_trace(go.Histogram(
                    x=data_h, nbinsx=40, name='Distribution',
                    marker=dict(color='rgba(119,0,255,0.6)',
                               line=dict(color='rgba(0,204,255,0.8)', width=0.5))
                ))
                kde = stats.gaussian_kde(data_h)
                x_kde = np.linspace(data_h.min(), data_h.max(), 300)
                scale = len(data_h) * (data_h.max()-data_h.min()) / 40
                fig.add_trace(go.Scatter(
                    x=x_kde, y=kde(x_kde)*scale, mode='lines', name='KDE',
                    line=dict(color='#00ccff', width=3)
                ))

            elif graph_type == "Boxplot":
                fig = go.Figure()
                for col in numeric_cols[:6]:
                    fig.add_trace(go.Box(y=df[col].dropna(), name=col,
                                        boxmean='sd'))

            elif graph_type == "Violin":
                fig = go.Figure()
                for col in numeric_cols[:4]:
                    fig.add_trace(go.Violin(y=df[col].dropna(), name=col,
                                           box_visible=True, meanline_visible=True))

            elif graph_type == "Pairplot (3 var)":
                vars_sel = numeric_cols[:3]
                fig = make_subplots(rows=len(vars_sel), cols=len(vars_sel))
                for i, vi in enumerate(vars_sel):
                    for j, vj in enumerate(vars_sel):
                        if i == j:
                            kde_d = stats.gaussian_kde(df[vi].dropna())
                            x_k = np.linspace(df[vi].min(), df[vi].max(), 100)
                            fig.add_trace(go.Scatter(x=x_k, y=kde_d(x_k),
                                mode='lines', line=dict(color='#00ccff', width=2)),
                                row=i+1, col=j+1)
                        else:
                            fig.add_trace(go.Scatter(
                                x=df[vj], y=df[vi], mode='markers',
                                marker=dict(size=3, color='rgba(0,204,255,0.5)')),
                                row=i+1, col=j+1)

            else:  # Heatmap
                corr = df[numeric_cols].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale=[[0,'#020817'],[0.5,'#7700ff'],[1,'#00ccff']],
                    zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
                    colorbar=dict(tickfont=dict(color='#c0d0ff'))
                ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 3 : CORRÉLATIONS
    # ============================================================
    with tab3:
        st.markdown("### 🔗 Analyse des corrélations")

        corr_method = st.radio("Méthode", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = df[numeric_cols].corr(method=corr_method)

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,'#020817'],[0.25,'#7700ff'],[0.5,'#ffffff'],
                       [0.75,'#00ccff'],[1,'#00ff88']],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(title='r', tickfont=dict(color='#c0d0ff'))
        ))
        fig_corr.update_layout(
            title=f"Matrice de corrélation ({corr_method})",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
            font=dict(color='#c0d0ff'), height=500,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top corrélations
        st.markdown("#### 🏆 Top corrélations")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    "Variable 1": corr.columns[i],
                    "Variable 2": corr.columns[j],
                    "Corrélation": round(corr.iloc[i, j], 4),
                    "|r|": round(abs(corr.iloc[i, j]), 4),
                    "Force": "Forte" if abs(corr.iloc[i, j]) > 0.7
                             else "Modérée" if abs(corr.iloc[i, j]) > 0.4 else "Faible"
                })
        df_pairs = pd.DataFrame(corr_pairs).sort_values("|r|", ascending=False)
        st.dataframe(df_pairs, use_container_width=True)

    # ============================================================
    # TAB 4 : MACHINE LEARNING
    # ============================================================
    with tab4:
        st.markdown("### 🤖 Machine Learning — Régression")

        col1, col2 = st.columns([1, 2])
        with col1:
            target = st.selectbox("Variable cible (y)", numeric_cols)
            features = st.multiselect(
                "Variables explicatives (X)",
                [c for c in numeric_cols if c != target],
                default=[c for c in numeric_cols if c != target][:3]
            )
            modele_nom = st.selectbox("Modèle", list(MODELES_INFO.keys()))
            scaler_nom = st.selectbox("Normalisation", ["Aucun", "StandardScaler", "MinMaxScaler"])
            test_size = st.slider("Taille test (%)", 10, 40, 20) / 100

            kwargs = {}
            if modele_nom in ["Ridge (L2)", "Lasso (L1)", "ElasticNet"]:
                kwargs["alpha"] = st.slider("Alpha (λ)", 0.001, 10.0, 1.0, 0.001)
            if modele_nom == "ElasticNet":
                kwargs["l1_ratio"] = st.slider("l1_ratio", 0.0, 1.0, 0.5, 0.01)
            if modele_nom in ["Random Forest", "Gradient Boosting"]:
                kwargs["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10)
            if modele_nom == "SVR":
                kwargs["kernel"] = st.selectbox("Kernel SVR", ["rbf", "linear", "poly"])
                kwargs["C"] = st.slider("C (SVR)", 0.01, 100.0, 1.0)

            lancer = st.button("🚀 Entraîner le modèle", use_container_width=True)

        with col2:
            if lancer and len(features) > 0:
                with st.spinner("Entraînement..."):
                    X = df[features].dropna()
                    y = df.loc[X.index, target]
                    res = engine.entrainer_modele(X, y, modele_nom,
                                                   test_size, scaler_nom, **kwargs)

                # Métriques
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("R²", f"{res['r2']:.4f}")
                with c2: st.metric("RMSE", f"{res['rmse']:.4f}")
                with c3: st.metric("MAE", f"{res['mae']:.4f}")
                with c4: st.metric("CV R² (5-fold)", f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

                # Réel vs Prédit
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=res['y_test'], y=res['y_pred'], mode='markers',
                    marker=dict(color='#00ccff', size=7, opacity=0.7),
                    name='Prédictions'
                ))
                lim = [min(res['y_test'].min(), res['y_pred'].min()),
                       max(res['y_test'].max(), res['y_pred'].max())]
                fig_pred.add_trace(go.Scatter(
                    x=lim, y=lim, mode='lines', name='Parfait',
                    line=dict(color='#ffcc00', width=2, dash='dash')
                ))
                fig_pred.update_layout(
                    title="Réel vs Prédit",
                    xaxis_title="Valeur réelle", yaxis_title="Valeur prédite",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=380,
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # Résidus
                fig_res = make_subplots(rows=1, cols=2,
                    subplot_titles=["Résidus vs prédit", "Distribution des résidus"])
                fig_res.add_trace(go.Scatter(
                    x=res['y_pred'], y=res['residus'], mode='markers',
                    marker=dict(color='#7700ff', size=6),
                    name='Résidus'
                ), row=1, col=1)
                fig_res.add_hline(y=0, line_color='#ffcc00', line_dash='dash', row=1, col=1)
                fig_res.add_trace(go.Histogram(
                    x=res['residus'], nbinsx=20,
                    marker=dict(color='rgba(119,0,255,0.6)'), name='Hist. résidus'
                ), row=1, col=2)
                fig_res.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'), height=320,
                    showlegend=False
                )
                fig_res.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_res.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                st.plotly_chart(fig_res, use_container_width=True)

                # Importance des variables
                if res['importance'] is not None:
                    st.markdown("#### 📌 Importance des variables")
                    imp_df = pd.DataFrame({
                        "Variable": res['features'],
                        "Importance": res['importance']
                    }).sort_values("Importance", ascending=True)
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["Importance"], y=imp_df["Variable"],
                        orientation='h',
                        marker=dict(color=imp_df["Importance"],
                                   colorscale=[[0,'#7700ff'],[1,'#00ccff']])
                    ))
                    fig_imp.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                        font=dict(color='#c0d0ff'),
                        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                        yaxis=dict(color='#c0d0ff'), height=300,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Prédiction utilisateur
                st.markdown("#### 🧠 Prédiction personnalisée")
                input_data = {}
                cols_inp = st.columns(min(len(features), 4))
                for i, feat in enumerate(features):
                    with cols_inp[i % len(cols_inp)]:
                        input_data[feat] = st.number_input(
                            feat, value=float(df[feat].mean()), step=0.01, key=f"inp_{feat}")

                if st.button("🎯 Prédire", key="pred_btn"):
                    inp_df = pd.DataFrame([input_data])
                    pred_val = res['pipe'].predict(inp_df)[0]
                    st.success(f"✅ Valeur prédite de **{target}** : **{pred_val:.4f}**")

    # ============================================================
    # TAB 5 : PCA
    # ============================================================
    with tab5:
        st.markdown("### 🔬 Analyse en Composantes Principales (PCA)")

        if len(numeric_cols) < 2:
            st.warning("Au moins 2 variables numériques requises.")
        else:
            n_comp = st.slider("Nombre de composantes", 2,
                                min(len(numeric_cols), 10), 2)
            pca_res = engine.pca_analyse(n_comp)

            # Variance expliquée
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(pca_res['explained_variance']))],
                y=pca_res['explained_variance'] * 100,
                marker=dict(color='#7700ff'), name='Variance individuelle'
            ))
            fig_var.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(pca_res['cumulative']))],
                y=pca_res['cumulative'] * 100,
                mode='lines+markers', name='Variance cumulée',
                line=dict(color='#00ccff', width=2.5),
                yaxis='y2'
            ))
            fig_var.add_hline(y=90, line_color='#ffcc00', line_dash='dash',
                              annotation_text="90%", yref='y2')
            fig_var.update_layout(
                title="Variance expliquée par composante",
                yaxis=dict(title="Variance (%)", color='#c0d0ff',
                          gridcolor='rgba(100,0,255,0.2)'),
                yaxis2=dict(title="Cumulée (%)", overlaying='y', side='right',
                           color='#00ccff', range=[0, 105]),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff'), legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=380,
            )
            st.plotly_chart(fig_var, use_container_width=True)

            # Projection PC1-PC2
            X_pca = pca_res['X_pca']
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                marker=dict(color='#00ccff', size=6, opacity=0.7),
                name='Observations'
            ))
            fig_pca.update_layout(
                title=f"Projection PC1-PC2 ({pca_res['explained_variance'][0]*100:.1f}% + {pca_res['explained_variance'][1]*100:.1f}%)",
                xaxis_title="PC1", yaxis_title="PC2",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=400,
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            # Loadings
            st.markdown("#### 📐 Loadings (contributions)")
            st.dataframe(pca_res['loadings'].round(4), use_container_width=True)

    # ============================================================
    # TAB 6 : CLUSTERING
    # ============================================================
    with tab6:
        st.markdown("### 🧩 Clustering non-supervisé")

        col1, col2 = st.columns([1, 2])
        with col1:
            n_clust = st.slider("Nombre de clusters k", 2, 10, 3)
            var_x = st.selectbox("Variable X (visualisation)",
                                  numeric_cols, key="clust_x")
            var_y = st.selectbox("Variable Y (visualisation)",
                                  numeric_cols, index=min(1, len(numeric_cols)-1),
                                  key="clust_y")
            show_elbow = st.checkbox("Afficher méthode du coude", True)

        with col2:
            clust_res = engine.clustering_kmeans(n_clust)

            ca, cb = st.columns(2)
            with ca: st.metric("Score silhouette", f"{clust_res['silhouette']:.4f}")
            with cb: st.metric("Inertie", f"{clust_res['inertia']:.2f}")

            df_plot = df[[var_x, var_y]].dropna().copy()
            df_plot['Cluster'] = clust_res['labels'][:len(df_plot)]

            colors_c = ['#00ccff','#7700ff','#ff00cc','#00ff88',
                        '#ffcc00','#ff4400','#88ccff','#cc88ff','#ff88cc','#88ff88']
            fig_cl = go.Figure()
            for k in range(n_clust):
                mask_k = df_plot['Cluster'] == k
                fig_cl.add_trace(go.Scatter(
                    x=df_plot.loc[mask_k, var_x],
                    y=df_plot.loc[mask_k, var_y],
                    mode='markers', name=f'Cluster {k+1}',
                    marker=dict(color=colors_c[k % len(colors_c)], size=7, opacity=0.8)
                ))
            fig_cl.update_layout(
                title=f"K-Means Clustering (k={n_clust})",
                xaxis_title=var_x, yaxis_title=var_y,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420,
            )
            st.plotly_chart(fig_cl, use_container_width=True)

        if show_elbow:
            st.markdown("#### 📉 Méthode du coude")
            ks, inertias, silhouettes = engine.elbow_method(k_max=8)
            fig_elb = make_subplots(rows=1, cols=2,
                subplot_titles=["Inertie (coude)", "Score silhouette"])
            fig_elb.add_trace(go.Scatter(
                x=ks, y=inertias, mode='lines+markers',
                line=dict(color='#00ccff', width=2.5),
                marker=dict(size=8)
            ), row=1, col=1)
            fig_elb.add_trace(go.Scatter(
                x=ks, y=silhouettes, mode='lines+markers',
                line=dict(color='#7700ff', width=2.5),
                marker=dict(size=8)
            ), row=1, col=2)
            fig_elb.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=320, showlegend=False
            )
            fig_elb.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                  title_text="k")
            fig_elb.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_elb, use_container_width=True)

    # ============================================================
    # TAB 7 : THÉORIE
    # ============================================================
    with tab7:
        st.markdown("### 📖 Formulaire Machine Learning")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES_ML.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📊 Comparaison des modèles")
        df_mod = pd.DataFrame([
            {"Modèle": k, **v} for k, v in MODELES_INFO.items()
        ])
        st.dataframe(df_mod, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006)",
            "Hastie et al. — *The Elements of Statistical Learning* (Springer, 2009)",
            "Géron — *Hands-On Machine Learning* (O'Reilly, 2022)",
        ]:
            st.markdown(f"- {r}")

    # Export
    st.markdown("---")
    st.download_button("💾 Export CSV", df.to_csv(index=False).encode(),
                       "export_dataset.csv", "text/csv")