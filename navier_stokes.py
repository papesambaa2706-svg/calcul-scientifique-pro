import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
FORMULES = {
    "Navier-Stokes incompressible": r"\rho\!\left(\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}\right)=-\nabla p+\mu\nabla^2\mathbf{u}+\mathbf{f}",
    "Continuité":                   r"\nabla\cdot\mathbf{u}=0",
    "Nombre de Reynolds":           r"Re=\frac{\rho U L}{\mu}=\frac{U L}{\nu}",
    "Équation de Poisson (p)":      r"\nabla^2 p = -\rho\,\nabla\cdot[(\mathbf{u}\cdot\nabla)\mathbf{u}]",
    "Vorticité":                    r"\boldsymbol{\omega}=\nabla\times\mathbf{u},\quad\omega_z=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y}",
    "Poiseuille":                   r"u(y)=U_{max}\left(1-\frac{y^2}{h^2}\right),\quad U_{max}=\frac{h^2}{2\mu}\left(-\frac{dp}{dx}\right)",
    "Couette":                      r"u(y)=U\frac{y}{h}",
    "Condition CFL":                r"C=\frac{u\,\Delta t}{\Delta x}\leq 1",
    "Énergie cinétique":            r"E_c=\frac{\rho}{2}\int\!(u^2+v^2)\,dV",
    "Dissipation visqueuse":        r"\Phi=\mu\int\!\left[2\left(\frac{\partial u}{\partial x}\right)^2+\cdots\right]dV",
}

PROFILS_ANALYTIQUES = {
    "Poiseuille":   "Écoulement laminaire entre plaques parallèles",
    "Couette":      "Cisaillement entre plaque fixe et mobile",
    "Tourbillon":   "Vortex potentiel 2D",
    "Stagnation":   "Point d'arrêt sur une paroi",
    "Jet gaussien": "Jet plan à profil gaussien",
}


# ============================================================
# FONCTIONS NUMÉRIQUES
# ============================================================
def build_obstacle(nx: int, ny: int, shape: str = "Cylindre") -> np.ndarray:
    obs = np.zeros((ny, nx))
    if shape == "Cylindre":
        cx, cy = nx//4, ny//2
        r = min(nx, ny) // 10
        for i in range(ny):
            for j in range(nx):
                if (j-cx)**2 + (i-cy)**2 < r**2:
                    obs[i, j] = 1
    elif shape == "Plaque":
        cy = ny//2
        obs[cy-2:cy+2, nx//4:nx//2] = 1
    elif shape == "Carré":
        x0, y0 = nx//4, ny//2 - ny//10
        w = ny//5
        obs[y0:y0+w, x0:x0+w] = 1
    elif shape == "Aile (NACA)":
        cx = nx//4
        for j in range(nx//4, nx//2):
            t = (j - nx//4) / (nx//4)
            y_naca = int(0.12 * ny * (0.2969*np.sqrt(t) - 0.1260*t -
                         0.3516*t**2 + 0.2843*t**3 - 0.1015*t**4))
            cy = ny//2
            if 0 < y_naca < ny//2:
                obs[cy-y_naca:cy+y_naca, j] = 1
    return obs


def laplacian(f: np.ndarray) -> np.ndarray:
    return (np.roll(f,1,0)+np.roll(f,-1,0)+
            np.roll(f,1,1)+np.roll(f,-1,1) - 4*f)


def navier_stokes_step(u: np.ndarray, v: np.ndarray, p: np.ndarray,
                        nu: float, dt: float, dx: float,
                        obstacle: np.ndarray,
                        U_inlet: float = 1.0) -> tuple:
    """Un pas de temps Navier-Stokes (projection method)."""
    # Gradients
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dx, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dx, axis=0)
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dx, axis=0)

    # Termes convectifs (schéma upwind)
    conv_u = u * du_dx + v * du_dy
    conv_v = u * dv_dx + v * dv_dy

    # Prédiction (sans pression)
    u_star = u + dt * (-conv_u + nu * laplacian(u) / dx**2)
    v_star = v + dt * (-conv_v + nu * laplacian(v) / dx**2)

    # CL entrée (inlet gauche)
    u_star[:, 0] = U_inlet
    v_star[:, 0] = 0

    # CL sortie (outlet droit — gradient nul)
    u_star[:, -1] = u_star[:, -2]
    v_star[:, -1] = v_star[:, -2]

    # Divergence
    div = np.gradient(u_star, dx, axis=1) + np.gradient(v_star, dx, axis=0)

    # Correction pression (Poisson simplifié — Jacobi)
    p_new = 0.25 * (np.roll(p,1,0)+np.roll(p,-1,0)+
                    np.roll(p,1,1)+np.roll(p,-1,1) - dx**2 * div)

    # Projection
    u_new = u_star - dt * np.gradient(p_new, dx, axis=1)
    v_new = v_star - dt * np.gradient(p_new, dx, axis=0)

    # Obstacle no-slip
    u_new[obstacle == 1] = 0
    v_new[obstacle == 1] = 0
    p_new[obstacle == 1] = 0

    # Stabilisation (clip)
    u_new = np.clip(u_new, -5, 5)
    v_new = np.clip(v_new, -5, 5)

    return u_new, v_new, p_new


def vorticite(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Champ de vorticité ω_z = ∂v/∂x - ∂u/∂y."""
    return np.gradient(v, dx, axis=1) - np.gradient(u, dx, axis=0)


def condition_cfl(u: np.ndarray, v: np.ndarray,
                  dt: float, dx: float) -> float:
    U_max = max(np.max(np.abs(u)), np.max(np.abs(v)), 1e-10)
    return U_max * dt / dx


@st.cache_data(show_spinner=False)
def build_obstacle(nx: int, ny: int, shape: str = "Cylindre") -> np.ndarray:
    obs = np.zeros((ny, nx))
    if shape == "Cylindre":
        cx, cy = nx//4, ny//2
        r = min(nx, ny) // 10
        for i in range(ny):
            for j in range(nx):
                if (j-cx)**2 + (i-cy)**2 < r**2:
                    obs[i, j] = 1
    elif shape == "Plaque":
        cy = ny//2
        obs[cy-2:cy+2, nx//4:nx//2] = 1
    elif shape == "Carré":
        x0, y0 = nx//4, ny//2 - ny//10
        w = ny//5
        obs[y0:y0+w, x0:x0+w] = 1
    elif shape == "Aile (NACA)":
        cx = nx//4
        for j in range(nx//4, nx//2):
            t = (j - nx//4) / (nx//4)
            y_naca = int(0.12 * ny * (0.2969*np.sqrt(t) - 0.1260*t -
                         0.3516*t**2 + 0.2843*t**3 - 0.1015*t**4))
            cy = ny//2
            if 0 < y_naca < ny//2:
                obs[cy-y_naca:cy+y_naca, j] = 1
    return obs


@st.cache_data(show_spinner=False)
def generate_vector_field_figure(flow_vf: str, nx_vf: int,
                                 U_vf: float, show_stream: bool) -> go.Figure:
    x_vf = np.linspace(-2, 2, nx_vf)
    y_vf = np.linspace(-2, 2, nx_vf)
    X_vf, Y_vf = np.meshgrid(x_vf, y_vf)
    r_vf = np.sqrt(X_vf**2 + Y_vf**2) + 0.01

    if flow_vf == "Tourbillon":
        U_f = -U_vf * Y_vf / r_vf**2 * (1 - np.exp(-r_vf**2))
        V_f =  U_vf * X_vf / r_vf**2 * (1 - np.exp(-r_vf**2))
    elif flow_vf == "Stagnation":
        U_f =  U_vf * X_vf
        V_f = -U_vf * Y_vf
    elif flow_vf == "Cisaillement":
        U_f = U_vf * Y_vf
        V_f = np.zeros_like(X_vf)
    elif flow_vf == "Double tourbillon":
        r1 = np.sqrt((X_vf-1)**2 + Y_vf**2) + 0.01
        r2 = np.sqrt((X_vf+1)**2 + Y_vf**2) + 0.01
        U_f = -U_vf*Y_vf/r1**2 + U_vf*Y_vf/r2**2
        V_f =  U_vf*(X_vf-1)/r1**2 - U_vf*(X_vf+1)/r2**2
    else:
        U_f = U_vf * X_vf / r_vf**2
        V_f = U_vf * Y_vf / r_vf**2

    speed_vf = np.sqrt(U_f**2 + V_f**2)
    U_n = U_f / (speed_vf + 1e-8)
    V_n = V_f / (speed_vf + 1e-8)

    try:
        fig_vf = ff.create_quiver(
            X_vf, Y_vf, U_n, V_n,
            scale=0.15, arrow_scale=0.25,
            line=dict(color='rgba(0,204,255,0.7)', width=1.5)
        )
    except Exception:
        fig_vf = go.Figure()

    fig_vf.add_trace(go.Heatmap(
        z=speed_vf, x=x_vf, y=y_vf,
        colorscale=[[0,'rgba(2,8,23,0)'],[0.5,'rgba(119,0,255,0.4)'],
                    [1,'rgba(0,204,255,0.6)']],
        showscale=True, opacity=0.5,
        colorbar=dict(title='|V|', tickfont=dict(color='#c0d0ff'))
    ))

    if show_stream:
        for y_start in np.linspace(-1.8, 1.8, 6):
            xs, ys = [x_vf[0]], [y_start]
            xc, yc = x_vf[0], y_start
            for _ in range(60):
                if abs(xc) > 2 or abs(yc) > 2:
                    break
                r_c = np.sqrt(xc**2 + yc**2) + 0.01
                if flow_vf == "Tourbillon":
                    uc = -U_vf * yc / r_c**2
                    vc =  U_vf * xc / r_c**2
                elif flow_vf == "Stagnation":
                    uc, vc = U_vf*xc, -U_vf*yc
                elif flow_vf == "Cisaillement":
                    uc, vc = U_vf*yc, 0
                else:
                    uc = U_vf * xc / r_c**2
                    vc = U_vf * yc / r_c**2
                nm = np.sqrt(uc**2+vc**2) + 1e-8
                xc += 0.05*uc/nm
                yc += 0.05*vc/nm
                xs.append(xc)
                ys.append(yc)
            fig_vf.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', showlegend=False,
                line=dict(color='rgba(255,200,0,0.35)', width=1.5)
            ))

    fig_vf.update_layout(
        title=f"Champ de vecteurs — {flow_vf}",
        xaxis_title="x", yaxis_title="y",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(5,0,20,0.8)',
        font=dict(color='#c0d0ff'),
        xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)',
                  range=[-2.2, 2.2]),
        yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)',
                  range=[-2.2, 2.2]),
        height=520,
    )
    return fig_vf


# ============================================================
# PROFILS ANALYTIQUES
# ============================================================
class AnalyticalFlows:

    @staticmethod
    def poiseuille(ny: int, U_max: float = 1.0) -> np.ndarray:
        y = np.linspace(-1, 1, ny)
        return U_max * (1 - y**2)

    @staticmethod
    def couette(ny: int, U_top: float = 1.0) -> np.ndarray:
        y = np.linspace(0, 1, ny)
        return U_top * y

    @staticmethod
    def tourbillon(nx: int, ny: int, Gamma: float = 1.0) -> tuple:
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2) + 0.01
        U = -Gamma * Y / (2 * np.pi * r**2)
        V =  Gamma * X / (2 * np.pi * r**2)
        return U, V

    @staticmethod
    def stagnation(nx: int, ny: int, k: float = 1.0) -> tuple:
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        X, Y = np.meshgrid(x, y)
        U =  k * X
        V = -k * Y
        return U, V

    @staticmethod
    def jet_gaussien(nx: int, ny: int,
                      U0: float = 1.0, sigma: float = 0.3) -> np.ndarray:
        y = np.linspace(-2, 2, ny)
        u_jet = U0 * np.exp(-y**2 / (2*sigma**2))
        return np.tile(u_jet[:, np.newaxis], (1, nx))


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def navier_stokes_page():
    st.markdown("## 🌊 Navier-Stokes CFD Avancé")
    st.markdown("*Simulation 2D, profils analytiques, vorticité, diagnostics*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Simulation CFD",
        "📈 Profils analytiques",
        "🌀 Champ de vecteurs",
        "⚗️ Diagnostic",
        "📖 Théorie"
    ])

    # ============================================================
    # TAB 1 : SIMULATION CFD
    # ============================================================
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            nx = st.slider("Grille X", 30, 120, 60)
            ny = st.slider("Grille Y", 30, 80, 40)
            obstacle_shape = st.selectbox("Obstacle", ["Cylindre", "Plaque", "Carré", "Aile (NACA)", "Aucun"])
        with col2:
            nu  = st.slider("Viscosité ν", 0.001, 0.5, 0.02, 0.001)
            dt  = st.slider("Pas temporel dt", 0.001, 0.05, 0.01, 0.001)
            U_in = st.slider("Vitesse inlet U", 0.1, 3.0, 1.0, 0.1)
        with col3:
            steps   = st.slider("Pas de simulation", 10, 500, 80)
            viz_type = st.selectbox("Visualisation", [
                "Vitesse U (heatmap)", "Vitesse V (heatmap)",
                "Vitesse |V| (norme)", "Pression", "Vorticité"
            ])
            use_quiver = st.checkbox("Superposer vecteurs", False)

        dx = 1.0

        # CFL info
        Re_approx = U_in * nx / (nu + 1e-10)
        cfl_approx = U_in * dt / dx
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Re (approx.)", f"{Re_approx:.1f}")
        with c2: st.metric("CFL", f"{cfl_approx:.3f}",
                           delta="OK" if cfl_approx < 1 else "⚠️ > 1")
        with c3: st.metric("Régime", "Laminaire" if Re_approx < 2300 else "Turbulent")

        if st.button("🚀 Lancer simulation CFD", width='stretch'):
            placeholder = st.empty()
            metrics_ph  = st.empty()

            # Init
            u = np.zeros((ny, nx))
            v = np.zeros((ny, nx))
            p = np.zeros((ny, nx))
            u[:, 0] = U_in

            obs = build_obstacle(nx, ny, obstacle_shape) \
                  if obstacle_shape != "Aucun" else np.zeros((ny, nx))

            history_Ekin = []
            history_vort = []

            for i in range(steps):
                u, v, p = navier_stokes_step(u, v, p, nu, dt, dx, obs, U_in)

                vort = vorticite(u, v, dx)
                E_kin = 0.5 * np.mean(u**2 + v**2)
                history_Ekin.append(E_kin)
                history_vort.append(np.std(vort))

                # Choisir donnée à afficher
                if viz_type == "Vitesse U (heatmap)":
                    Z_viz = u
                    titre = f"u(x,y) — step {i+1}/{steps}"
                elif viz_type == "Vitesse V (heatmap)":
                    Z_viz = v
                    titre = f"v(x,y) — step {i+1}/{steps}"
                elif viz_type == "Vitesse |V| (norme)":
                    Z_viz = np.sqrt(u**2 + v**2)
                    titre = f"|V|(x,y) — step {i+1}/{steps}"
                elif viz_type == "Pression":
                    Z_viz = p
                    titre = f"p(x,y) — step {i+1}/{steps}"
                else:
                    Z_viz = vort
                    titre = f"ω_z(x,y) — step {i+1}/{steps}"

                fig_sim = go.Figure()
                fig_sim.add_trace(go.Heatmap(
                    z=Z_viz,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    showscale=True,
                    colorbar=dict(tickfont=dict(color='#c0d0ff'))
                ))

                # Obstacle
                if obstacle_shape != "Aucun":
                    fig_sim.add_trace(go.Heatmap(
                        z=obs, colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(255,100,0,0.8)']],
                        showscale=False, opacity=0.7
                    ))

                # Quiver
                if use_quiver and i % 5 == 0:
                    step_q = max(nx//12, 1)
                    xi = np.arange(0, nx, step_q)
                    yi = np.arange(0, ny, step_q)
                    Xq, Yq = np.meshgrid(xi, yi)
                    Uq = u[::step_q, ::step_q][:len(yi), :len(xi)]
                    Vq = v[::step_q, ::step_q][:len(yi), :len(xi)]
                    norm_q = np.sqrt(Uq**2 + Vq**2) + 1e-8
                    fig_sim.add_trace(go.Scatter(
                        x=Xq.flatten(), y=Yq.flatten(),
                        mode='markers',
                        marker=dict(color='rgba(255,255,255,0.4)', size=2),
                        showlegend=False
                    ))

                fig_sim.update_layout(
                    title=titre,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    height=420,
                    margin=dict(l=10,r=10,t=40,b=10)
                )
                placeholder.plotly_chart(fig_sim, width='stretch')

                # Métriques live
                with metrics_ph.container():
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1: st.metric("E_cin", f"{E_kin:.4f}")
                    with mc2: st.metric("u_max", f"{np.max(np.abs(u)):.3f}")
                    with mc3: st.metric("Vorticité std", f"{np.std(vort):.4f}")
                    with mc4: st.metric("CFL réel", f"{condition_cfl(u,v,dt,dx):.3f}")

                time.sleep(0.02)

            st.success(f"✅ Simulation terminée — {steps} pas")

            # Graphiques finaux
            fig_post = make_subplots(rows=1, cols=2,
                subplot_titles=["Énergie cinétique E_c(t)", "Vorticité std(t)"])
            fig_post.add_trace(go.Scatter(y=history_Ekin, mode='lines',
                line=dict(color='#00ccff', width=2.5), name='E_c'), row=1, col=1)
            fig_post.add_trace(go.Scatter(y=history_vort, mode='lines',
                line=dict(color='#7700ff', width=2.5), name='|ω|'), row=1, col=2)
            fig_post.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=300, showlegend=False
            )
            fig_post.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                   title_text="Pas de temps")
            fig_post.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            st.plotly_chart(fig_post, width='stretch')

            # Export
            df_exp = pd.DataFrame({
                "u_mean": np.mean(u, axis=0),
                "v_mean": np.mean(v, axis=0),
                "p_mean": np.mean(p, axis=0),
                "vort_mean": np.mean(vorticite(u, v, dx), axis=0)
            })
            st.download_button("💾 Export profils CSV",
                               df_exp.to_csv(index=False).encode(),
                               "navier_stokes.csv", "text/csv")

    # ============================================================
    # TAB 2 : PROFILS ANALYTIQUES
    # ============================================================
    with tab2:
        st.markdown("### 📈 Solutions analytiques de Navier-Stokes")
        col1, col2 = st.columns([1, 2])

        with col1:
            profil = st.selectbox("Profil", list(PROFILS_ANALYTIQUES.keys()))
            st.caption(PROFILS_ANALYTIQUES[profil])
            nx_a = st.slider("nx", 20, 100, 50, key="nxa")
            ny_a = st.slider("ny", 20, 80, 40, key="nya")
            U0_a = st.slider("U₀", 0.1, 3.0, 1.0, 0.1, key="U0a")

            flows = AnalyticalFlows()

        with col2:
            if profil == "Poiseuille":
                u_a = np.tile(flows.poiseuille(ny_a, U0_a)[:, np.newaxis], (1, nx_a))
                v_a = np.zeros_like(u_a)
                titre_a = f"Poiseuille — u_max={U0_a}"
            elif profil == "Couette":
                u_a = np.tile(flows.couette(ny_a, U0_a)[:, np.newaxis], (1, nx_a))
                v_a = np.zeros_like(u_a)
                titre_a = f"Couette — U_top={U0_a}"
            elif profil == "Tourbillon":
                u_a, v_a = flows.tourbillon(nx_a, ny_a, Gamma=U0_a)
                titre_a = f"Tourbillon — Γ={U0_a}"
            elif profil == "Stagnation":
                u_a, v_a = flows.stagnation(nx_a, ny_a, k=U0_a)
                titre_a = f"Point de stagnation — k={U0_a}"
            else:
                u_a = flows.jet_gaussien(nx_a, ny_a, U0=U0_a)
                v_a = np.zeros_like(u_a)
                titre_a = f"Jet gaussien — U₀={U0_a}"

            speed_a = np.sqrt(u_a**2 + v_a**2)
            vort_a = vorticite(u_a, v_a, 1.0)

            fig_a = make_subplots(rows=1, cols=2,
                subplot_titles=["Vitesse |V|", "Vorticité ω_z"])

            fig_a.add_trace(go.Heatmap(
                z=speed_a,
                colorscale=[[0,'#020817'],[0.4,'#7700ff'],[0.7,'#00ccff'],[1,'#ffffff']],
                showscale=True, colorbar=dict(tickfont=dict(color='#c0d0ff'), x=0.45)
            ), row=1, col=1)
            fig_a.add_trace(go.Heatmap(
                z=vort_a,
                colorscale=[[0,'#020817'],[0.5,'#ffffff'],[1,'#ff4400']],
                showscale=True, colorbar=dict(tickfont=dict(color='#c0d0ff'))
            ), row=1, col=2)

            fig_a.update_layout(
                title=titre_a,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                height=400,
            )
            st.plotly_chart(fig_a, width='stretch')

            # Profil de vitesse
            if profil in ["Poiseuille", "Couette", "Jet gaussien"]:
                y_prof = np.linspace(0, 1, ny_a)
                u_prof = u_a[:, nx_a//2]
                fig_prof = go.Figure()
                fig_prof.add_trace(go.Scatter(
                    x=u_prof, y=y_prof, mode='lines',
                    name='u(y)', line=dict(color='#00ccff', width=3),
                    fill='tozerox', fillcolor='rgba(0,204,255,0.15)'
                ))
                fig_prof.update_layout(
                    title="Profil de vitesse u(y)",
                    xaxis_title="u", yaxis_title="y/H",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                    height=330,
                )
                st.plotly_chart(fig_prof, width='stretch')

    # ============================================================
    # TAB 3 : CHAMP DE VECTEURS
    # ============================================================
    with tab3:
        st.markdown("### 🌀 Champ de vecteurs interactif")
        col1, col2 = st.columns([1, 2])

        with col1:
            flow_vf = st.selectbox("Écoulement", [
                "Tourbillon", "Stagnation", "Cisaillement",
                "Double tourbillon", "Source"
            ])
            nx_vf = st.slider("Résolution", 15, 40, 20, key="nvf")
            U_vf = st.slider("Intensité", 0.1, 3.0, 1.0, 0.1, key="uvf")
            show_stream = st.checkbox("Lignes de courant (pseudo)", False)

        with col2:
            fig_vf = generate_vector_field_figure(flow_vf, nx_vf, U_vf, show_stream)
            st.plotly_chart(fig_vf, width='stretch')

    # ============================================================
    # TAB 4 : DIAGNOSTIC
    # ============================================================
    with tab4:
        st.markdown("### ⚗️ Diagnostic CFD")

        st.markdown("#### 📋 Tableau d'erreurs numériques")
        err_table = {
            "Problème": ["CFL > 1", "Divergence", "Instabilité oscillatoire",
                         "Boundary layer mal résolue", "Pression oscillante"],
            "Cause": ["dt trop grand", "ν trop faible ou dt grand",
                      "Schéma explicite instable", "Maillage trop grossier",
                      "Projection pression incomplète"],
            "Symptôme": ["Explosion de la vitesse", "NaN/Inf", "Oscillations parasites",
                         "Profil non physique", "Checker-board"],
            "Solution": ["Réduire dt", "Augmenter ν ou réduire dt",
                         "Schéma implicite", "Raffiner le maillage",
                         "Plus d'itérations Poisson"]
        }
        st.dataframe(pd.DataFrame(err_table), width='stretch')

        st.markdown("#### 📐 Critères de stabilité")
        nu_d  = st.slider("ν (diagnostic)", 0.001, 0.5, 0.02, 0.001)
        dt_d  = st.slider("dt", 0.001, 0.1, 0.01, 0.001)
        dx_d  = st.slider("dx", 0.01, 1.0, 0.1, 0.01)
        U_d   = st.slider("U_max", 0.1, 5.0, 1.0, 0.1)

        cfl_d    = U_d * dt_d / dx_d
        diff_num = nu_d * dt_d / dx_d**2
        Re_d     = U_d * dx_d / nu_d

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("CFL", f"{cfl_d:.4f}",
                      delta="✅ OK" if cfl_d < 1 else "❌ > 1")
        with c2:
            st.metric("Diffusion num. (νdt/dx²)", f"{diff_num:.4f}",
                      delta="✅ OK" if diff_num < 0.5 else "❌ > 0.5")
        with c3:
            st.metric("Re local", f"{Re_d:.2f}",
                      delta="Laminaire" if Re_d < 2300 else "Turbulent")

    # ============================================================
    # TAB 5 : THÉORIE
    # ============================================================
    with tab5:
        st.markdown("### 📖 Formulaire Navier-Stokes")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📊 Régimes d'écoulement")
        df_reg = pd.DataFrame([
            {"Régime": "Rampant (Stokes)", "Re": "< 1",    "Caractéristique": "Linéaire, réversible"},
            {"Régime": "Laminaire",         "Re": "1–2300", "Caractéristique": "Couches ordonnées"},
            {"Régime": "Transitoire",        "Re": "2300–4000","Caractéristique": "Instable"},
            {"Régime": "Turbulent",          "Re": "> 4000","Caractéristique": "Chaotique, 3D"},
        ])
        st.dataframe(df_reg, width='stretch')

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Batchelor — *An Introduction to Fluid Dynamics* (Cambridge, 2000)",
            "Ferziger & Perić — *Computational Methods for Fluid Dynamics* (Springer, 2020)",
            "Chorin — *Numerical solution of Navier-Stokes equations* (Math. Comp., 1968)",
        ]:
            st.markdown(f"- {r}")