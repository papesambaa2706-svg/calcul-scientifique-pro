import streamlit as st
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORMULAIRE SCIENTIFIQUE
# ============================================================
FORMULES = {
    "Fonction de transfert":    r"H(s) = \frac{Y(s)}{U(s)} = \frac{b_m s^m + \cdots + b_0}{a_n s^n + \cdots + a_0}",
    "1er ordre":                r"H(s) = \frac{K}{\tau s + 1}",
    "2ème ordre":               r"H(s) = \frac{K\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}",
    "PID":                      r"C(s) = K_p\left(1 + \frac{1}{T_i s} + T_d s\right)",
    "Critère de Nyquist":       r"\text{Stable} \iff Z = N + P = 0",
    "Marge de gain":            r"G_m = -20\log|H(j\omega_{180°})|\text{ (dB)}",
    "Marge de phase":           r"\phi_m = 180° + \angle H(j\omega_c)",
    "Temps de réponse 5%":      r"t_r \approx \frac{3}{\zeta\omega_n}",
    "Dépassement":              r"D\% = 100\,e^{-\pi\zeta/\sqrt{1-\zeta^2}}",
    "BFCL":                     r"H_{BF}(s) = \frac{C(s)H(s)}{1 + C(s)H(s)}",
    "Stabilité Routh-Hurwitz":  r"\text{Stable} \iff \text{tous les coef. } a_i > 0",
}

SYSTEMES_PREDEFINIS = {
    "1er ordre — K/(τs+1)":         {"num": [1],        "den": [1, 1],      "label": "K=1, τ=1"},
    "2ème ordre — sous-amorti":      {"num": [4],        "den": [1, 0.4, 4], "label": "ωn=2, ζ=0.1"},
    "2ème ordre — critique":         {"num": [4],        "den": [1, 4, 4],   "label": "ωn=2, ζ=1"},
    "2ème ordre — sur-amorti":       {"num": [1],        "den": [1, 3, 2],   "label": "ζ=1.5"},
    "Intégrateur K/s":               {"num": [1],        "den": [1, 0],      "label": "K=1"},
    "Oscillateur pur":               {"num": [1],        "den": [1, 0, 1],   "label": "ωn=1"},
    "Système instable":              {"num": [1],        "den": [1, -1, 1],  "label": "pôle instable"},
    "3ème ordre":                    {"num": [1],        "den": [1, 6, 11, 6],"label": "stable"},
}


# ============================================================
# MOTEUR AUTOMATIQUE
# ============================================================
class AutomatiqueEngine:
    """Moteur d'analyse de systèmes automatiques."""

    def __init__(self, num: list, den: list):
        self.num = num
        self.den = den
        try:
            self.sys = signal.TransferFunction(num, den)
            self.valide = True
        except:
            self.valide = False

    def reponse_indicielle(self, t_max: float, n: int = 2000) -> tuple:
        t = np.linspace(0, t_max, n)
        try:
            t_out, y_out = signal.step(self.sys, T=t)
            return t_out, y_out
        except:
            return t, np.zeros(n)

    def reponse_impulsionnelle(self, t_max: float, n: int = 2000) -> tuple:
        t = np.linspace(0, t_max, n)
        try:
            t_out, y_out = signal.impulse(self.sys, T=t)
            return t_out, y_out
        except:
            return t, np.zeros(n)

    def reponse_rampe(self, t_max: float, n: int = 2000) -> tuple:
        t = np.linspace(0, t_max, n)
        try:
            num_r = np.polymul(self.num, [1, 0])
            den_r = np.polymul(self.den, [1, 0])
            sys_r = signal.TransferFunction(num_r, den_r)
            t_out, y_out = signal.step(sys_r, T=t)
            return t_out, y_out
        except:
            return t, t

    def bode(self, n: int = 1000) -> tuple:
        w = np.logspace(-3, 4, n)
        try:
            w_out, mag, phase = signal.bode(self.sys, w=w)
            return w_out, mag, phase
        except:
            return w, np.zeros(n), np.zeros(n)

    def poles_zeros(self) -> tuple:
        try:
            z, p, k = signal.tf2zpk(self.num, self.den)
            return z, p, k
        except:
            return np.array([]), np.array([]), 1.0

    def marges_stabilite(self) -> dict:
        try:
            gm, pm, wg, wp = signal.margin(self.sys)
            gm_db = 20 * np.log10(gm) if gm > 0 else np.inf
            stable = gm_db > 0 and pm > 0
            return {
                "gm_db": gm_db, "pm_deg": pm,
                "wg_rad": wg, "wp_rad": wp,
                "stable": stable,
            }
        except:
            return {"gm_db": np.nan, "pm_deg": np.nan,
                    "wg_rad": np.nan, "wp_rad": np.nan, "stable": None}

    def caracteristiques_indicielle(self, t: np.ndarray, y: np.ndarray) -> dict:
        """Analyse temporelle complète de la réponse indicielle."""
        if len(y) == 0 or np.all(np.isnan(y)):
            return {}
        val_fin = y[-1]
        if abs(val_fin) < 1e-10:
            return {"valeur_finale": 0}

        # Temps de montée (10% → 90%)
        try:
            idx_10 = np.where(y >= 0.1 * val_fin)[0][0]
            idx_90 = np.where(y >= 0.9 * val_fin)[0][0]
            t_montee = t[idx_90] - t[idx_10]
        except: t_montee = np.nan

        # Temps de réponse 5%
        try:
            in_band = np.where(np.abs(y - val_fin) <= 0.05 * abs(val_fin))[0]
            settled = next((i for i in range(len(in_band)-10)
                           if all(in_band[i:i+10] == np.arange(in_band[i], in_band[i]+10))),
                          None)
            t_reponse = t[in_band[settled]] if settled is not None else t[-1]
        except: t_reponse = np.nan

        # Dépassement
        y_max = np.max(y)
        depassement = (y_max - val_fin) / abs(val_fin) * 100 if y_max > val_fin else 0

        # Premier pic
        try:
            idx_pic = np.argmax(y)
            t_pic = t[idx_pic]
        except: t_pic = np.nan

        return {
            "valeur_finale": val_fin,
            "t_montee_s": t_montee,
            "t_reponse_5pct_s": t_reponse,
            "depassement_pct": depassement,
            "t_premier_pic_s": t_pic,
            "y_max": y_max,
        }

    def simuler_pid(self, Kp: float, Ki: float, Kd: float,
                     t_max: float, n: int = 2000) -> tuple:
        """Simulation boucle fermée avec PID."""
        t = np.linspace(0, t_max, n)
        dt = t[1] - t[0]
        ref = np.ones(n)
        y_out = np.zeros(n)
        e_int = 0.0
        e_prev = 0.0

        for i in range(1, n):
            e = ref[i] - y_out[i-1]
            e_int += e * dt
            e_der = (e - e_prev) / dt if dt > 0 else 0
            u = Kp * e + Ki * e_int + Kd * e_der
            # Réponse du système (Euler + step approximé)
            _, y_step = signal.step(self.sys, T=np.array([0, dt]))
            dy = y_step[-1] * u if len(y_step) > 1 else 0
            y_out[i] = y_out[i-1] + dy * dt
            y_out[i] = np.clip(y_out[i], -100, 100)
            e_prev = e

        return t, y_out

    def lieu_evans(self, k_range: np.ndarray) -> list:
        """Lieu des racines pour un gain K variable."""
        loci = []
        for k in k_range:
            num_k = [k * n for n in self.num]
            den_cl = np.polyadd(self.den, num_k)
            try:
                roots = np.roots(den_cl)
                loci.append(roots)
            except:
                loci.append(np.array([]))
        return loci

    def diagnostiquer(self) -> list:
        """Diagnostic automatique du système."""
        diag = []
        z, p, k = self.poles_zeros()
        marges = self.marges_stabilite()

        # Stabilité pôles
        poles_instables = [pi for pi in p if pi.real > 0]
        diag.append({
            "Test": "Stabilité (pôles)",
            "Valeur": f"{len(poles_instables)} pôle(s) instable(s)",
            "Statut": "✅ Stable" if len(poles_instables) == 0 else "❌ Instable",
            "Action": "OK" if len(poles_instables) == 0 else "Revoir les paramètres"
        })

        # Marge de gain
        gm = marges.get("gm_db", np.nan)
        diag.append({
            "Test": "Marge de gain",
            "Valeur": f"{gm:.1f} dB" if not np.isnan(gm) else "∞",
            "Statut": "✅ OK" if np.isnan(gm) or gm > 6 else "⚠️ Faible" if gm > 0 else "❌",
            "Action": "OK" if np.isnan(gm) or gm > 6 else "Réduire le gain"
        })

        # Marge de phase
        pm = marges.get("pm_deg", np.nan)
        diag.append({
            "Test": "Marge de phase",
            "Valeur": f"{pm:.1f}°" if not np.isnan(pm) else "N/A",
            "Statut": "✅ OK" if not np.isnan(pm) and pm > 45 else "⚠️ Faible" if not np.isnan(pm) and pm > 0 else "❌",
            "Action": "OK" if not np.isnan(pm) and pm > 45 else "Compenser en phase"
        })

        # Ordre
        ordre = len(self.den) - 1
        diag.append({
            "Test": "Ordre du système",
            "Valeur": str(ordre),
            "Statut": "✅ OK" if ordre <= 4 else "⚠️ Ordre élevé",
            "Action": "OK" if ordre <= 4 else "Réduction d'ordre conseillée"
        })

        return diag


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def automatique_page():
    st.markdown("## 🤖 Automatique & Systèmes de Contrôle")
    st.markdown("*Analyse fréquentielle, stabilité, PID, lieu des racines*")
    st.markdown("---")

    tab_system, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚙️ Système",
        "📈 Réponse temporelle",
        "📊 Bode & Nyquist",
        "🔵 Pôles & Zéros",
        "🎛️ Correcteur PID",
        "⚗️ Diagnostic",
        "📖 Théorie"
    ])

    # ============================================================
    # MODULE SISTÈME INTÉGRÉ DANS LA PAGE
    # ============================================================
    with tab_system:
        st.markdown("### ⚙️ Système")
        st.markdown("*Étudiez et paramétrez le système avant d’accéder aux outils d’analyse.*")

        config_mode = st.radio("Configuration", ["Prédéfini", "Manuel"], horizontal=True)

        if config_mode == "Prédéfini":
            sys_name = st.selectbox("Système", list(SYSTEMES_PREDEFINIS.keys()))
            sys_def = SYSTEMES_PREDEFINIS[sys_name]
            num = sys_def["num"]
            den = sys_def["den"]
            st.caption(sys_def["label"])
        else:
            st.markdown("**Numérateur (b₀,...,bₘ)**")
            num_str = st.text_input("num", "1")
            st.markdown("**Dénominateur (a₀,...,aₙ)**")
            den_str = st.text_input("den", "1 2 1")
            try:
                num = [float(x) for x in num_str.split()]
                den = [float(x) for x in den_str.split()]
            except:
                num, den = [1], [1, 1]

        t_max = st.slider("Temps max (s)", 1.0, 100.0, 20.0, 1.0)

    engine = AutomatiqueEngine(num, den)

    if not engine.valide:
        st.error("❌ Système invalide. Vérifiez num/den.")
        return

    # ============================================================
    # TAB 1 : RÉPONSE TEMPORELLE
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### ⚙️ Paramètres")
            resp_type = st.radio("Type de réponse", [
                "Indicielle (échelon)",
                "Impulsionnelle",
                "Rampe",
                "Sinusoïdale"
            ])

            if resp_type == "Sinusoïdale":
                f_sin = st.slider("Fréquence (rad/s)", 0.01, 100.0, 1.0, 0.01)

            n_pts = st.slider("Résolution", 500, 5000, 2000, 100)

        with col2:
            if resp_type == "Indicielle (échelon)":
                t, y = engine.reponse_indicielle(t_max, n_pts)
                carac = engine.caracteristiques_indicielle(t, y)
                title = "Réponse indicielle h(t)"
                ref_label = "Consigne"
                ref_val = y[-1] if len(y) > 0 else 1

            elif resp_type == "Impulsionnelle":
                t, y = engine.reponse_impulsionnelle(t_max, n_pts)
                carac = {}
                title = "Réponse impulsionnelle g(t)"
                ref_val = None

            elif resp_type == "Rampe":
                t, y = engine.reponse_rampe(t_max, n_pts)
                carac = {}
                title = "Réponse à la rampe"
                ref_val = None

            else:
                t = np.linspace(0, t_max, n_pts)
                u = np.sin(f_sin * t)
                _, y, _ = signal.lsim(engine.sys, u, t)
                carac = {}
                title = f"Réponse sinusoïdale (ω={f_sin} rad/s)"
                ref_val = None

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t, y=y, mode='lines', name='y(t)',
                line=dict(color='#00ccff', width=3)
            ))

            if resp_type == "Indicielle (échelon)" and carac:
                vf = carac.get("valeur_finale", 1)
                fig.add_hline(y=vf, line_color='#ffcc00', line_dash='dash',
                              annotation_text=f"y∞={vf:.3f}")
                fig.add_hline(y=vf * 1.05, line_color='rgba(255,100,0,0.4)',
                              line_dash='dot', annotation_text="+5%")
                fig.add_hline(y=vf * 0.95, line_color='rgba(255,100,0,0.4)',
                              line_dash='dot', annotation_text="-5%")

                tr = carac.get("t_reponse_5pct_s")
                if tr and not np.isnan(tr):
                    fig.add_vline(x=tr, line_color='#00ff88', line_dash='dash',
                                  annotation_text=f"tr={tr:.2f}s")

            elif resp_type == "Sinusoïdale":
                fig.add_trace(go.Scatter(
                    x=t, y=np.sin(f_sin * t), mode='lines', name='Entrée u(t)',
                    line=dict(color='rgba(119,0,255,0.5)', width=1.5, dash='dot')
                ))

            fig.update_layout(
                title=title, xaxis_title="Temps (s)", yaxis_title="Amplitude",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Caractéristiques temporelles
            if carac and resp_type == "Indicielle (échelon)":
                st.markdown("#### 📐 Caractéristiques temporelles")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Valeur finale", f"{carac.get('valeur_finale', 0):.4f}")
                with c2: st.metric("Dépassement (%)", f"{carac.get('depassement_pct', 0):.2f}")
                with c3: st.metric("t montée (s)", f"{carac.get('t_montee_s', 0):.3f}")
                with c4: st.metric("t réponse 5% (s)", f"{carac.get('t_reponse_5pct_s', 0):.3f}")

            # Export
            df_exp = pd.DataFrame({"t": t, "y": y})
            st.download_button("💾 Export CSV", df_exp.to_csv(index=False).encode(),
                               "reponse_temporelle.csv", "text/csv")

    # ============================================================
    # TAB 2 : BODE & NYQUIST
    # ============================================================
    with tab2:
        st.markdown("### 📊 Diagrammes fréquentiels")
        mode_freq = st.radio("Diagramme", ["Bode", "Nyquist", "Black-Nichols"], horizontal=True)

        w, mag, phase = engine.bode()
        marges = engine.marges_stabilite()

        if mode_freq == "Bode":
            fig_bode = make_subplots(rows=2, cols=1,
                subplot_titles=["Gain (dB)", "Phase (°)"],
                vertical_spacing=0.12)

            fig_bode.add_trace(go.Scatter(
                x=w, y=mag, mode='lines', name='|H(jω)| dB',
                line=dict(color='#00ccff', width=2.5)
            ), row=1, col=1)

            fig_bode.add_trace(go.Scatter(
                x=w, y=phase, mode='lines', name='∠H(jω)',
                line=dict(color='#7700ff', width=2.5)
            ), row=2, col=1)

            fig_bode.add_hline(y=0, line_color='rgba(255,255,255,0.3)',
                               line_dash='dash', row=1, col=1)
            fig_bode.add_hline(y=-3, line_color='#ffcc00', line_dash='dot',
                               annotation_text="-3dB", row=1, col=1)
            fig_bode.add_hline(y=-180, line_color='rgba(255,100,0,0.5)',
                               line_dash='dot', annotation_text="-180°", row=2, col=1)

            wp = marges.get("wp_rad")
            wg = marges.get("wg_rad")
            if wp and not np.isnan(wp):
                fig_bode.add_vline(x=wp, line_color='#00ff88', line_dash='dash',
                                   annotation_text=f"ωc={wp:.2f}", row=1, col=1)
            if wg and not np.isnan(wg):
                fig_bode.add_vline(x=wg, line_color='#ff00cc', line_dash='dash',
                                   annotation_text=f"ω180={wg:.2f}", row=2, col=1)

            fig_bode.update_xaxes(type='log', gridcolor='rgba(100,0,255,0.2)',
                                   color='#c0d0ff', title_text="ω (rad/s)")
            fig_bode.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
            fig_bode.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'), height=560,
                legend=dict(bgcolor='rgba(0,0,0,0.5)')
            )
            st.plotly_chart(fig_bode, use_container_width=True)

        elif mode_freq == "Nyquist":
            H_jw = np.array([np.polyval(engine.num, 1j*wi) /
                             np.polyval(engine.den, 1j*wi) for wi in w])
            fig_ny = go.Figure()
            fig_ny.add_trace(go.Scatter(
                x=H_jw.real, y=H_jw.imag, mode='lines', name='H(jω)',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_ny.add_trace(go.Scatter(
                x=H_jw.real[::-1], y=-H_jw.imag[::-1], mode='lines',
                name='H(−jω)', line=dict(color='rgba(0,204,255,0.3)', width=1.5, dash='dot')
            ))
            fig_ny.add_trace(go.Scatter(
                x=[-1], y=[0], mode='markers', name='Point critique (-1, 0)',
                marker=dict(color='#ff0000', size=14, symbol='x')
            ))
            fig_ny.update_layout(
                title="Diagramme de Nyquist",
                xaxis_title="Re[H(jω)]", yaxis_title="Im[H(jω)]",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)', zeroline=True,
                          zerolinecolor='rgba(255,255,255,0.3)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)', zeroline=True,
                          zerolinecolor='rgba(255,255,255,0.3)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=500,
            )
            st.plotly_chart(fig_ny, use_container_width=True)

        else:  # Black-Nichols
            fig_bn = go.Figure()
            fig_bn.add_trace(go.Scatter(
                x=phase, y=mag, mode='lines', name='H(jω)',
                line=dict(color='#00ccff', width=2.5)
            ))
            fig_bn.add_vline(x=-180, line_color='rgba(255,100,0,0.5)', line_dash='dash')
            fig_bn.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_dash='dash')
            fig_bn.update_layout(
                title="Abaque de Black-Nichols",
                xaxis_title="Phase (°)", yaxis_title="Gain (dB)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                height=480,
            )
            st.plotly_chart(fig_bn, use_container_width=True)

        # Marges
        st.markdown("#### 📐 Marges de stabilité")
        mc1, mc2, mc3, mc4 = st.columns(4)
        gm = marges.get("gm_db", np.nan)
        pm = marges.get("pm_deg", np.nan)
        with mc1: st.metric("Marge de gain (dB)", f"{gm:.2f}" if not np.isnan(gm) else "∞")
        with mc2: st.metric("Marge de phase (°)", f"{pm:.2f}" if not np.isnan(pm) else "N/A")
        with mc3: st.metric("ω crossover (rad/s)", f"{marges.get('wp_rad', 0):.3f}"
                            if marges.get('wp_rad') else "N/A")
        with mc4: st.metric("Stabilité", "✅ Stable" if marges.get("stable") else
                            "❌ Instable" if marges.get("stable") is False else "⚠️ N/A")

    # ============================================================
    # TAB 3 : PÔLES & ZÉROS
    # ============================================================
    with tab3:
        st.markdown("### 🔵 Plan des pôles et zéros")

        z, p, k = engine.poles_zeros()

        fig_pz = go.Figure()
        # Pôles
        if len(p) > 0:
            fig_pz.add_trace(go.Scatter(
                x=p.real, y=p.imag, mode='markers', name='Pôles',
                marker=dict(color='#ff4444', size=16, symbol='x',
                           line=dict(width=3, color='#ff4444'))
            ))
        # Zéros
        if len(z) > 0:
            fig_pz.add_trace(go.Scatter(
                x=z.real, y=z.imag, mode='markers', name='Zéros',
                marker=dict(color='#00ccff', size=14, symbol='circle-open',
                           line=dict(width=3))
            ))

        # Axe imaginaire (frontière stabilité)
        y_range = max(3, max(abs(p.imag.max()), abs(p.real.max())) + 1) if len(p) > 0 else 3
        fig_pz.add_vline(x=0, line_color='rgba(255,255,255,0.4)',
                         line_dash='dash', annotation_text="Stabilité")

        fig_pz.update_layout(
            title="Plan complexe — Pôles (×) et Zéros (○)",
            xaxis_title="Re(s)", yaxis_title="Im(s)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
            font=dict(color='#c0d0ff'),
            xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)',
                      zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
            yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)',
                      zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=480,
        )
        st.plotly_chart(fig_pz, use_container_width=True)

        # Tableau pôles
        st.markdown("#### 📊 Tableau des pôles")
        if len(p) > 0:
            df_poles = pd.DataFrame({
                "Pôle": [f"{pi:.4f}" for pi in p],
                "Re(p)": [f"{pi.real:.4f}" for pi in p],
                "Im(p)": [f"{pi.imag:.4f}" for pi in p],
                "Module": [f"{abs(pi):.4f}" for pi in p],
                "Stabilité": ["✅ Stable" if pi.real < 0 else "❌ Instable" for pi in p],
                "Amortissement ζ": [f"{-pi.real/abs(pi):.4f}" if abs(pi) > 0 else "N/A" for pi in p],
            })
            st.dataframe(df_poles, use_container_width=True)

        # Lieu d'Evans
        st.markdown("#### 🔄 Lieu des racines (Evans)")
        k_max_evans = st.slider("K max", 0.1, 100.0, 20.0, 0.1)
        k_range = np.linspace(0, k_max_evans, 200)
        loci = engine.lieu_evans(k_range)

        fig_evans = go.Figure()
        loci_array = np.array([r for r in loci if len(r) > 0])
        if len(loci_array) > 0 and loci_array.ndim == 2:
            for i in range(loci_array.shape[1]):
                fig_evans.add_trace(go.Scatter(
                    x=loci_array[:, i].real, y=loci_array[:, i].imag,
                    mode='lines', name=f'Branche {i+1}',
                    line=dict(width=2)
                ))
        fig_evans.add_vline(x=0, line_color='rgba(255,255,255,0.3)', line_dash='dash')
        fig_evans.update_layout(
            title="Lieu des racines",
            xaxis_title="Re(s)", yaxis_title="Im(s)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
            font=dict(color='#c0d0ff'),
            xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
            yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=420,
        )
        st.plotly_chart(fig_evans, use_container_width=True)

    # ============================================================
    # TAB 4 : PID
    # ============================================================
    with tab4:
        st.markdown("### 🎛️ Correcteur PID")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### ⚙️ Paramètres PID")
            Kp = st.slider("Kp (Proportionnel)", 0.0, 20.0, 1.0, 0.1)
            Ki = st.slider("Ki (Intégral)", 0.0, 10.0, 0.5, 0.05)
            Kd = st.slider("Kd (Dérivé)", 0.0, 5.0, 0.1, 0.05)

            # Info PID
            st.markdown(f"""
            **Effets :**
            - Kp={Kp} → {'Réduction erreur' if Kp > 0 else 'Inactif'}
            - Ki={Ki} → {'Erreur statique nulle' if Ki > 0 else 'Inactif'}
            - Kd={Kd} → {'Anticipation' if Kd > 0 else 'Inactif'}
            """)

            # Règles de Ziegler-Nichols
            with st.expander("📐 Réglage automatique (Ziegler-Nichols)"):
                Ku = st.slider("Ku (gain ultime)", 0.1, 50.0, 5.0, 0.1)
                Tu = st.slider("Tu (période ultime, s)", 0.1, 20.0, 2.0, 0.1)
                zn_Kp = 0.6 * Ku
                zn_Ki = 2 * zn_Kp / Tu
                zn_Kd = zn_Kp * Tu / 8
                st.metric("Kp Z-N", f"{zn_Kp:.3f}")
                st.metric("Ki Z-N", f"{zn_Ki:.3f}")
                st.metric("Kd Z-N", f"{zn_Kd:.3f}")

        with col2:
            # Boucle ouverte vs boucle fermée
            t_pid = np.linspace(0, t_max, 2000)
            t_bo, y_bo = engine.reponse_indicielle(t_max, 2000)

            # BF avec PID via scipy
            try:
                pid_num = [Kd, Kp, Ki]
                pid_den = [1, 0]
                pid_sys = signal.TransferFunction(pid_num, pid_den)

                # H_BF = C*H / (1 + C*H)
                num_ch = np.polymul(pid_num, engine.num)
                den_ch = np.polymul(pid_den, engine.den)
                num_bf = num_ch
                den_bf = np.polyadd(den_ch, num_ch)
                sys_bf = signal.TransferFunction(num_bf, den_bf)
                t_bf, y_bf = signal.step(sys_bf, T=t_pid)
                bf_ok = True
            except:
                t_bf, y_bf = t_pid, np.zeros_like(t_pid)
                bf_ok = False

            fig_pid = go.Figure()
            fig_pid.add_trace(go.Scatter(
                x=t_bo, y=y_bo, mode='lines', name='Sans correcteur',
                line=dict(color='rgba(119,0,255,0.5)', width=2, dash='dot')
            ))
            if bf_ok:
                fig_pid.add_trace(go.Scatter(
                    x=t_bf, y=y_bf, mode='lines', name='Avec PID',
                    line=dict(color='#00ccff', width=3)
                ))
            fig_pid.add_hline(y=1, line_color='#ffcc00', line_dash='dash',
                              annotation_text="Consigne")
            fig_pid.update_layout(
                title=f"Boucle fermée avec PID (Kp={Kp}, Ki={Ki}, Kd={Kd})",
                xaxis_title="Temps (s)", yaxis_title="y(t)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                yaxis=dict(color='#c0d0ff', gridcolor='rgba(100,0,255,0.2)'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'), height=450,
            )
            st.plotly_chart(fig_pid, use_container_width=True)

            if bf_ok:
                carac_pid = engine.caracteristiques_indicielle(t_bf, y_bf)
                if carac_pid:
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Dépassement (%)", f"{carac_pid.get('depassement_pct', 0):.2f}")
                    with c2: st.metric("t réponse 5% (s)", f"{carac_pid.get('t_reponse_5pct_s', 0):.3f}")
                    with c3: st.metric("Valeur finale", f"{carac_pid.get('valeur_finale', 0):.4f}")

    # ============================================================
    # TAB 5 : DIAGNOSTIC
    # ============================================================
    with tab5:
        st.markdown("### ⚗️ Diagnostic automatique du système")
        diag = engine.diagnostiquer()
        st.dataframe(pd.DataFrame(diag), use_container_width=True)

        st.markdown("#### 📋 Tableau de stabilité")
        err_table = {
            "Problème": ["Pôles instables", "Marge de gain faible", "Marge de phase faible",
                         "Dépassement excessif", "Oscillations persistantes"],
            "Cause": ["Re(p) > 0", "Gm < 6 dB", "φm < 30°",
                      "ζ trop faible", "Ki trop grand"],
            "Symptôme": ["y → ∞", "Sensible aux perturbations", "Instabilité conditionnelle",
                         "D% > 20%", "y oscille autour de y∞"],
            "Solution": ["Réduire K", "Correcteur à avance de phase", "Augmenter φm",
                         "Augmenter ζ via Kd", "Réduire Ki"]
        }
        st.dataframe(pd.DataFrame(err_table), use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire Automatique")
        cols = st.columns(2)
        col_idx = 0
        
        for nom, formule in FORMULES.items():
            with cols[col_idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{nom}**")
                    st.latex(formule)
            col_idx += 1

        st.markdown("---")
        st.markdown("### 📚 Références")
        for r in [
            "Ogata — *Modern Control Engineering* (Pearson, 2010)",
            "Franklin et al. — *Feedback Control of Dynamic Systems* (Pearson, 2014)",
            "Dorf & Bishop — *Modern Control Systems* (Pearson, 2016)",
        ]:
            st.markdown(f"- {r}")