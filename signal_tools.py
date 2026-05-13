import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sp_signal
from scipy import stats, integrate
from numpy.fft import fft, fftfreq, ifft, rfft, rfftfreq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES & FORMULAIRE
# ============================================================
FORMULES_SIGNAL = {
    "Transformée de Fourier": r"\hat{f}(\nu) = \int_{-\infty}^{+\infty} f(t)\,e^{-2\pi i \nu t}\,dt",
    "TFD (FFT)": r"X_k = \sum_{n=0}^{N-1} x_n\,e^{-2\pi i kn/N}",
    "Théorème de Shannon": r"f_s \geq 2 f_{max} \quad \text{(critère de Nyquist)}",
    "Énergie de Parseval": r"\sum_n |x_n|^2 = \frac{1}{N}\sum_k |X_k|^2",
    "Convolution": r"(f * g)(t) = \int_{-\infty}^{+\infty} f(\tau)g(t-\tau)\,d\tau",
    "Filtre passe-bas RC": r"H(f) = \frac{1}{1 + j2\pi f RC}",
    "Densité spectrale": r"S_{xx}(f) = \lim_{T\to\infty}\frac{1}{T}|X_T(f)|^2",
    "SNR": r"\text{SNR} = 10\log_{10}\frac{P_{signal}}{P_{bruit}} \quad \text{(dB)}",
    "THD": r"\text{THD} = \frac{\sqrt{\sum_{n=2}^{\infty} V_n^2}}{V_1} \times 100\%",
    "Chirp linéaire": r"s(t) = A\cos\left(2\pi\left(f_0 t + \frac{f_1 - f_0}{2T}t^2\right)\right)",
}

FILTRES_INFO = {
    "Butterworth": "Réponse maximalement plate en bande passante",
    "Chebyshev I": "Ondulations en bande passante, transition raide",
    "Chebyshev II": "Ondulations en bande rejetée, bande passante plate",
    "Elliptique": "Transition la plus raide possible (ondulations des deux côtés)",
    "Bessel": "Délai de groupe maximal plate, préservation de la forme d'onde",
}


# ============================================================
# MOTEUR SIGNAL
# ============================================================
class SignalEngine:
    """Moteur de traitement du signal avancé."""

    def __init__(self, fs: float = 2000.0):
        self.fs = fs

    # --- Génération ---
    def generer(self, sig_type: str, freq: float, amp: float,
                duration: float, freq2: float = None,
                noise_level: float = 0.0, phase: float = 0.0) -> tuple:
        N = int(self.fs * duration)
        t = np.linspace(0, duration, N, endpoint=False)

        if sig_type == "Sinus":
            s = amp * np.sin(2*np.pi*freq*t + phase)
        elif sig_type == "Carré":
            s = amp * sp_signal.square(2*np.pi*freq*t + phase)
        elif sig_type == "Triangle":
            s = amp * sp_signal.sawtooth(2*np.pi*freq*t + phase, 0.5)
        elif sig_type == "Dent de scie":
            s = amp * sp_signal.sawtooth(2*np.pi*freq*t + phase)
        elif sig_type == "Chirp":
            f2 = freq2 or freq * 5
            s = amp * sp_signal.chirp(t, f0=freq, f1=f2, t1=duration, method='linear')
        elif sig_type == "Chirp exponentiel":
            f2 = freq2 or freq * 10
            s = amp * sp_signal.chirp(t, f0=freq, f1=f2, t1=duration, method='exponential')
        elif sig_type == "Bruit blanc":
            s = amp * np.random.normal(0, 1, N)
        elif sig_type == "Bruit rose":
            white = np.random.normal(0, 1, N)
            f_n = rfftfreq(N, 1/self.fs)
            f_n[0] = 1
            pink_filter = 1 / np.sqrt(f_n)
            pink_filter[0] = 0
            s = amp * np.real(np.fft.irfft(np.fft.rfft(white) * pink_filter, n=N))
        elif sig_type == "Multi-sinusoïdal":
            harmoniques = [1, 2, 3, 5, 7]
            s = sum(amp/h * np.sin(2*np.pi*freq*h*t) for h in harmoniques)
        elif sig_type == "Impulsion gaussienne":
            t0 = duration / 2
            sigma = duration / 10
            s = amp * np.exp(-0.5*((t - t0)/sigma)**2) * np.cos(2*np.pi*freq*t)
        else:
            s = amp * np.sin(2*np.pi*freq*t)

        if noise_level > 0:
            s += np.random.normal(0, noise_level * amp, N)

        return t, s

    # --- FFT avancée ---
    def compute_fft(self, s: np.ndarray) -> tuple:
        N = len(s)
        win = np.hanning(N)
        s_win = s * win
        yf = rfft(s_win)
        xf = rfftfreq(N, 1/self.fs)
        magnitude = (2.0/N) * np.abs(yf)
        phase = np.angle(yf, deg=True)
        power_db = 20 * np.log10(magnitude + 1e-12)
        return xf, magnitude, phase, power_db

    # --- Métriques ---
    def metriques(self, s: np.ndarray) -> dict:
        rms = np.sqrt(np.mean(s**2))
        crest = np.max(np.abs(s)) / (rms + 1e-12)
        energy = np.sum(s**2) / self.fs
        xf, mag, _, _ = self.compute_fft(s)
        f_dom = xf[np.argmax(mag)] if len(mag) > 0 else 0

        # THD
        if f_dom > 0 and len(mag) > 0:
            idx_fund = np.argmax(mag)
            v1 = mag[idx_fund]
            harmonics_power = 0
            for h in range(2, 6):
                idx_h = np.argmin(np.abs(xf - h * f_dom))
                if idx_h < len(mag):
                    harmonics_power += mag[idx_h]**2
            thd = np.sqrt(harmonics_power) / (v1 + 1e-12) * 100
        else:
            thd = 0

        # SNR estimé
        noise_floor = np.percentile(mag, 10)
        signal_peak = np.max(mag)
        snr = 20 * np.log10(signal_peak / (noise_floor + 1e-12))

        return {
            "RMS": rms,
            "Crête": np.max(np.abs(s)),
            "Facteur de crête": crest,
            "Énergie (J/Ω)": energy,
            "Fréquence dominante (Hz)": f_dom,
            "THD (%)": thd,
            "SNR estimé (dB)": snr,
            "Moyenne": np.mean(s),
            "Écart-type": np.std(s),
            "Skewness": float(stats.skew(s)),
            "Kurtosis": float(stats.kurtosis(s)),
        }

    # --- Filtrage ---
    def filtrer(self, s: np.ndarray, ftype: str, btype: str,
                cutoff, order: int = 4) -> np.ndarray:
        nyq = self.fs / 2
        if isinstance(cutoff, (list, tuple)):
            Wn = [c / nyq for c in cutoff]
        else:
            Wn = cutoff / nyq

        Wn = np.clip(Wn, 1e-4, 0.9999)

        try:
            if ftype == "Butterworth":
                b, a = sp_signal.butter(order, Wn, btype=btype)
            elif ftype == "Chebyshev I":
                b, a = sp_signal.cheby1(order, 1, Wn, btype=btype)
            elif ftype == "Chebyshev II":
                b, a = sp_signal.cheby2(order, 40, Wn, btype=btype)
            elif ftype == "Elliptique":
                b, a = sp_signal.ellip(order, 1, 40, Wn, btype=btype)
            elif ftype == "Bessel":
                b, a = sp_signal.bessel(order, Wn, btype=btype, norm='phase')
            else:
                b, a = sp_signal.butter(order, Wn, btype=btype)

            return sp_signal.filtfilt(b, a, s), b, a
        except Exception as e:
            return s, None, None

    # --- Réponse fréquentielle filtre ---
    def reponse_filtre(self, b, a) -> tuple:
        w, h = sp_signal.freqz(b, a, worN=4096, fs=self.fs)
        return w, h

    # --- Autocorrélation ---
    def autocorrelation(self, s: np.ndarray) -> tuple:
        N = len(s)
        acf = np.correlate(s - s.mean(), s - s.mean(), mode='full')
        acf = acf / acf[N-1]
        lags = np.arange(-(N-1), N) / self.fs
        return lags, acf

    # --- Densité spectrale de puissance ---
    def dsp(self, s: np.ndarray) -> tuple:
        f, Pxx = sp_signal.welch(s, self.fs, nperseg=min(256, len(s)//4))
        return f, Pxx

    # --- Hilbert & enveloppe ---
    def enveloppe(self, s: np.ndarray) -> tuple:
        analytic = sp_signal.hilbert(s)
        envelope = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) / (2*np.pi) * self.fs
        return envelope, inst_phase, inst_freq

    # --- Cepstrum ---
    def cepstrum(self, s: np.ndarray) -> tuple:
        S = np.fft.fft(s)
        log_S = np.log(np.abs(S) + 1e-12)
        cep = np.real(np.fft.ifft(log_S))
        quefrency = np.arange(len(cep)) / self.fs
        return quefrency, cep


# ============================================================
# PAGE PRINCIPALE
# ============================================================
def signal_page():
    st.markdown("## 〰️ Traitement du Signal Avancé (DSP)")
    st.markdown("*Génération, analyse spectrale, filtrage, métriques, diagnostic*")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎛️ Génération & Analyse",
        "🔊 Filtrage",
        "📡 Analyses avancées",
        "📊 Métriques",
        "⚗️ Diagnostic",
        "📖 Théorie"
    ])

    # Sidebar de configuration partagée
    fs_global = 2000
    with st.sidebar.expander("⚙️ Config Signal", expanded=False):
        fs_global = st.slider("Fe (Hz)", 500, 20000, 2000, 500, key="fs_glob")

    engine = SignalEngine(fs=fs_global)

    # ============================================================
    # TAB 1 : GÉNÉRATION & ANALYSE
    # ============================================================
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🎛️ Paramètres")
            sig_type = st.selectbox("Type de signal", [
                "Sinus", "Carré", "Triangle", "Dent de scie",
                "Chirp", "Chirp exponentiel", "Bruit blanc",
                "Bruit rose", "Multi-sinusoïdal", "Impulsion gaussienne"
            ])
            freq = st.slider("Fréquence f₁ (Hz)", 1.0, 500.0, 10.0, 1.0)
            amp = st.slider("Amplitude A", 0.1, 10.0, 1.0, 0.1)
            duration = st.slider("Durée (s)", 0.1, 5.0, 1.0, 0.1)
            phase = st.slider("Phase φ (rad)", 0.0, 2*np.pi, 0.0, 0.01)
            noise = st.slider("Bruit relatif σ", 0.0, 1.0, 0.0, 0.01)
            analyse = st.radio("Mode d'analyse", [
                "Signal temporel", "FFT", "Spectrogramme",
                "DSP (Welch)", "Phase"
            ], horizontal=False)

        with col2:
            t, s = engine.generer(sig_type, freq, amp, duration,
                                  noise_level=noise, phase=phase)
            st.session_state['t_sig'] = t
            st.session_state['s_sig'] = s
            st.session_state['fs_sig'] = fs_global

            if analyse == "Signal temporel":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t, y=s, mode='lines', name='s(t)',
                    line=dict(color='#00ccff', width=2)
                ))
                fig.update_layout(
                    title=f"Signal temporel — {sig_type}",
                    xaxis_title="Temps (s)", yaxis_title="Amplitude",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif analyse == "FFT":
                xf, mag, phase_fft, power_db = engine.compute_fft(s)
                f_dom = xf[np.argmax(mag)] if len(mag) > 0 else 0

                fig = make_subplots(rows=2, cols=1,
                    subplot_titles=["Magnitude |X(f)|", "Phase ∠X(f) (°)"])

                fig.add_trace(go.Scatter(
                    x=xf, y=mag, mode='lines', name='|X(f)|',
                    line=dict(color='#00ccff', width=2),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.15)'
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=xf, y=phase_fft, mode='lines', name='Phase',
                    line=dict(color='#7700ff', width=1.5)
                ), row=2, col=1)

                fig.add_vline(x=f_dom, line_color='#ffcc00',
                             line_dash='dash', annotation_text=f"{f_dom:.1f} Hz")

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    height=500,
                )
                fig.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                title_text="Fréquence (Hz)")
                fig.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Fréquence dominante", f"{f_dom:.2f} Hz")
                with c2: st.metric("Amplitude crête", f"{np.max(mag):.4f}")
                with c3: st.metric("Bande passante", f"{xf[-1]:.0f} Hz")

            elif analyse == "Spectrogramme":
                nperseg = st.slider("Fenêtre (samples)", 32, 512, 128, key="spec_win")
                overlap = st.slider("Overlap (%)", 0, 95, 75, key="spec_ov")
                noverlap = int(nperseg * overlap / 100)
                f_spec, tt_spec, Sxx = sp_signal.spectrogram(
                    s, fs_global, nperseg=nperseg, noverlap=noverlap, window='hann'
                )
                Z_spec = 10 * np.log10(Sxx + 1e-12)

                fig = go.Figure(data=go.Heatmap(
                    z=Z_spec, x=tt_spec, y=f_spec,
                    colorscale=[[0,'#020817'],[0.3,'#7700ff'],[0.6,'#00ccff'],[1,'#ffffff']],
                    colorbar=dict(title='dB', tickfont=dict(color='#c0d0ff'))
                ))
                fig.update_layout(
                    title="Spectrogramme temps-fréquence",
                    xaxis_title="Temps (s)", yaxis_title="Fréquence (Hz)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(color='#c0d0ff'),
                    yaxis=dict(color='#c0d0ff'),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif analyse == "DSP (Welch)":
                f_dsp, Pxx = engine.dsp(s)
                Pxx_db = 10 * np.log10(Pxx + 1e-12)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=f_dsp, y=Pxx_db, mode='lines', name='DSP (Welch)',
                    line=dict(color='#00ccff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,204,255,0.1)'
                ))
                fig.update_layout(
                    title="Densité spectrale de puissance",
                    xaxis_title="Fréquence (Hz)", yaxis_title="PSD (dB/Hz)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=430,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif analyse == "Phase":
                xf, mag, phase_fft, _ = engine.compute_fft(s)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xf, y=phase_fft, mode='lines', name='Phase (°)',
                    line=dict(color='#7700ff', width=2)
                ))
                fig.update_layout(
                    title="Spectre de phase",
                    xaxis_title="Fréquence (Hz)", yaxis_title="Phase (°)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Export CSV
        df_export = pd.DataFrame({"temps": t, "signal": s})
        st.download_button("💾 Export CSV", df_export.to_csv(index=False).encode(),
                          "signal.csv", "text/csv")

    # ============================================================
    # TAB 2 : FILTRAGE
    # ============================================================
    with tab2:
        st.markdown("### 🔊 Filtrage avancé")

        if 's_sig' not in st.session_state:
            st.info("Générez d'abord un signal dans l'onglet Génération.")
        else:
            s_filt = st.session_state['s_sig']
            t_filt = st.session_state['t_sig']
            fs_filt = st.session_state['fs_sig']
            engine_filt = SignalEngine(fs=fs_filt)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### ⚙️ Configuration du filtre")
                ftype = st.selectbox("Type de filtre", list(FILTRES_INFO.keys()))
                st.info(FILTRES_INFO[ftype])
                btype = st.selectbox("Bande", ["low", "high", "bandpass", "bandstop"])
                order = st.slider("Ordre", 1, 10, 4)

                nyq = fs_filt / 2
                if btype in ["bandpass", "bandstop"]:
                    f_low = st.slider("Fréquence basse (Hz)", 1.0, nyq*0.45, min(20.0, nyq*0.1))
                    f_high = st.slider("Fréquence haute (Hz)", f_low+1, nyq*0.95, min(200.0, nyq*0.5))
                    cutoff = [f_low, f_high]
                else:
                    cutoff = st.slider("Fréquence de coupure (Hz)", 1.0, nyq*0.95, min(100.0, nyq*0.3))

            with col2:
                s_filtered, b_f, a_f = engine_filt.filtrer(s_filt, ftype, btype, cutoff, order)

                # Signal avant/après
                fig_f = make_subplots(rows=2, cols=1,
                    subplot_titles=["Signal temporel", "Spectre (avant/après)"])

                fig_f.add_trace(go.Scatter(
                    x=t_filt, y=s_filt, mode='lines', name='Original',
                    line=dict(color='rgba(0,204,255,0.4)', width=1.5, dash='dot')
                ), row=1, col=1)
                fig_f.add_trace(go.Scatter(
                    x=t_filt, y=s_filtered, mode='lines', name='Filtré',
                    line=dict(color='#00ccff', width=2.5)
                ), row=1, col=1)

                # Spectres
                xf_o, mag_o, _, _ = engine_filt.compute_fft(s_filt)
                xf_f, mag_f, _, _ = engine_filt.compute_fft(s_filtered)
                fig_f.add_trace(go.Scatter(
                    x=xf_o, y=mag_o, mode='lines', name='Spectre original',
                    line=dict(color='rgba(119,0,255,0.5)', width=1.5)
                ), row=2, col=1)
                fig_f.add_trace(go.Scatter(
                    x=xf_f, y=mag_f, mode='lines', name='Spectre filtré',
                    line=dict(color='#00ccff', width=2)
                ), row=2, col=1)

                fig_f.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_f.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                fig_f.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                st.plotly_chart(fig_f, use_container_width=True)

                # Réponse fréquentielle du filtre
                if b_f is not None and a_f is not None:
                    st.markdown("#### 📡 Réponse fréquentielle du filtre")
                    w_r, h_r = engine_filt.reponse_filtre(b_f, a_f)
                    mag_r = 20 * np.log10(np.abs(h_r) + 1e-12)
                    phase_r = np.angle(h_r, deg=True)

                    fig_resp = make_subplots(rows=2, cols=1,
                        subplot_titles=["Gain (dB)", "Phase (°)"])
                    fig_resp.add_trace(go.Scatter(
                        x=w_r, y=mag_r, mode='lines', name='Gain',
                        line=dict(color='#00ccff', width=2)
                    ), row=1, col=1)
                    fig_resp.add_trace(go.Scatter(
                        x=w_r, y=phase_r, mode='lines', name='Phase',
                        line=dict(color='#7700ff', width=2)
                    ), row=2, col=1)
                    fig_resp.add_hline(y=-3, line_color='#ffcc00', line_dash='dash',
                                      annotation_text="-3 dB", row=1, col=1)
                    fig_resp.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(5,0,20,0.8)',
                        font=dict(color='#c0d0ff'),
                        height=420,
                    )
                    fig_resp.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                         title_text="Fréquence (Hz)")
                    fig_resp.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                    st.plotly_chart(fig_resp, use_container_width=True)

    # ============================================================
    # TAB 3 : ANALYSES AVANCÉES
    # ============================================================
    with tab3:
        st.markdown("### 📡 Analyses avancées")

        if 's_sig' not in st.session_state:
            st.info("Générez d'abord un signal dans l'onglet Génération.")
        else:
            s_adv = st.session_state['s_sig']
            t_adv = st.session_state['t_sig']
            fs_adv = st.session_state['fs_sig']
            engine_adv = SignalEngine(fs=fs_adv)

            analyse_adv = st.radio("Analyse", [
                "Autocorrélation", "Enveloppe & Fréquence inst.",
                "Cepstrum", "Portrait de phase"
            ], horizontal=True)

            if analyse_adv == "Autocorrélation":
                lags, acf = engine_adv.autocorrelation(s_adv)
                max_lag = st.slider("Afficher lag max (s)", 0.01, min(0.5, t_adv[-1]/2), 0.1)
                mask = np.abs(lags) <= max_lag

                fig_acf = go.Figure()
                fig_acf.add_trace(go.Scatter(
                    x=lags[mask], y=acf[mask], mode='lines', name='ACF',
                    line=dict(color='#00ccff', width=2)
                ))
                fig_acf.add_hline(y=0, line_color='rgba(255,255,255,0.3)')
                fig_acf.add_hline(y=1.96/np.sqrt(len(s_adv)),
                                  line_color='#ffcc00', line_dash='dash',
                                  annotation_text="IC 95%")
                fig_acf.add_hline(y=-1.96/np.sqrt(len(s_adv)),
                                  line_color='#ffcc00', line_dash='dash')
                fig_acf.update_layout(
                    title="Fonction d'autocorrélation (ACF)",
                    xaxis_title="Décalage τ (s)", yaxis_title="R(τ)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=400,
                )
                st.plotly_chart(fig_acf, use_container_width=True)

            elif analyse_adv == "Enveloppe & Fréquence inst.":
                env, inst_ph, inst_f = engine_adv.enveloppe(s_adv)
                t_if = t_adv[:-1]

                fig_env = make_subplots(rows=2, cols=1,
                    subplot_titles=["Signal + Enveloppe", "Fréquence instantanée"])
                fig_env.add_trace(go.Scatter(
                    x=t_adv, y=s_adv, mode='lines', name='Signal',
                    line=dict(color='rgba(0,204,255,0.5)', width=1.5)
                ), row=1, col=1)
                fig_env.add_trace(go.Scatter(
                    x=t_adv, y=env, mode='lines', name='Enveloppe',
                    line=dict(color='#ff00cc', width=2.5)
                ), row=1, col=1)
                fig_env.add_trace(go.Scatter(
                    x=t_adv, y=-env, mode='lines', name='-Enveloppe',
                    line=dict(color='#ff00cc', width=2.5, dash='dot')
                ), row=1, col=1)
                fig_env.add_trace(go.Scatter(
                    x=t_if, y=np.clip(inst_f, 0, fs_adv/2), mode='lines',
                    name='f instantanée', line=dict(color='#00ff88', width=2)
                ), row=2, col=1)

                fig_env.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    height=520,
                    legend=dict(bgcolor='rgba(0,0,0,0.5)')
                )
                fig_env.update_xaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                                     title_text="Temps (s)")
                fig_env.update_yaxes(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff')
                st.plotly_chart(fig_env, use_container_width=True)

            elif analyse_adv == "Cepstrum":
                q, cep = engine_adv.cepstrum(s_adv)
                max_q = st.slider("Quefrency max (s)", 0.001, 0.5, 0.1)
                mask_c = q <= max_q

                fig_cep = go.Figure()
                fig_cep.add_trace(go.Scatter(
                    x=q[mask_c], y=cep[mask_c], mode='lines', name='Cepstrum',
                    line=dict(color='#00ccff', width=2)
                ))
                fig_cep.update_layout(
                    title="Cepstrum (analyse de periodicité)",
                    xaxis_title="Quefrency (s)", yaxis_title="Amplitude",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=400,
                )
                st.plotly_chart(fig_cep, use_container_width=True)

            elif analyse_adv == "Portrait de phase":
                decal = st.slider("Décalage τ (samples)", 1, 100, 10)
                s1 = s_adv[:-decal]
                s2 = s_adv[decal:]

                fig_pp = go.Figure()
                fig_pp.add_trace(go.Scatter(
                    x=s1, y=s2, mode='lines', name='Portrait de phase',
                    line=dict(color='#00ccff', width=1.5,
                              colorscale=None),
                    opacity=0.8
                ))
                fig_pp.update_layout(
                    title=f"Portrait de phase (τ={decal} samples)",
                    xaxis_title="s(t)", yaxis_title=f"s(t+τ)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(5,0,20,0.8)',
                    font=dict(color='#c0d0ff'),
                    xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff'),
                    height=450,
                )
                st.plotly_chart(fig_pp, use_container_width=True)

    # ============================================================
    # TAB 4 : MÉTRIQUES
    # ============================================================
    with tab4:
        st.markdown("### 📊 Métriques scientifiques complètes")

        if 's_sig' not in st.session_state:
            st.info("Générez d'abord un signal dans l'onglet Génération.")
        else:
            s_m = st.session_state['s_sig']
            engine_m = SignalEngine(fs=st.session_state['fs_sig'])
            metriques = engine_m.metriques(s_m)

            # Grille de métriques
            keys = list(metriques.keys())
            vals = list(metriques.values())
            cols = st.columns(4)
            for i, (k, v) in enumerate(zip(keys, vals)):
                with cols[i % 4]:
                    st.metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

            st.markdown("---")
            st.markdown("#### 📈 Distribution du signal")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=s_m, nbinsx=50,
                marker=dict(color='rgba(119,0,255,0.6)',
                           line=dict(color='rgba(0,204,255,0.8)', width=0.5)),
                name='Distribution'
            ))
            # KDE
            kde = stats.gaussian_kde(s_m)
            x_kde = np.linspace(s_m.min(), s_m.max(), 300)
            scale = len(s_m) * (s_m.max()-s_m.min()) / 50
            fig_hist.add_trace(go.Scatter(
                x=x_kde, y=kde(x_kde)*scale, mode='lines',
                name='KDE', line=dict(color='#00ccff', width=3)
            ))
            fig_hist.update_layout(
                title="Distribution & KDE",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(5,0,20,0.8)',
                font=dict(color='#c0d0ff'),
                xaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                          title='Amplitude'),
                yaxis=dict(gridcolor='rgba(100,0,255,0.2)', color='#c0d0ff',
                          title='Fréquence'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)'),
                height=350,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Export métriques
            df_met = pd.DataFrame({"Métrique": keys,
                                   "Valeur": [f"{v:.6f}" if isinstance(v, float) else str(v)
                                              for v in vals]})
            st.download_button("💾 Export métriques CSV",
                              df_met.to_csv(index=False).encode(),
                              "metriques_signal.csv", "text/csv")

    # ============================================================
    # TAB 5 : DIAGNOSTIC
    # ============================================================
    with tab5:
        st.markdown("### ⚗️ Diagnostic & Validation")

        if 's_sig' not in st.session_state:
            st.info("Générez d'abord un signal dans l'onglet Génération.")
        else:
            s_d = st.session_state['s_sig']
            fs_d = st.session_state['fs_sig']
            engine_d = SignalEngine(fs=fs_d)
            met = engine_d.metriques(s_d)

            # Tests
            diagnostics = []

            # Shannon
            nyq = fs_d / 2
            xf_d, mag_d, _, _ = engine_d.compute_fft(s_d)
            f_max = xf_d[np.argmax(mag_d)] if len(mag_d) > 0 else 0
            shannon_ok = fs_d >= 2 * f_max
            diagnostics.append({
                "Test": "Critère de Shannon",
                "Valeur": f"fe={fs_d} Hz, f_max≈{f_max:.1f} Hz",
                "Statut": "✅ OK" if shannon_ok else "❌ Aliasing!",
                "Action": "OK" if shannon_ok else f"Augmenter fe > {2*f_max:.0f} Hz"
            })

            # Clipping
            clip_ratio = np.sum(np.abs(s_d) >= 0.99*np.max(np.abs(s_d))) / len(s_d)
            diagnostics.append({
                "Test": "Saturation (clipping)",
                "Valeur": f"{clip_ratio*100:.2f}% des samples",
                "Statut": "✅ OK" if clip_ratio < 0.01 else "⚠️ Saturation",
                "Action": "OK" if clip_ratio < 0.01 else "Réduire l'amplitude"
            })

            # SNR
            snr = met["SNR estimé (dB)"]
            diagnostics.append({
                "Test": "SNR",
                "Valeur": f"{snr:.1f} dB",
                "Statut": "✅ Bon" if snr > 20 else "⚠️ Faible" if snr > 10 else "❌ Mauvais",
                "Action": "OK" if snr > 20 else "Réduire le bruit"
            })

            # Stationnarité (test ADF simplifié)
            N_d = len(s_d)
            half = N_d // 2
            std1 = np.std(s_d[:half])
            std2 = np.std(s_d[half:])
            non_stat = abs(std1 - std2) / (max(std1, std2) + 1e-12) > 0.2
            diagnostics.append({
                "Test": "Stationnarité approx.",
                "Valeur": f"|σ₁-σ₂|/max = {abs(std1-std2)/max(std1,std2,1e-12):.3f}",
                "Statut": "✅ Stationnaire" if not non_stat else "⚠️ Non-stationnaire",
                "Action": "OK" if not non_stat else "Envisager STFT ou ondelettes"
            })

            # Gaussianité (kurtosis)
            kurt = met["Kurtosis"]
            diagnostics.append({
                "Test": "Gaussianité (kurtosis)",
                "Valeur": f"κ = {kurt:.3f}",
                "Statut": "✅ Gaussien" if abs(kurt) < 1 else "⚠️ Non-gaussien",
                "Action": "OK" if abs(kurt) < 1 else "Signal non-gaussien détecté"
            })

            df_diag = pd.DataFrame(diagnostics)
            st.dataframe(df_diag, use_container_width=True)

            # Tableau d'erreurs DSP
            st.markdown("#### 🚨 Tableau de diagnostic DSP")
            erreurs = {
                "Problème": ["Aliasing", "Fuite spectrale", "Effet Gibbs", "Bruit plancher élevé",
                             "THD élevé", "Non-stationnarité"],
                "Cause": ["fe < 2·fmax", "Fenêtrage rectangulaire", "Signal tronqué",
                          "Bruit numérique/électronique", "Non-linéarités", "Signal variable"],
                "Symptôme": ["Repliement spectre", "Élargissement raies", "Oscillations",
                             "SNR faible", "Harmoniques parasites", "Variance spectrale"],
                "Solution": ["Augmenter fe", "Appliquer Hanning/Blackman", "Zero-padding",
                             "Filtrage ou moyennage", "Linéariser système", "STFT / Ondelettes"]
            }
            st.dataframe(pd.DataFrame(erreurs), use_container_width=True)

    # ============================================================
    # TAB 6 : THÉORIE
    # ============================================================
    with tab6:
        st.markdown("### 📖 Formulaire Scientifique DSP")

        for nom, formule in FORMULES_SIGNAL.items():
            st.markdown(f"**{nom}**")
            st.latex(formule)

        st.markdown("---")
        st.markdown("### 🔊 Types de filtres")
        df_filtres = pd.DataFrame([
            {"Filtre": k, "Caractéristique": v} for k, v in FILTRES_INFO.items()
        ])
        st.dataframe(df_filtres, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Références")
        refs = [
            "Proakis & Manolakis — *Digital Signal Processing* (Pearson, 2006)",
            "Oppenheim & Schafer — *Discrete-Time Signal Processing* (MIT, 2009)",
            "Cooley & Tukey — *FFT Algorithm* (Math. Comput., 1965)",
            "Welch — *Power Spectral Density Estimation* (IEEE Trans., 1967)",
        ]
        for r in refs:
            st.markdown(f"- {r}")