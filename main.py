"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  APPLICATION DE CALCUL SCIENTIFIQUE PRO                     ║
║  Simulation Avancée • Laboratoire de Physique Numérique et Modélisation Scientifique ║
║  Version: 2.0.0 | Mode: Production                          ║
║                                                              ║
║  © 2026 Papa Samba Fall - Tous droits réservés              ║
║  Ce projet est la propriété exclusive de Papa Samba Fall.   ║
║  Toute reproduction, modification ou utilisation sans       ║
║  autorisation préalable est strictement interdite.          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Optional
import logging

# ─────────────────────────────────────────
#  CONFIGURATION LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
#  CONFIG PAGE - Streamlit
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Calcul Scientifique PRO - LPN-MS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "Application de simulation scientifique avancée v2.0.0 - LPN-MS"
    }
)

# ─────────────────────────────────────────
#  IMPORTS DES MODULES
# ─────────────────────────────────────────
import_error: Optional[Exception] = None
try:
    from gaussian import gaussian_page
    from laser_simulation import laser_page
    from cavity_losses import cavity_page
    from navier_stokes import navier_stokes_page
    from energy import energy_page
    from data_science import data_science_page
    from integration import integration_page
    from interpolation import interpolation_page
    from equ_diff import equ_diff_page
    from signal_tools import signal_page
    from optimisation import optimisation_page
    from automatique import automatique_page
    from electronique_analogique import electronique_analogique_page
    from mecanique_fluides import mecanique_fluides_page
    from mecanique_quantique import mecanique_quantique_page
    from physique_nucleaire import physique_nucleaire_page
    logger.info("✅ Tous les modules importés avec succès")
except Exception as e:
    import_error = e
    logger.error(f"❌ Erreur lors du chargement des modules: {e}")

if import_error is not None:
    st.error(f"Erreur lors du chargement des modules: {import_error}")

# ─────────────────────────────────────────
#  VÉRIFICATION DROITS D'AUTEUR - PROTECTION ANTI-SUPPRESSION
# ─────────────────────────────────────────
def verify_copyright():
    """
    Fonction de protection des droits d'auteur.
    Vérifie que l'empreinte de Papa Samba Fall est présente dans le code.
    Si elle est supprimée, l'application refuse de démarrer.
    """
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_copyrights = [
            "© 2026 Papa Samba Fall - Tous droits réservés",
            "Ce projet est la propriété exclusive de Papa Samba Fall",
            "Papa Samba Fall"
        ]
        
        for copyright_text in required_copyrights:
            if copyright_text not in content:
                st.error("🚫 ERREUR DE SÉCURITÉ : Droits d'auteur violés")
                st.error("Cette application est la propriété exclusive de Papa Samba Fall.")
                st.error("Toute modification non autorisée est interdite.")
                st.stop()
                return False
        
        logger.info("✅ Droits d'auteur vérifiés - Application autorisée")
        return True
        
    except Exception as e:
        st.error("🚫 ERREUR : Impossible de vérifier les droits d'auteur")
        st.stop()
        return False

# Vérifier les droits d'auteur avant de continuer
if not verify_copyright():
    st.stop()

# ─────────────────────────────────────────
#  CHARGEMENT DU CSS
# ─────────────────────────────────────────
try:
    with open("main.css", encoding="utf-8") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.info("✅ CSS chargé avec succès")
except FileNotFoundError:
    logger.warning("⚠️ Fichier main.css non trouvé")
    st.warning("CSS non disponible - interface en mode basique")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du CSS: {e}")

# ─────────────────────────────────────────
#  SESSION STATE — GESTION D'ÉTAT
# ─────────────────────────────────────────
# Initialiser l'état de session
if "page" not in st.session_state:
    st.session_state.page = "Accueil"
    logger.info("🔄 Session initialisée - Page: Accueil")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "language" not in st.session_state:
    st.session_state.language = "fr"

def navigate(page: str) -> None:
    """
    Fonction de navigation entre les pages

    Args:
        page (str): Nom de la page à afficher
    """
    st.session_state.page = page
    logger.info(f"🔀 Navigation vers: {page}")


def module_button(label: str, icon: str, page: str, key: str) -> bool:
    """
    Crée un bouton de module avec navigation
    
    Args:
        label (str): Libellé du bouton
        icon (str): Emoji/icône du bouton
        page (str): Page cible de la navigation
        key (str): Clé unique du bouton
    
    Returns:
        bool: True si le bouton a été cliqué
    """
    return st.button(
        f"{icon}  {label}",
        key=key,
        use_container_width=True,
        on_click=navigate,
        args=(page,)
    )

# ─────────────────────────────────────────
#  SIDEBAR — NAVIGATION PRINCIPALE
# ─────────────────────────────────────────
with st.sidebar:
    # Logo & En-tête
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">⚛️</div>
        <div class="sidebar-logo-text">
            <div class="logo-text">Calcul Scientifique</div>
            <div class="logo-pro">PRO</div>
        </div>
    </div>
    <div class="sidebar-chip">LPN-MS</div>
    <div class="sidebar-intro">Système de simulation avancé • Style cyberpunk • Interface premium</div>
    """, unsafe_allow_html=True)

    st.divider()

    # Bouton Accueil
    st.button(
        "🏠  Accueil",
        key="nav_accueil",
        use_container_width=True,
        on_click=navigate,
        args=("Accueil",)
    )

    st.divider()

    # ════════════════════════════════════════
    # 📊 MODULES PHYSIQUE
    # ════════════════════════════════════════
    st.markdown('<div class="sidebar-section-label">⚛️ Physique Numérique</div>', unsafe_allow_html=True)
    with st.expander("Modules Physique", expanded=True):
        modules_physique = {
            "Profil Gaussien":       ("📊", "Profil Gaussien"),
            "Laser de simulation":   ("💡", "Laser de simulation"),
            "Pertes de Cavité":      ("📉", "Pertes de Cavité"),
            "Navier-Stokes":         ("🌊", "Navier-Stokes"),
            "Énergie":               ("⚡", "Énergie"),
            "Mécanique des Fluides": ("💧", "Mécanique des Fluides"),
            "Mécanique Quantique":   ("⚛️", "Mécanique Quantique"),
            "Physique Nucléaire":    ("☢️", "Physique Nucléaire"),
            "Électronique Analogique": ("🔌", "Électronique Analogique"),
        }
        for label, (icon, page) in modules_physique.items():
            st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                on_click=navigate,
                args=(page,)
            )

    # ════════════════════════════════════════
    # 📐 MODULES MATHÉMATIQUES
    # ════════════════════════════════════════
    st.markdown('<div class="sidebar-section-label">📐 Mathématiques</div>', unsafe_allow_html=True)
    with st.expander("Modules Mathématiques", expanded=False):
        modules_maths = {
            "Intégration":   ("∫", "Intégration"),
            "Interpolation": ("📈", "Interpolation"),
            "Éq. Diff":      ("dy/dx", "Éq. Diff"),
        }
        for label, (icon, page) in modules_maths.items():
            st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                on_click=navigate,
                args=(page,)
            )

    # ════════════════════════════════════════
    # 📡 SIGNAL & AUTOMATIQUE
    # ════════════════════════════════════════
    st.markdown('<div class="sidebar-section-label">📡 Signal & Automatique</div>', unsafe_allow_html=True)
    with st.expander("Modules Signal & Automatique", expanded=False):
        modules_signal = {
            "Outils de Signal": ("📡", "Outils de Signal"),
            "Automatique":      ("⚙️", "Automatique"),
            "Optimisation":     ("🎯", "Optimisation"),
        }
        for label, (icon, page) in modules_signal.items():
            st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                on_click=navigate,
                args=(page,)
            )

    # ════════════════════════════════════════
    # 📊 MODULES DONNÉES
    # ════════════════════════════════════════
    st.markdown('<div class="sidebar-section-label">📊 Données & Export</div>', unsafe_allow_html=True)
    with st.expander("Modules Données", expanded=False):
        modules_data = {
            "Science des données":  ("📊", "Science des données"),
        }
        for label, (icon, page) in modules_data.items():
            st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                use_container_width=True,
                on_click=navigate,
                args=(page,)
            )

    st.divider()

    # Footer Sidebar
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; padding-top: 10px; border-top: 1px solid rgba(200,216,255,0.2);">
        <small style="color: rgba(160,180,255,0.5);">
            <strong>v2.0.0</strong> | Calcul Scientifique PRO<br>
            Mode Production • LPN-MS<br>
            <em>© 2026 Papa Samba Fall - Tous droits réservés</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FONCTIONS PAR MODULE
# ─────────────────────────────────────────

def page_accueil():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-topbar">
            <span class="hero-tag">Système LPN-MS</span>
            <div class="hero-badges">
                <span>Mode PRO</span>
            </div>
        </div>
        <div class="hero-title-block">
            <h1>Application de <span>Calcul Scientifique PRO - LPN-MS</span></h1>
            <p>Explorez l’univers caché derrière chaque équation avec LPN-MS. Visualisez des simulations néon, des courbes dynamiques et une interface laboratoire avancée.</p>
        </div>
        <div class="hero-actions">
            <div class="hero-action-pill">✔ Système stabilisé</div>
            <div class="hero-action-pill">✔ Scénarios préchargés</div>
            <div class="hero-action-pill">✔ Visualisation instantanée</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.button(
        "▶  Démarrer la simulation",
        key="cta_start",
        use_container_width=True,
        on_click=navigate,
        args=("Profil Gaussien",)
    )

    st.markdown('<div class="hero-subline">Fond spatial animé • LPN-MS</div>', unsafe_allow_html=True)

    x = np.linspace(-4, 4, 60)
    y = np.linspace(-4, 4, 60)
    X, Y = np.meshgrid(x, y)
    Z = (np.exp(-0.4*(X**2 + Y**2)) +
         0.5*np.exp(-1.5*((X-2)**2 + (Y-1)**2)) +
         0.4*np.exp(-1.5*((X+2)**2 + (Y+1)**2)) +
         0.35*np.exp(-1.5*((X-1)**2 + (Y+2)**2)))

    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y,
        colorscale=[
            [0.0,  "rgba(14,0,40,0.9)"],
            [0.25, "rgba(60,0,150,0.9)"],
            [0.55, "rgba(0,100,200,0.96)"],
            [0.8,  "rgba(0,220,255,0.98)"],
            [1.0,  "rgba(220,250,255,1)"],
        ],
        showscale=False,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.6, roughness=0.22),
        lightposition=dict(x=3, y=3, z=4),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#5ee0ff", project_z=True)
        ),
    )])

    fig.update_layout(
        height=390,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(5,0,20,0.0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="hero-chart-note">Hologramme scientifique 3D en temps réel • Effet LPN-MS</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Modules d’exploration</div>', unsafe_allow_html=True)

    modules_home = [
        ("Profil Gaussien", "📊", "Profil Gaussien"),
        ("Laser de simulation", "💡", "Laser de simulation"),
        ("Pertes de Cavité", "📉", "Pertes de Cavité"),
        ("Navier-Stokes", "🌊", "Navier-Stokes"),
        ("Énergie", "⚡", "Énergie"),
        ("Outils de Signal", "📡", "Outils de Signal"),
        ("Mécanique des Fluides", "💧", "Mécanique des Fluides"),
        ("Mécanique Quantique", "⚛️", "Mécanique Quantique"),
        ("Physique Nucléaire", "☢️", "Physique Nucléaire"),
    ]

    for row in [modules_home[i:i+3] for i in range(0, len(modules_home), 3)]:
        cols = st.columns(3, gap="large")
        for col, (label, icon, page) in zip(cols, row):
            with col:
                st.markdown(f"""
                    <div class="module-card module-card-home">
                        <span class="module-card-icon">{icon}</span>
                        <div class="module-card-label">{label}</div>
                    </div>
                """, unsafe_allow_html=True)
                module_button(label, icon, page, key=f"home_{label}")

    st.markdown("<div class=\"section-footer\">Cliquez sur un module pour explorer les simulations et analyses.</div>", unsafe_allow_html=True)

    # footer rendered globally after route selection


# ─────────────────────────────────────────
def page_placeholder(titre: str, icone: str, message: Optional[str] = None):
    st.title(f"{icone} {titre}")
    st.markdown("---")
    if message:
        st.warning(message)
    st.markdown(f"""
    <div class="panel-card" style="text-align:center; padding:60px 20px;">
        <div style="font-size:4rem; margin-bottom:16px; opacity:0.5;">{icone}</div>
        <h2 style="color:#c8d8ff; margin-bottom:10px;">Module en développement</h2>
        <p style="color:rgba(160,180,255,0.6);">Ce module sera disponible prochainement.</p>
    </div>
    """, unsafe_allow_html=True)


def make_unavailable_page(titre: str, icone: str, raison: str):
    def unavailable():
        page_placeholder(
            titre,
            icone,
            message=f"Ce module est temporairement indisponible : {raison}"
        )
    return unavailable


def page_cavity_advanced():
    """Page pour le module avancé de cavités optiques"""
    page_placeholder("Cavités Optiques Avancées", "🔬")


def page_signal_advanced():
    """Page pour le module avancé de traitement du signal"""
    page_placeholder("Outils de Signal Avancés", "🔬")


def page_data_science_advanced():
    """Page pour le module avancé de science des données"""
    page_placeholder("Science des données avancée", "🔬")


def page_not_found():
    page_placeholder(
        "Page introuvable",
        "❓",
        message="La page demandée n'a pas pu être affichée. Vérifiez le menu de navigation."
    )


# Fallback automatique si un module n'a pas pu être importé
fallback_pages = {
    "gaussian_page": ("Profil Gaussien", "📊"),
    "laser_page": ("Laser de simulation", "💡"),
    "cavity_page": ("Pertes de Cavité", "📉"),
    "navier_stokes_page": ("Navier-Stokes", "🌊"),
    "energy_page": ("Énergie", "⚡"),
    "data_science_page": ("Science des données", "📊"),
    "integration_page": ("Intégration", "∫"),
    "interpolation_page": ("Interpolation", "📈"),
    "equ_diff_page": ("Éq. Diff", "dy/dx"),
    "signal_page": ("Outils de Signal", "📡"),
    "optimisation_page": ("Optimisation", "🎯"),
    "automatique_page": ("Automatique", "⚙️"),
    "electronique_analogique_page": ("Électronique Analogique", "🔌"),
    "mecanique_fluides_page": ("Mécanique des Fluides", "💧"),
    "mecanique_quantique_page": ("Mécanique Quantique", "⚛️"),
    "physique_nucleaire_page": ("Physique Nucléaire", "☢️"),
}
for attr, (titre, icone) in fallback_pages.items():
    if attr not in globals():
        globals()[attr] = make_unavailable_page(
            titre,
            icone,
            raison="Module non chargé ou erreur d'import"
        )


# ─────────────────────────────────────────
#  ROUTEUR PRINCIPAL
# ─────────────────────────────────────────
page = st.session_state.page

routes = {
    "Accueil":              page_accueil,
    "Profil Gaussien":      gaussian_page,
    "Laser de simulation":  laser_page,
    "Pertes de Cavité":     cavity_page,
    "Navier-Stokes":        navier_stokes_page,
    "Énergie":              energy_page,
    "Intégration":          integration_page,
    "Interpolation":        interpolation_page,
    "Éq. Diff":             equ_diff_page,
    "Outils de Signal":     signal_page,
    "Automatique":          automatique_page,
    "Optimisation":         optimisation_page,
    "Science des données":  data_science_page,
    "Électronique Analogique": electronique_analogique_page,
    "Mécanique des Fluides": mecanique_fluides_page,
    "Mécanique Quantique": mecanique_quantique_page,
    "Physique Nucléaire": physique_nucleaire_page,
}

routes.get(page, page_not_found)()
