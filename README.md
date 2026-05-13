# Application de calcul scientifique PRO - LPN-MS

Application web Streamlit permettant d'effectuer des calculs scientifiques et des simulations en physique, mathématiques et automatique. LPN-MS (Laboratoire de Physique Numérique et Modélisation Scientifique) offre un environnement interactif pour l'exploration scientifique.

## Auteur et Droits

**© 2026 Papa Samba Fall - Tous droits réservés**

Ce projet est la propriété exclusive de Papa Samba Fall. Toute reproduction, modification ou utilisation sans autorisation préalable est strictement interdite.

## Installation

1. Créez un environnement virtuel:
```bash
python -m venv venv
```

2. Activez l'environnement:
```bash
# Sur Windows
venv\Scripts\activate

# Sur macOS / Linux
source venv/bin/activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancez l'application:
```bash
streamlit run main.py
```

L'application s'ouvrira ensuite dans votre navigateur à `http://localhost:8501/` par défaut.

> Si le port par défaut est occupé, utilisez un autre port, par exemple :
>
> ```bash
> streamlit run main.py --server.port=8511
> ```

## Fonctionnalités

- **Profil gaussien** : génération et visualisation de courbes gaussiennes
- **Simulation laser** : modélisation et visualisation de lasers
- **Pertes de cavité** : analyse des pertes optiques dans une cavité
- **Optimisation** : minimisation et optimisation de fonctions
- **Automatique** : contrôle et régulation de systèmes
- **Navier-Stokes** : simulation de fluides et d'écoulements
- **Data Science** : exploration et visualisation de données
- **Énergie** : calculs et représentations liés à l'énergie
- **Intégration** : calculs d'intégrales numériques
- **Interpolation** : interpolation et approximation de données
- **Équations différentielles** : résolution et visualisation
- **Signal** : traitement et analyse de signaux

## Structure du projet

```
APPLICATION_VScode/
├── automatique.py
├── cavity_losses.py
├── data_science.py
├── energy.py
├── equ_diff.py
├── gaussian.py
├── integration.py
├── interpolation.py
├── laser_simulation.py
├── main.css
├── main.py
├── navier_stokes.py
├── optimisation.py
├── requirements.txt
├── signal_tools.py
└── README.md
```

## Dépendances

Les packages requis sont listés dans `requirements.txt` :

- streamlit>=1.35.0
- numpy>=2.0.0
- scipy>=1.14.0
- pandas>=2.2.2
- plotly>=5.22.0
- matplotlib>=3.9.0
- openpyxl>=3.1.2
- scikit-learn>=1.5.0

## Commandes utiles

```bash
# Lancer l'application principale
streamlit run main.py

# Lancer sur un port spécifique
streamlit run main.py --server.port=8502

# Activer le mode debug de Streamlit
streamlit run main.py --logger.level=debug
```

## Déploiement Streamlit Community Cloud

1. Pousse ce dépôt sur GitHub.
2. Ouvre https://share.streamlit.io/ et connecte ton compte GitHub.
3. Crée une nouvelle application en sélectionnant ce dépôt.
4. Indique `main.py` comme fichier de démarrage.
5. Lance le déploiement.

> La configuration de Streamlit est déjà préparée dans `.streamlit/config.toml`.

## Remarques

- L'application est modulaire et chaque page est définie dans un module Python indépendant.
- Le fichier d'entrée principal est `main.py`.
- `streamlit_app.py` a été supprimé : il n'est plus utilisé par ce projet.
- Si Streamlit indique qu'un port est occupé, changez le port avec `--server.port`.
- Assurez-vous d'activer l'environnement virtuel avant d'installer les dépendances et de lancer l'application.
