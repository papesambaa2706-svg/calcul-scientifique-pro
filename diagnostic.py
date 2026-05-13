#!/usr/bin/env python3
"""
Script de diagnostic pour les nouveaux modules
"""
import sys
import os
import traceback

# Ajouter le répertoire courant au path
sys.path.insert(0, os.getcwd())

# Mock streamlit pour éviter les erreurs d'import
class MockST:
    @staticmethod
    def cache_data(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

    @staticmethod
    def markdown(text): pass

    @staticmethod
    def tabs(tabs_list):
        return [MockTab() for _ in tabs_list]

    @staticmethod
    def columns(cols):
        return [MockCol() for _ in cols]

    @staticmethod
    def selectbox(label, options, **kwargs):
        return options[0] if options else None

    @staticmethod
    def slider(label, min_val, max_val, value, **kwargs):
        return value

    @staticmethod
    def radio(label, options, **kwargs):
        return options[0] if options else None

    @staticmethod
    def metric(label, value): pass

    @staticmethod
    def plotly_chart(fig, **kwargs): pass

    @staticmethod
    def expander(label, **kwargs):
        return MockExpander()

    @staticmethod
    def checkbox(label, **kwargs):
        return False

    @staticmethod
    def number_input(label, **kwargs):
        return kwargs.get('value', 0)

    @staticmethod
    def text_input(label, **kwargs):
        return ""

    @staticmethod
    def button(label, **kwargs):
        return False

    @staticmethod
    def warning(text): print(f"WARNING: {text}")

    @staticmethod
    def error(text): print(f"ERROR: {text}")

    @staticmethod
    def info(text): print(f"INFO: {text}")

    @staticmethod
    def dataframe(df, **kwargs): pass

    @staticmethod
    def data_editor(df, **kwargs): pass

class MockTab:
    def __enter__(self): return self
    def __exit__(self, *args): pass

class MockCol:
    def __enter__(self): return self
    def __exit__(self, *args): pass

class MockExpander:
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Remplacer streamlit
sys.modules['streamlit'] = MockST()

# Importer numpy et autres dépendances
try:
    import numpy as np
    import pandas as pd
    from scipy import special, integrate, optimize, linalg
    print("✅ Dépendances importées")
except Exception as e:
    print(f"❌ Erreur dépendances: {e}")
    sys.exit(1)

# Tester les modules un par un
modules_to_test = [
    'electronique_analogique',
    'mecanique_fluides',
    'mecanique_quantique',
    'physique_nucleaire'
]

for module_name in modules_to_test:
    print(f"\n🔍 Test du module: {module_name}")
    try:
        # Importer le module
        module = __import__(module_name)
        print(f"✅ Module {module_name} importé")

        # Tester la fonction page
        page_func_name = f"{module_name}_page"
        if hasattr(module, page_func_name):
            page_func = getattr(module, page_func_name)
            print(f"✅ Fonction {page_func_name} trouvée")

            # Tester l'exécution (avec timeout)
            try:
                page_func()
                print(f"✅ Fonction {page_func_name} exécutée sans erreur")
            except Exception as e:
                print(f"❌ Erreur lors de l'exécution de {page_func_name}: {e}")
                traceback.print_exc()
        else:
            print(f"❌ Fonction {page_func_name} non trouvée")

    except Exception as e:
        print(f"❌ Erreur lors de l'import de {module_name}: {e}")
        traceback.print_exc()

print("\n🎯 Diagnostic terminé")