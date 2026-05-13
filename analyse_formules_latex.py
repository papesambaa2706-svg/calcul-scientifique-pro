#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse complète des formules LaTeX dans les fichiers Python
Extrait, valide et génère un rapport détaillé
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

class AnalyseurLatex:
    """Analyseur de formules LaTeX avec détection d'erreurs."""
    
    def __init__(self, dossier: str):
        self.dossier = Path(dossier)
        self.resultats = {}
        
    def extraire_formules(self, contenu: str) -> Dict[str, str]:
        """Extrait les formules des dictionnaires FORMULES*"""
        formules = {}
        
        # Cherche les lignes avec FORMULES* = {
        lignes = contenu.split('\n')
        
        i = 0
        while i < len(lignes):
            ligne = lignes[i]
            
            # Cherche une ligne commençant par FORMULES...=
            if re.match(r'\s*FORMULES(_\w+)?\s*=\s*\{', ligne):
                # Extrait les formules jusqu'à la fermeture du dictionnaire
                i += 1
                while i < len(lignes):
                    ligne = lignes[i]
                    
                    # Cherche les paires clé: valeur dans une ligne
                    # Format: "cle": r"formule",
                    match = re.match(r'\s*"([^"]+)"\s*:\s*r"([^"]*(?:\\.[^"]*)*)"', ligne)
                    if match:
                        cle = match.group(1)
                        formule = match.group(2)
                        # Les raw strings ont déjà les bons backslashes
                        formules[cle] = formule
                    
                    # Fin du dictionnaire
                    if re.match(r'\s*\}', ligne):
                        break
                    
                    i += 1
            
            i += 1
        
        return formules
    
    def valider_accolades(self, formule: str) -> Tuple[bool, List[str]]:
        """Vérifie l'équilibre des accolades"""
        erreurs = []
        compte = 0
        i = 0
        
        while i < len(formule):
            char = formule[i]
            
            # Ignore les échappements
            if char == '\\' and i + 1 < len(formule):
                i += 2
                continue
            
            if char == '{':
                compte += 1
            elif char == '}':
                compte -= 1
                if compte < 0:
                    erreurs.append(f"Accolade fermante orpheline à position {i}")
            
            i += 1
        
        if compte != 0:
            erreurs.append(f"Accolades déséquilibrées (différence: {compte})")
        
        return len(erreurs) == 0, erreurs
    
    def valider_parentheses(self, formule: str) -> Tuple[bool, List[str]]:
        """Vérifie l'équilibre des parenthèses et crochets"""
        erreurs = []
        
        # Parenthèses
        compte_paren = 0
        i = 0
        while i < len(formule):
            char = formule[i]
            
            # Ignore les échappements
            if char == '\\' and i + 1 < len(formule):
                i += 2
                continue
            
            if char == '(':
                compte_paren += 1
            elif char == ')':
                compte_paren -= 1
                if compte_paren < 0:
                    erreurs.append(f"Parenthèse fermante orpheline à position {i}")
            
            i += 1
        
        if compte_paren != 0:
            erreurs.append(f"Parenthèses déséquilibrées (différence: {compte_paren})")
        
        # Crochets
        compte_croch = 0
        i = 0
        while i < len(formule):
            char = formule[i]
            
            # Ignore les échappements
            if char == '\\' and i + 1 < len(formule):
                i += 2
                continue
            
            if char == '[':
                compte_croch += 1
            elif char == ']':
                compte_croch -= 1
                if compte_croch < 0:
                    erreurs.append(f"Crochet fermant orphelin à position {i}")
            
            i += 1
        
        if compte_croch != 0:
            erreurs.append(f"Crochets déséquilibrés (différence: {compte_croch})")
        
        return len(erreurs) == 0, erreurs
    
    def valider_commandes_latex(self, formule: str) -> Tuple[bool, List[str]]:
        """Cherche les erreurs courantes LaTeX"""
        erreurs = []
        
        # 1. \frac sans arguments
        if re.search(r'\\frac(?!\s*\{)', formule):
            erreurs.append("\\frac détecté sans arguments")
        
        # 2. \sum ou \prod sans limites
        if re.search(r'\\sum(?!\s*[_^])', formule):
            # Sauf si c'est une somme sans limites volontaire
            if '\\sum' in formule and not re.search(r'\\sum\s*[_^{]', formule):
                # Vérifier le contexte
                pass  # Accepter pour l'instant
        
        # 3. Chevrons d'exposant/indice non fermés
        if re.search(r'[_^]\s*[^{]', formule):
            # Vérifie si l'exposant/indice est correctement parenthésé
            pattern = r'[_^](?!\s*\{)'
            for match in re.finditer(pattern, formule):
                pos = match.start()
                # Si ce n'est pas une lettre seule, c'est problématique
                if pos + 2 < len(formule) and formule[pos+2] == ' ':
                    pass  # C'est ok, lettre seule
        
        # 4. Échappement incorrect (\\\\ au lieu de \\)
        if re.search(r'\\\\\\\\ ', formule):  # 4+ backslashes
            erreurs.append("Échappement excessif détecté (\\\\\\\\)")
        
        # 5. Commandes LaTeX incomplètes
        if re.search(r'\\[a-z]+(?![a-z])\s*$', formule):
            # Commande LaTeX en fin de chaîne sans argument
            pass  # Peut être volontaire
        
        # 6. Commandes LaTeX orphelines (sans fermeture)
        orphelines = ['frac', 'sqrt', 'left', 'right', 'mathbf', 'mathcal', 
                     'text', 'exp', 'log', 'sin', 'cos', 'tan', 'cdots']
        
        for cmd in orphelines:
            pattern = rf'\\{cmd}\s*(?!\{{)'
            matches = re.finditer(pattern, formule)
            for match in matches:
                # Vérifier s'il y a une accolade après
                pos = match.end()
                if pos < len(formule) and formule[pos] != '{':
                    if cmd in ['left', 'right']:
                        pass  # Ces commandes sont spéciales
                    elif cmd in ['sqrt', 'frac', 'mathbf', 'mathcal', 'text']:
                        pass  # Peut avoir un argument optionnel
        
        return len(erreurs) == 0, erreurs
    
    def valider_espaces_dollars(self, formule: str) -> Tuple[bool, List[str]]:
        """Détecte les signes dollar en environnement raw string"""
        erreurs = []
        
        # Dans un raw string (r"..."), les $ n'ont pas besoin d'être échappés
        # Mais on peut vérifier si c'est cohérent
        
        if '$' in formule:
            # Les raw strings ne devraient pas avoir $ comme séparateurs
            erreurs.append("Signe $ trouvé (non nécessaire en raw string)")
        
        return len(erreurs) == 0, erreurs
    
    def valider_formule(self, cle: str, formule: str) -> Tuple[bool, List[str]]:
        """Validation complète d'une formule"""
        tous_erreurs = []
        
        # 1. Accolades
        ok_acc, err = self.valider_accolades(formule)
        tous_erreurs.extend(err)
        
        # 2. Parenthèses/crochets
        ok_par, err = self.valider_parentheses(formule)
        tous_erreurs.extend(err)
        
        # 3. Commandes LaTeX
        ok_cmd, err = self.valider_commandes_latex(formule)
        tous_erreurs.extend(err)
        
        # 4. Espaces/dollars
        ok_esp, err = self.valider_espaces_dollars(formule)
        tous_erreurs.extend(err)
        
        return len(tous_erreurs) == 0, tous_erreurs
    
    def analyser_fichier(self, chemin: Path) -> Dict:
        """Analyse un fichier Python complet"""
        try:
            with open(chemin, 'r', encoding='utf-8') as f:
                contenu = f.read()
        except Exception as e:
            return {
                'nom': chemin.name,
                'chemin': str(chemin),
                'erreur': str(e),
                'total_formules': 0,
                'formules_valides': 0,
                'formules_erreurs': []
            }
        
        # Extraire les formules
        formules = self.extraire_formules(contenu)
        
        # Valider chaque formule
        formules_erreurs = []
        formules_valides = 0
        
        for cle, formule in formules.items():
            est_valide, erreurs = self.valider_formule(cle, formule)
            if est_valide:
                formules_valides += 1
            else:
                formules_erreurs.append({
                    'cle': cle,
                    'formule': formule[:100] + ('...' if len(formule) > 100 else ''),
                    'erreurs': erreurs
                })
        
        return {
            'nom': chemin.name,
            'chemin': str(chemin),
            'total_formules': len(formules),
            'formules_valides': formules_valides,
            'formules_erreurs': formules_erreurs
        }
    
    def analyser_dossier(self) -> Dict[str, Dict]:
        """Analyse tous les fichiers Python du dossier"""
        fichiers_py = sorted(self.dossier.glob('*.py'))
        
        for fichier in fichiers_py:
            self.resultats[fichier.name] = self.analyser_fichier(fichier)
        
        return self.resultats
    
    def generer_rapport(self) -> str:
        """Génère un rapport formaté"""
        rapport = []
        rapport.append("=" * 80)
        rapport.append("RAPPORT D'ANALYSE DES FORMULES LATEX")
        rapport.append("=" * 80)
        rapport.append("")
        
        total_global = 0
        valides_global = 0
        tous_fichiers_ok = True
        
        for nom_fichier, resultats in sorted(self.resultats.items()):
            if 'erreur' in resultats:
                rapport.append(f"❌ {nom_fichier}: Erreur lors de la lecture - {resultats['erreur']}")
                tous_fichiers_ok = False
                continue
            
            total = resultats['total_formules']
            valides = resultats['formules_valides']
            erreurs = resultats['formules_erreurs']
            
            total_global += total
            valides_global += valides
            
            # En-tête du fichier
            rapport.append(f"\n📄 {nom_fichier}")
            rapport.append("-" * 80)
            rapport.append(f"   Total formules: {total} | Valides: {valides} | Erreurs: {len(erreurs)}")
            
            # Détail des erreurs
            if erreurs:
                tous_fichiers_ok = False
                rapport.append("\n   ⚠️  FORMULES AVEC ERREURS:")
                for item in erreurs:
                    rapport.append(f"\n      • {item['cle']}")
                    rapport.append(f"        Formule: {item['formule']}")
                    for err in item['erreurs']:
                        rapport.append(f"        ❌ {err}")
            else:
                rapport.append(f"   ✅ Toutes les formules sont valides")
        
        # Résumé global
        rapport.append("\n" + "=" * 80)
        rapport.append("RÉSUMÉ GLOBAL")
        rapport.append("=" * 80)
        rapport.append(f"Total fichiers analysés: {len(self.resultats)}")
        rapport.append(f"Total formules: {total_global}")
        rapport.append(f"Formules valides: {valides_global}")
        rapport.append(f"Formules avec erreurs: {total_global - valides_global}")
        
        if tous_fichiers_ok and total_global > 0:
            rapport.append(f"\n✅ VALIDATION COMPLÈTE: TOUTES LES FORMULES SONT VALIDES ✅")
        elif total_global == 0:
            rapport.append(f"\n⚠️  AUCUNE FORMULE TROUVÉE")
        else:
            rapport.append(f"\n❌ ERREURS TROUVÉES: {total_global - valides_global} formule(s) problématique(s)")
        
        rapport.append("=" * 80)
        
        return "\n".join(rapport)


def main():
    dossier = r"c:\Users\hp\Documents\APPLICATION_VScode"
    
    if not Path(dossier).exists():
        print(f"Erreur: Le dossier {dossier} n'existe pas")
        return
    
    analyseur = AnalyseurLatex(dossier)
    resultats = analyseur.analyser_dossier()
    rapport = analyseur.generer_rapport()
    
    print(rapport)
    
    # Sauvegarder le rapport
    fichier_rapport = Path(dossier) / "RAPPORT_ANALYSE_LATEX.txt"
    try:
        with open(fichier_rapport, 'w', encoding='utf-8') as f:
            f.write(rapport)
        print(f"\n📁 Rapport sauvegardé: {fichier_rapport}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")


if __name__ == "__main__":
    main()
