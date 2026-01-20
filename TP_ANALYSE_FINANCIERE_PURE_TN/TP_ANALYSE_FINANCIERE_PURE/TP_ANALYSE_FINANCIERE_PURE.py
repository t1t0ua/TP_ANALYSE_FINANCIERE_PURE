import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

############
# PARTIE 2: EXPLORATION DES DONNÉES
############

print("+"*80)
print("ÉTAPE 2 : EXPLORATION DES DONNÉES")
print("+"*80)

# Récupération des données historiques de Microsoft sur 15 ans
data = yf.download("MSFT", start="2010-01-01", end="2025-01-01")

# Erreur en cas de non-chargement du fichier
if data is None or data.empty:
    print("Erreur de chargement")
    exit()

# Aplatir la structure multi-index des colonnes
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Trouver le nom de la colonne du prix ajusté
prix_col = None
for col in data.columns:
    if 'adj' in col.lower() and 'close' in col.lower():
        prix_col = col
        break

if prix_col is None:
    prix_col = 'Close'

print(f"\nColonne utilisée pour les prix : {prix_col}\n")

# Afficher les 10 premières lignes
print("10 premières lignes :")
print(data.head(10))
print()

# Afficher les 5 dernières lignes
print("5 dernières lignes :")
print(data.tail(5))
print()

# Afficher le nombre de jours de trading
print(f"Nombre de jours de trading : {len(data)}\n")

# Afficher les types de données de chaque colonne
print("Types de données présents dans le DataFrame :")
for col in data.columns:
    print(f"   {col:12} : {type(data[col].iloc[0]).__name__}")
print()

# Afficher la mémoire utilisée par le DataFrame
print(f"Taille du DataFrame : {data.size} éléments")
print(f"Mémoire utilisée : {data.memory_usage(deep=True).sum() / 1024:.2f} KB\n")

# Recherche de périodes manquantes
data_temp = data.copy()
data_temp['date_diff'] = data_temp.index.to_series().diff().dt.days

ecarts_stats = data_temp['date_diff'].value_counts().sort_index()
print("Distribution des écarts entre dates consécutives :")
for jours, count in ecarts_stats.head(10).items():
    if pd.notna(jours):
        print(f"   {int(jours)} jour(s) d'écart : {count:,} occurrences")

gaps_weekend = data_temp[data_temp['date_diff'] == 3]
gaps_anormaux = data_temp[data_temp['date_diff'] > 4]

print(f"\n   Weekends normaux (3 jours)  : {len(gaps_weekend):,} occurrences")
print(f"   Gaps anormaux (>4 jours)    : {len(gaps_anormaux):,} occurrences")

if len(gaps_anormaux) > 0:
    print("\nDétail des gaps anormaux :")
    for idx in gaps_anormaux.head(10).index:
        nb_jours_gap = int(gaps_anormaux.loc[idx, 'date_diff'])
        print(f"   {idx.strftime('%d/%m/%Y')} : gap de {nb_jours_gap} jours")
print()

# Afficher les volumes d'échange
print("Statistiques descriptives du volume :")
print(f"   Minimum          : {data['Volume'].min():>15,.0f} actions")
print(f"   25e percentile   : {data['Volume'].quantile(0.25):>15,.0f} actions")
print(f"   Médiane (50%)    : {data['Volume'].median():>15,.0f} actions")
print(f"   Moyenne          : {data['Volume'].mean():>15,.0f} actions")
print(f"   75e percentile   : {data['Volume'].quantile(0.75):>15,.0f} actions")
print(f"   Maximum          : {data['Volume'].max():>15,.0f} actions")
print(f"   Écart-type       : {data['Volume'].std():>15,.0f} actions")
print()

coef_variation = (data['Volume'].std() / data['Volume'].mean()) * 100
print(f"Coefficient de variation : {coef_variation:.1f}%\n")

# Top 5 des jours avec le plus fort volume
print("Top 5 des jours avec le volume le plus élevé :")
top_volume = data.nlargest(5, 'Volume')
for idx, row in top_volume.iterrows():
    print(f"   {idx.strftime('%d/%m/%Y')} : {row['Volume']:>15,.0f} actions (prix: ${row[prix_col]:.2f})")
print()

# Afficher la tendance visuelle générale
date_debut = data.index.min()
date_fin = data.index.max()
nb_annees = (date_fin - date_debut).days / 365.25

prix_initial = data[prix_col].iloc[0]
prix_final = data[prix_col].iloc[-1]
rendement_total = ((prix_final - prix_initial) / prix_initial) * 100

prix_min = data[prix_col].min()
prix_max = data[prix_col].max()
date_min = data[prix_col].idxmin()
date_max = data[prix_col].idxmax()

print("Évolution des prix :")
print(f"   Prix initial ({date_debut.strftime('%d/%m/%Y')})    : ${prix_initial:>8.2f}")
print(f"   Prix final ({date_fin.strftime('%d/%m/%Y')})      : ${prix_final:>8.2f}")
print(f"   Prix minimum (historique)            : ${prix_min:>8.2f} le {date_min.strftime('%d/%m/%Y')}")
print(f"   Prix maximum (historique)            : ${prix_max:>8.2f} le {date_max.strftime('%d/%m/%Y')}")
print()

print("Performance brute :")
print(f"   Rendement total sur {nb_annees:.1f} ans : {rendement_total:+,.2f}%")
print(f"   Multiplication du capital           : x{(prix_final/prix_initial):.2f}")
print()

cagr = ((prix_final / prix_initial) ** (1 / nb_annees) - 1) * 100
print(f"   CAGR (rendement annualisé)          : {cagr:+.2f}% par an")
print()

if rendement_total > 0:
    print("Tendance générale : HAUSSIÈRE")
else:
    print("Tendance générale : BAISSIÈRE")

print("\n" + "="*80)

############
# PARTIE 3: NETTOYAGE ET PRÉPARATION DES DONNÉES
############

print("\n" + "$"*80)
print("ÉTAPE 3 : NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("$"*80)

# 3.1 ÉVALUATION DE LA QUALITÉ
print("\n--- 3.1 ÉVALUATION DE LA QUALITÉ ---\n")


# Vérification des valeurs manquantes
valeurs_manquantes = data.isnull().sum()
print("Nombre de valeurs manquantes par colonne :")
for col in data.columns:
    nb_nan = valeurs_manquantes[col]
    print(f"   {col:12} : {nb_nan} valeurs manquantes")

print()
total_nan = valeurs_manquantes.sum()
print(f"Total de valeurs manquantes : {total_nan}")

if total_nan > 0:
    pourcentage_nan = (total_nan / data.size) * 100
    print(f"Pourcentage de données manquantes : {pourcentage_nan:.2f}%")
    print("\nLignes contenant des valeurs manquantes :")
    lignes_nan = data[data.isnull().any(axis=1)]
    print(lignes_nan)
else:
    print("Aucune valeur manquante détectée dans le dataset")

print()

# Vérification de l'unicité des lignes
nb_total = len(data)
nb_unique = data.index.nunique()
nb_doublons = nb_total - nb_unique

print("Vérification de l'unicité des dates :")
print(f"   Nombre total de lignes     : {nb_total}")
print(f"   Nombre de dates uniques    : {nb_unique}")
print(f"   Nombre de doublons         : {nb_doublons}")

if nb_doublons > 0:
    print("\nDates en double détectées :")
    doublons = data[data.index.duplicated(keep=False)]
    print(doublons)
else:
    print("Aucun doublon détecté")

print()

# Vérification de la cohérence des données
print("Vérification de la cohérence des données :\n")

anomalies_totales = 0

# Test 1: High >= Low (corrigé)
anomalies = data[data['High'] < data['Low']]
if len(anomalies) == 0:
    print("   High >= Low : Toutes les lignes sont cohérentes")
else:
    print(f"   High >= Low : {len(anomalies)} anomalie(s) détectée(s)")
    print(anomalies[['Open', 'High', 'Low', 'Close']])
    anomalies_totales += len(anomalies)

# Test 2: High >= Close >= Low (corrigé)
anomalies = data[(data['Close'] > data['High']) | (data['Close'] < data['Low'])]
if len(anomalies) == 0:
    print("   High >= Close >= Low : Toutes les lignes sont cohérentes")
else:
    print(f"   High >= Close >= Low : {len(anomalies)} anomalie(s) détectée(s)")
    print(anomalies[['Open', 'High', 'Low', 'Close']])
    anomalies_totales += len(anomalies)

# Test 3: High >= Open >= Low (corrigé)
anomalies = data[(data['Open'] > data['High']) | (data['Open'] < data['Low'])]
if len(anomalies) == 0:
    print("   High >= Open >= Low : Toutes les lignes sont cohérentes")
else:
    print(f"   High >= Open >= Low : {len(anomalies)} anomalie(s) détectée(s)")
    print(anomalies[['Open', 'High', 'Low', 'Close']])
    anomalies_totales += len(anomalies)

# Test 4: Volume > 0
anomalies = data[data['Volume'] <= 0]
if len(anomalies) == 0:
    print("   Volume > 0 : Toutes les lignes sont cohérentes")
else:
    print(f"   Volume > 0 : {len(anomalies)} anomalie(s) détectée(s)")
    print(anomalies[['Volume']])
    anomalies_totales += len(anomalies)

print(f"\nTotal d'anomalies détectées : {anomalies_totales}\n")

# 3.2 TRAITEMENT DES PROBLÈMES DE QUALITÉ
print("\n--- 3.2 TRAITEMENT DES PROBLÈMES ---\n")

# Traitement des valeurs manquantes
if total_nan > 0:
    print("Traitement des valeurs manquantes...\n")
    
    # Stratégie : interpolation linéaire pour Close, forward-fill pour les autres
    data_avant = data.copy()
    
    # Pour Close : interpolation linéaire
    if data['Close'].isnull().sum() > 0:
        data['Close'] = data['Close'].interpolate(method='linear')
        print(f"   Close : {data_avant['Close'].isnull().sum()} NaN -> interpolation linéaire")
    
    # Pour les autres colonnes : forward-fill
    for col in ['Open', 'High', 'Low', 'Volume']:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(method='ffill')
            print(f"   {col} : {data_avant[col].isnull().sum()} NaN -> forward-fill")
    
    print(f"\n   Résultat : {data.isnull().sum().sum()} valeurs manquantes restantes")
else:
    print("Aucune valeur manquante à traiter\n")

# Suppression des doublons
if nb_doublons > 0:
    print("Suppression des doublons...\n")
    data = data[~data.index.duplicated(keep='first')]
    print(f"   {nb_doublons} doublon(s) supprimé(s)\n")

print()

# 3.3 TRANSFORMATION DES DONNÉES
print("\n--- 3.3 TRANSFORMATION DES DONNÉES ---\n")

# Vérification du tri chronologique
if not data.index.is_monotonic_increasing:
    print("Tri chronologique des données...")
    data = data.sort_index()
    print("   Données triées par ordre chronologique\n")
else:
    print("Données déjà triées par ordre chronologique\n")

# 3.4 CRÉATION DE VARIABLES DÉRIVÉES
print("\n--- 3.4 CRÉATION DE VARIABLES DÉRIVÉES ---\n")

# Variables de rendement
print("Calcul des variables de rendement...")
data['Rendement_Quotidien'] = data['Close'].pct_change() * 100
data['Rendement_Cumule'] = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
data['Log_Rendement'] = np.log(data['Close'] / data['Close'].shift(1))
print("   Rendement quotidien (%)")
print("   Rendement cumulé (%)")
print("   Log rendement\n")

# Variables temporelles
print("Extraction des variables temporelles...")
data['Annee'] = data.index.year
data['Mois'] = data.index.month
data['Jour_Semaine'] = data.index.dayofweek  # 0=Lundi, 6=Dimanche
data['Jour_Semaine_Nom'] = data.index.day_name()
data['Trimestre'] = data.index.quarter
print("   Année")
print("   Mois")
print("   Jour de la semaine")
print("   Trimestre\n")

# Variables de volatilité
print("Calcul des variables de volatilité...")
data['Volatilite_30j'] = data['Rendement_Quotidien'].rolling(window=30).std()
data['Volatilite_90j'] = data['Rendement_Quotidien'].rolling(window=90).std()
data['Range_Quotidien'] = ((data['High'] - data['Low']) / data['Close']) * 100
print("   Volatilité mobile 30 jours")
print("   Volatilité mobile 90 jours")
print("   Range quotidien (%)\n")

# Variables techniques (moyennes mobiles)
print("Calcul des indicateurs techniques...")
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Distance_Max_Historique'] = ((data['Close'] - data['Close'].expanding().max()) / data['Close'].expanding().max()) * 100
print("   SMA 20 jours")
print("   SMA 50 jours")
print("   SMA 200 jours")
print("   Distance au plus haut historique (%)\n")

# 3.5 VALIDATION DES NOUVELLES VARIABLES
print("\n--- 3.5 VALIDATION DES NOUVELLES VARIABLES ---\n")

# Vérification des NaN introduits par les calculs
print("Vérification des NaN dans les nouvelles variables :\n")
nouvelles_colonnes = ['Rendement_Quotidien', 'Rendement_Cumule', 'Log_Rendement',
                      'Volatilite_30j', 'Volatilite_90j', 'Range_Quotidien',
                      'SMA_20', 'SMA_50', 'SMA_200', 'Distance_Max_Historique']

for col in nouvelles_colonnes:
    nb_nan = data[col].isnull().sum()
    if nb_nan > 0:
        print(f"   {col:30} : {nb_nan} NaN (normal pour calculs mobiles)")
    else:
        print(f"   {col:30} : Aucun NaN")

print("\nAperçu des données enrichies (10 dernières lignes) :\n")
colonnes_affichage = ['Close', 'Rendement_Quotidien', 'SMA_20', 'SMA_50', 'Volatilite_30j']
print(data[colonnes_affichage].tail(10))

print("\n" + "&"*80)
print("RÉSUMÉ FINAL")
print("&"*80)
print(f"Données nettoyées et enrichies")
print(f"{len(data)} lignes de données")
print(f"{len(data.columns)} colonnes au total")
print(f"{len(nouvelles_colonnes)} nouvelles variables créées")
print(f"Période : {data.index.min().strftime('%d/%m/%Y')} -> {data.index.max().strftime('%d/%m/%Y')}")
print("~é"*80)

