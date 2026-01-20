import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Recuperation des donnees historiques de Microsoft sur 15 ans
data = yf.download("MSFT", start="2010-01-01", end="2025-01-01")

# Erreur en cas de non-chargement du fichier
if data is None or data.empty:
    print("erreur de chargement")
    exit()

# Aplatir la structure multi-index des colonnes
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Trouver le nom de la colonne du prix ajuste
prix_col = None
for col in data.columns:
    if 'adj' in col.lower() and 'close' in col.lower():
        prix_col = col
        break

if prix_col is None:
    prix_col = 'Close'

print(f"Colonne utilisee pour les prix : {prix_col}")
print()

# Afficher les 10 premieres lignes
print(data.head(10))

# Afficher les 5 dernieres lignes
print(data.tail(5))

# Afficher le nombre de jours de trading
print(f"Nombre de jours de trading : {len(data)}")
print()

# Afficher les types de donnees de chaque colonne
print("type de donnee present dans le dataframe")
for donne in data.columns:
    print(donne, data[donne].iloc[0], type(data[donne].iloc[0]))
print()

# Afficher la memoire utilisee par le DataFrame
print("Taille du DataFrame : " + str(data.size))
print()

# Recherche de periode manquante
data_temp = data.copy()
data_temp['date_diff'] = data_temp.index.to_series().diff().dt.days

ecarts_stats = data_temp['date_diff'].value_counts().sort_index()
print("Distribution des ecarts entre dates consecutives :")
for jours, count in ecarts_stats.head(10).items():
    if pd.notna(jours):
        print(f"   {int(jours)} jour(s) d'ecart : {count:,} occurrences")

gaps_weekend = data_temp[data_temp['date_diff'] == 3]
gaps_anormaux = data_temp[data_temp['date_diff'] > 4]

print()
print(f"Weekends normaux (3 jours)  : {len(gaps_weekend):,} occurrences")
print(f"Gaps anormaux (>4 jours)    : {len(gaps_anormaux):,} occurrences")

if len(gaps_anormaux) > 0:
    print()
    print("Detail des gaps anormaux :")
    for idx in gaps_anormaux.head().index:
        nb_jours_gap = int(gaps_anormaux.loc[idx, 'date_diff'])
        print(f"   {idx.strftime('%d/%m/%Y')} : gap de {nb_jours_gap} jours")
print()

# Afficher les volumes d'echange
volume_stats = data['Volume'].describe()
print("Statistiques descriptives du volume :")
print(f"   Minimum          : {data['Volume'].min():>15,.0f} actions")
print(f"   25e percentile   : {data['Volume'].quantile(0.25):>15,.0f} actions")
print(f"   Mediane (50%)    : {data['Volume'].median():>15,.0f} actions")
print(f"   Moyenne          : {data['Volume'].mean():>15,.0f} actions")
print(f"   75e percentile   : {data['Volume'].quantile(0.75):>15,.0f} actions")
print(f"   Maximum          : {data['Volume'].max():>15,.0f} actions")
print(f"   Ecart-type       : {data['Volume'].std():>15,.0f} actions")
print()

coef_variation = (data['Volume'].std() / data['Volume'].mean()) * 100
print(f"Coefficient de variation : {coef_variation:.1f}%")
print()

# Top 5 des jours avec le plus fort volume
print("Top 5 des jours avec le volume le plus eleve :")
top_volume = data.nlargest(5, 'Volume')

for idx, row in top_volume.iterrows():
    print(f"   {idx.strftime('%d/%m/%Y')} : {row['Volume']:>15,.0f} actions (prix: ${row[prix_col]:.2f})")

print()

# Afficher la tendance visuelle generale
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

print("Evolution des prix :")
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
print(f"   CAGR (rendement annualise)          : {cagr:+.2f}% par an")
print()

if rendement_total > 0:
    print("Tendance generale : HAUSSIERE (forte croissance)")
else:
    print("Tendance generale : BAISSIERE (decroissance)")

print()

# Création graphique 
plt.figure(figsize=(14, 6))
plt.plot(data.index, data[prix_col], linewidth=1.5, color='#0078D4', label=f'Prix Ajuste ({prix_col})')
plt.title('Evolution du prix de l\'action Microsoft (2010-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Prix ($)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

print()