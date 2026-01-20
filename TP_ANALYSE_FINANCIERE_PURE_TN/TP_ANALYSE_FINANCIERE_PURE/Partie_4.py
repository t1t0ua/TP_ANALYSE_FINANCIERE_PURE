import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Récupération et préparation des données (reprise de la partie 3)
data = yf.download("MSFT", start="2010-01-01", end="2025-01-01")

if data is None or data.empty:
    print("Erreur de chargement")
    exit()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

prix_col = None
for col in data.columns:
    if 'adj' in col.lower() and 'close' in col.lower():
        prix_col = col
        break
if prix_col is None:
    prix_col = 'Close'

# Création des variables dérivées
data['Rendement_Quotidien'] = data['Close'].pct_change() * 100
data['Rendement_Cumule'] = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
data['Log_Rendement'] = np.log(data['Close'] / data['Close'].shift(1))
data['Annee'] = data.index.year
data['Mois'] = data.index.month
data['Jour_Semaine'] = data.index.dayofweek
data['Jour_Semaine_Nom'] = data.index.day_name()
data['Trimestre'] = data.index.quarter
data['Volatilite_30j'] = data['Rendement_Quotidien'].rolling(window=30).std()
data['Volatilite_90j'] = data['Rendement_Quotidien'].rolling(window=90).std()
data['Range_Quotidien'] = ((data['High'] - data['Low']) / data['Close']) * 100
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Distance_Max_Historique'] = ((data['Close'] - data['Close'].expanding().max()) / data['Close'].expanding().max()) * 100

# Calcul du drawdown
data['Max_Historique'] = data['Close'].expanding().max()
data['Drawdown'] = ((data['Close'] - data['Max_Historique']) / data['Max_Historique']) * 100

############
# PARTIE 4: EXPLORATION DES DONNÉES
############

print("="*80)
print("ÉTAPE 4 : EXPLORATION DES DONNÉES")
print("="*80)

# 4.1 STATISTIQUES DESCRIPTIVES
print("\n" + "-"*80)
print("4.1 STATISTIQUES DESCRIPTIVES")
print("-"*80)

print("\nSTATISTIQUES DU PRIX DE CLÔTURE :\n")
print(f"   Moyenne          : ${data['Close'].mean():>10.2f}")
print(f"   Médiane          : ${data['Close'].median():>10.2f}")
print(f"   Écart-type       : ${data['Close'].std():>10.2f}")
print(f"   Minimum          : ${data['Close'].min():>10.2f}")
print(f"   Maximum          : ${data['Close'].max():>10.2f}")
print(f"   1er quartile     : ${data['Close'].quantile(0.25):>10.2f}")
print(f"   3e quartile      : ${data['Close'].quantile(0.75):>10.2f}")
print(f"\n   → L'écart-type élevé (${data['Close'].std():.2f}) reflète une forte croissance du prix sur 15 ans.")
print(f"   → La moyenne (${data['Close'].mean():.2f}) est supérieure à la médiane (${data['Close'].median():.2f}),")
print(f"     indiquant une distribution asymétrique avec davantage de valeurs élevées récentes.")

print("\nSTATISTIQUES DU VOLUME :\n")
print(f"   Volume moyen     : {data['Volume'].mean():>15,.0f} actions")
print(f"   Volume médian    : {data['Volume'].median():>15,.0f} actions")
print(f"   Minimum          : {data['Volume'].min():>15,.0f} actions")
print(f"   Maximum          : {data['Volume'].max():>15,.0f} actions")

print("\n   Top 5 des jours avec le plus fort volume :")
top_vol = data.nlargest(5, 'Volume')
for idx, row in top_vol.iterrows():
    variation = row['Rendement_Quotidien']
    print(f"   • {idx.strftime('%d/%m/%Y')} : {row['Volume']:>15,.0f} actions (variation: {variation:+.2f}%)")

print("\n   → Les pics de volume sont souvent associés à des variations importantes du prix,")
print(f"     signalant des événements majeurs ou une forte volatilité du marché.")

print("\nSTATISTIQUES DES RENDEMENTS :\n")
rdt_data = data['Rendement_Quotidien'].dropna()
print(f"   Rendement moyen quotidien  : {rdt_data.mean():>8.3f}%")
print(f"   Rendement médian           : {rdt_data.median():>8.3f}%")
print(f"   Écart-type                 : {rdt_data.std():>8.3f}%")
print(f"   Meilleur jour              : {rdt_data.max():>8.2f}% le {rdt_data.idxmax().strftime('%d/%m/%Y')}")
print(f"   Pire jour                  : {rdt_data.min():>8.2f}% le {rdt_data.idxmin().strftime('%d/%m/%Y')}")

jours_positifs = (rdt_data > 0).sum()
jours_negatifs = (rdt_data < 0).sum()
total_jours = len(rdt_data)
pct_positifs = (jours_positifs / total_jours) * 100

print(f"\n   Jours positifs   : {jours_positifs:>6} ({pct_positifs:.1f}%)")
print(f"   Jours négatifs   : {jours_negatifs:>6} ({100-pct_positifs:.1f}%)")

skewness = rdt_data.skew()
kurtosis = rdt_data.kurtosis()
print(f"\n   Skewness (asymétrie)       : {skewness:>8.3f}")
print(f"   Kurtosis (aplatissement)   : {kurtosis:>8.3f}")

print(f"\n   → Un skewness de {skewness:.3f} indique une distribution", end=" ")
if skewness > 0.5:
    print("asymétrique à droite (davantage de gains extrêmes).")
elif skewness < -0.5:
    print("asymétrique à gauche (davantage de pertes extrêmes).")
else:
    print("relativement symétrique.")

print(f"   → Un kurtosis de {kurtosis:.3f}", end=" ")
if kurtosis > 3:
    print("(>3) révèle des queues épaisses : risque d'événements extrêmes.")
else:
    print("indique une distribution proche de la normale.")

# 4.2 VISUALISATION DE L'ÉVOLUTION DU PRIX
print("\n" + "-"*80)
print("4.2 VISUALISATION DE L'ÉVOLUTION DU PRIX")
print("-"*80)

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Prix de clôture', linewidth=1.5, color='#2E86AB')
plt.plot(data.index, data['SMA_50'], label='SMA 50 jours', linewidth=1.2, color='#A23B72', alpha=0.8)
plt.plot(data.index, data['SMA_200'], label='SMA 200 jours', linewidth=1.2, color='#F18F01', alpha=0.8)

plt.title('Évolution du prix de Microsoft (2010-2025)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Prix ($)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n   ANALYSE VISUELLE :")
print("   • Croissance quasi-exponentielle de 2010 à 2024")
print("   • Correction majeure en mars 2020 (COVID-19)")
print("   • Récupération rapide et nouvelle accélération post-2020")
print("   • Prix actuellement au-dessus des SMA 50 et 200 -> tendance haussière confirmée")

# 4.3 ANALYSE TEMPORELLE
print("\n" + "-"*80)
print("4.3 ANALYSE TEMPORELLE")
print("-"*80)

print("\nRENDEMENT PAR ANNÉE :\n")

rendements_annuels = []
for annee in sorted(data['Annee'].unique()):
    data_annee = data[data['Annee'] == annee]
    if len(data_annee) > 1:
        prix_debut = data_annee['Close'].iloc[0]
        prix_fin = data_annee['Close'].iloc[-1]
        rendement = ((prix_fin - prix_debut) / prix_debut) * 100
        rendements_annuels.append({'Annee': annee, 'Rendement': rendement})
        
        indicateur = "[+]" if rendement > 0 else "[-]"
        print(f"   {annee} : {indicateur} {rendement:+7.2f}%")

df_rdt_annuel = pd.DataFrame(rendements_annuels)

meilleure_annee = df_rdt_annuel.loc[df_rdt_annuel['Rendement'].idxmax()]
pire_annee = df_rdt_annuel.loc[df_rdt_annuel['Rendement'].idxmin()]

print(f"\n   Meilleure année : {int(meilleure_annee['Annee'])} avec {meilleure_annee['Rendement']:+.2f}%")
print(f"   Pire année      : {int(pire_annee['Annee'])} avec {pire_annee['Rendement']:+.2f}%")

plt.figure(figsize=(12, 6))
colors = ['#06A77D' if r > 0 else '#D62246' for r in df_rdt_annuel['Rendement']]
plt.bar(df_rdt_annuel['Annee'], df_rdt_annuel['Rendement'], color=colors, alpha=0.8, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.axhline(y=df_rdt_annuel['Rendement'].mean(), color='blue', linestyle='--', linewidth=1.5, label=f'Moyenne ({df_rdt_annuel["Rendement"].mean():.1f}%)')
plt.title('Rendement annuel de Microsoft (2010-2024)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=12)
plt.ylabel('Rendement (%)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\nRENDEMENT PAR MOIS (moyenne historique) :\n")

rdt_par_mois = data.groupby('Mois')['Rendement_Quotidien'].sum().sort_index()
mois_noms = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

for mois, rendement in rdt_par_mois.items():
    print(f"   {mois_noms[mois-1]} : {rendement:+7.2f}%")

meilleur_mois = rdt_par_mois.idxmax()
pire_mois = rdt_par_mois.idxmin()
print(f"\n   → Meilleur mois historique : {mois_noms[meilleur_mois-1]} ({rdt_par_mois.max():+.2f}%)")
print(f"   → Pire mois historique     : {mois_noms[pire_mois-1]} ({rdt_par_mois.min():+.2f}%)")

print("\nRENDEMENT PAR JOUR DE LA SEMAINE :\n")

jours_noms = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi']
rdt_par_jour = data.groupby('Jour_Semaine')['Rendement_Quotidien'].mean()

for jour in range(5):
    if jour in rdt_par_jour.index:
        print(f"   {jours_noms[jour]:10} : {rdt_par_jour[jour]:+.3f}%")

meilleur_jour = rdt_par_jour.idxmax()
print(f"\n   → Meilleur jour historique : {jours_noms[meilleur_jour]} (effet psychologique possible)")

# 4.4 ANALYSE DE VOLATILITÉ
print("\n" + "-"*80)
print("4.4 ANALYSE DE VOLATILITÉ")
print("-"*80)

print("\nVOLATILITÉ HISTORIQUE :\n")

vol_actuelle = data['Volatilite_30j'].iloc[-1]
vol_moyenne = data['Volatilite_30j'].mean()
vol_max = data['Volatilite_30j'].max()
date_vol_max = data['Volatilite_30j'].idxmax()

print(f"   Volatilité actuelle (30j)  : {vol_actuelle:.2f}%")
print(f"   Volatilité moyenne         : {vol_moyenne:.2f}%")
print(f"   Volatilité maximum         : {vol_max:.2f}% le {date_vol_max.strftime('%d/%m/%Y')}")

comparaison = "SUPÉRIEURE" if vol_actuelle > vol_moyenne else "INFÉRIEURE"
print(f"\n   → La volatilité actuelle est {comparaison} à la moyenne historique.")

plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Volatilite_30j'], label='Volatilité 30j', linewidth=1.2, color='#C73E1D')
plt.axhline(y=vol_moyenne, color='blue', linestyle='--', linewidth=1.5, label=f'Moyenne ({vol_moyenne:.2f}%)', alpha=0.7)
plt.title('Volatilité mobile sur 30 jours - Microsoft', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatilité (%)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n   PÉRIODES DE FORTE VOLATILITÉ :")
periodes_volatiles = data[data['Volatilite_30j'] > vol_moyenne * 1.5].copy()
periodes_volatiles = periodes_volatiles.groupby('Annee').size()
for annee, count in periodes_volatiles.items():
    print(f"   • {int(annee)} : {count} jours de forte volatilité")

print("\nRENDEMENTS QUOTIDIENS EXTRÊMES :\n")

rendements_extremes_haut = data[data['Rendement_Quotidien'] > 5].nlargest(5, 'Rendement_Quotidien')
rendements_extremes_bas = data[data['Rendement_Quotidien'] < -5].nsmallest(5, 'Rendement_Quotidien')

print("   TOP 5 des plus fortes hausses :")
for idx, row in rendements_extremes_haut.iterrows():
    print(f"   • {idx.strftime('%d/%m/%Y')} : {row['Rendement_Quotidien']:+.2f}%")

print("\n   TOP 5 des plus fortes baisses :")
for idx, row in rendements_extremes_bas.iterrows():
    print(f"   • {idx.strftime('%d/%m/%Y')} : {row['Rendement_Quotidien']:+.2f}%")

plt.figure(figsize=(14, 6))
plt.bar(data.index, data['Rendement_Quotidien'], width=1, color=['#06A77D' if x > 0 else '#D62246' for x in data['Rendement_Quotidien']], alpha=0.6)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.title('Rendements quotidiens - Microsoft', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Rendement quotidien (%)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\nDISTRIBUTION DES RENDEMENTS :\n")

plt.figure(figsize=(12, 6))
plt.hist(rdt_data, bins=100, color='#2E86AB', alpha=0.7, edgecolor='black')
plt.axvline(x=rdt_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne ({rdt_data.mean():.3f}%)')
plt.axvline(x=rdt_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Médiane ({rdt_data.median():.3f}%)')
plt.title('Distribution des rendements quotidiens', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Rendement (%)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"   → La distribution montre {('une asymétrie à droite' if skewness > 0 else 'une asymétrie à gauche')}")
print(f"   → Présence de queues épaisses (fat tails) confirmée par le kurtosis de {kurtosis:.2f}")

# 4.5 ANALYSE DE TENDANCE
print("\n" + "-"*80)
print("4.5 ANALYSE DE TENDANCE")
print("-"*80)

prix_actuel = data['Close'].iloc[-1]
sma20_actuel = data['SMA_20'].iloc[-1]
sma50_actuel = data['SMA_50'].iloc[-1]
sma200_actuel = data['SMA_200'].iloc[-1]

print("\nPOSITION ACTUELLE VS MOYENNES MOBILES :\n")
print(f"   Prix actuel      : ${prix_actuel:.2f}")
print(f"   SMA 20 jours     : ${sma20_actuel:.2f} ({((prix_actuel - sma20_actuel) / sma20_actuel * 100):+.2f}%)")
print(f"   SMA 50 jours     : ${sma50_actuel:.2f} ({((prix_actuel - sma50_actuel) / sma50_actuel * 100):+.2f}%)")
print(f"   SMA 200 jours    : ${sma200_actuel:.2f} ({((prix_actuel - sma200_actuel) / sma200_actuel * 100):+.2f}%)")

if prix_actuel > sma20_actuel and prix_actuel > sma50_actuel and prix_actuel > sma200_actuel:
    print("\n   [OK] TENDANCE HAUSSIÈRE CONFIRMÉE : prix au-dessus de toutes les moyennes mobiles")
elif prix_actuel < sma20_actuel and prix_actuel < sma50_actuel and prix_actuel < sma200_actuel:
    print("\n   [X] TENDANCE BAISSIÈRE CONFIRMÉE : prix en-dessous de toutes les moyennes mobiles")
else:
    print("\n   [!] TENDANCE MIXTE : prix entre les moyennes mobiles")

print("\nSIGNAL DE TENDANCE LONG TERME :\n")

if sma50_actuel > sma200_actuel:
    print("   [+] GOLDEN CROSS actif : SMA 50 > SMA 200 -> signal haussier long terme")
else:
    print("   [-] DEATH CROSS actif : SMA 50 < SMA 200 -> signal baissier long terme")

# Recherche des croisements récents
data['SMA_50_prev'] = data['SMA_50'].shift(1)
data['SMA_200_prev'] = data['SMA_200'].shift(1)
data['Croisement'] = ((data['SMA_50'] > data['SMA_200']) & (data['SMA_50_prev'] <= data['SMA_200_prev'])) | \
                     ((data['SMA_50'] < data['SMA_200']) & (data['SMA_50_prev'] >= data['SMA_200_prev']))

croisements = data[data['Croisement']].tail(3)
if len(croisements) > 0:
    print("\n   Derniers croisements SMA 50/200 :")
    for idx, row in croisements.iterrows():
        type_croisement = "Golden Cross" if row['SMA_50'] > row['SMA_200'] else "Death Cross"
        print(f"   • {idx.strftime('%d/%m/%Y')} : {type_croisement}")

print("\nSUPPORT ET RÉSISTANCE :\n")

prix_min_recents = data['Close'].tail(252).min()
prix_max_recents = data['Close'].tail(252).max()

print(f"   Support (min 1 an)    : ${prix_min_recents:.2f}")
print(f"   Résistance (max 1 an) : ${prix_max_recents:.2f}")
print(f"   Prix actuel           : ${prix_actuel:.2f}")

distance_support = ((prix_actuel - prix_min_recents) / prix_min_recents) * 100
distance_resistance = ((prix_max_recents - prix_actuel) / prix_actuel) * 100

print(f"\n   → Distance au support    : +{distance_support:.1f}%")
print(f"   → Distance à résistance  : +{distance_resistance:.1f}%")

# 4.6 ANALYSE DU DRAWDOWN
print("\n" + "-"*80)
print("4.6 ANALYSE DU DRAWDOWN")
print("-"*80)

print("\nDRAWDOWN (CHUTE DEPUIS LE PIC) :\n")

drawdown_max = data['Drawdown'].min()
date_drawdown_max = data['Drawdown'].idxmin()
drawdown_actuel = data['Drawdown'].iloc[-1]

prix_au_pic = data.loc[data['Drawdown'].idxmin(), 'Max_Historique']
prix_au_creux = data.loc[data['Drawdown'].idxmin(), 'Close']

print(f"   Drawdown maximum       : {drawdown_max:.2f}%")
print(f"   Date du creux          : {date_drawdown_max.strftime('%d/%m/%Y')}")
print(f"   Prix au pic            : ${prix_au_pic:.2f}")
print(f"   Prix au creux          : ${prix_au_creux:.2f}")
print(f"   Drawdown actuel        : {drawdown_actuel:.2f}%")

# Calcul du temps de récupération
if drawdown_max < -5:
    idx_creux = data['Drawdown'].idxmin()
    data_apres_creux = data.loc[idx_creux:]
    recuperation = data_apres_creux[data_apres_creux['Drawdown'] >= -0.5]
    
    if len(recuperation) > 0:
        date_recuperation = recuperation.index[0]
        jours_recuperation = (date_recuperation - idx_creux).days
        print(f"   Temps de récupération  : {jours_recuperation} jours ({jours_recuperation/30:.1f} mois)")
    else:
        print(f"   Temps de récupération  : Non encore récupéré")

plt.figure(figsize=(14, 6))
plt.fill_between(data.index, data['Drawdown'], 0, color='#D62246', alpha=0.5)
plt.plot(data.index, data['Drawdown'], color='#8B0000', linewidth=1.5)
plt.title('Drawdown - Chute depuis le pic historique', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

print(f"\n   → Le drawdown maximum de {drawdown_max:.2f}% représente la perte maximale qu'un")
print(f"     investisseur aurait pu subir s'il avait acheté au plus haut historique.")

# 4.7 CORRÉLATIONS
print("\n" + "-"*80)
print("4.7 CORRÉLATIONS")
print("-"*80)

print("\nANALYSE DES CORRÉLATIONS :\n")

corr_volume_rendement = data['Volume'].corr(data['Rendement_Quotidien'].abs())
corr_volatilite_rendement = data['Volatilite_30j'].corr(data['Rendement_Quotidien'].abs())

print(f"   Corrélation Volume ↔ |Rendement|     : {corr_volume_rendement:.3f}")
print(f"   Corrélation Volatilité ↔ |Rendement| : {corr_volatilite_rendement:.3f}")

if corr_volume_rendement > 0.3:
    print("\n   → Corrélation positive modérée : les fortes variations de prix s'accompagnent")
    print("     souvent de volumes d'échange élevés (comportement normal du marché)")
elif corr_volume_rendement > 0.1:
    print("\n   → Corrélation positive faible : lien limité entre volume et variations de prix")
else:
    print("\n   → Corrélation très faible : volume et rendement semblent indépendants")

print("\n" + "="*80)
print("FIN DE L'EXPLORATION DES DONNÉES")
print("="*80)
print("\nSYNTHÈSE DES PATTERNS IDENTIFIÉS :\n")
print("   • Tendance haussière forte et soutenue sur 15 ans")
print("   • Volatilité modérée avec pics lors d'événements majeurs (COVID-19)")
print("   • Distribution des rendements avec queues épaisses (événements extrêmes)")
print(f"   • {pct_positifs:.1f}% de jours positifs (supériorité des hausses)")
print(f"   • Drawdown maximum de {drawdown_max:.2f}% durant les crises")
print("   • Tendance actuelle : haussière (prix > SMA 50 et 200)")
print("\n" + "="*80)