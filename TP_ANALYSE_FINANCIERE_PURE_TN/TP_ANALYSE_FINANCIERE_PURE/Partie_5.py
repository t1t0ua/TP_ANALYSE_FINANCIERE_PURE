import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Récupération et préparation des données (reprise des parties précédentes)
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
data['Trimestre'] = data.index.quarter
data['Volatilite_30j'] = data['Rendement_Quotidien'].rolling(window=30).std()
data['Volatilite_90j'] = data['Rendement_Quotidien'].rolling(window=90).std()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Max_Historique'] = data['Close'].expanding().max()
data['Drawdown'] = ((data['Close'] - data['Max_Historique']) / data['Max_Historique']) * 100

############
# PARTIE 5: ANALYSER ET CALCULER LES KPI
############

print("="*80)
print("ÉTAPE 5 : ANALYSER ET CALCULER LES KPI")
print("="*80)

# 5.1 KPI DE PERFORMANCE
print("\n" + "-"*80)
print("5.1 KPI DE PERFORMANCE")
print("-"*80)

# Calcul des rendements sur différentes périodes
prix_initial = data['Close'].iloc[0]
prix_final = data['Close'].iloc[-1]
nb_annees_total = (data.index[-1] - data.index[0]).days / 365.25

# Rendement total sur 15 ans
rendement_total_15ans = ((prix_final - prix_initial) / prix_initial) * 100

# Rendement sur 10 ans
data_10ans = data[data.index >= (data.index[-1] - pd.DateOffset(years=10))]
if len(data_10ans) > 0:
    prix_initial_10ans = data_10ans['Close'].iloc[0]
    rendement_10ans = ((prix_final - prix_initial_10ans) / prix_initial_10ans) * 100
else:
    rendement_10ans = None

# Rendement sur 5 ans
data_5ans = data[data.index >= (data.index[-1] - pd.DateOffset(years=5))]
if len(data_5ans) > 0:
    prix_initial_5ans = data_5ans['Close'].iloc[0]
    rendement_5ans = ((prix_final - prix_initial_5ans) / prix_initial_5ans) * 100
else:
    rendement_5ans = None

# Rendement sur 1 an
data_1an = data[data.index >= (data.index[-1] - pd.DateOffset(years=1))]
if len(data_1an) > 0:
    prix_initial_1an = data_1an['Close'].iloc[0]
    rendement_1an = ((prix_final - prix_initial_1an) / prix_initial_1an) * 100
else:
    rendement_1an = None

print("\nRENDEMENT TOTAL SUR DIFFÉRENTES PÉRIODES :\n")
print(f"   Prix initial (15 ans)       : ${prix_initial:.2f}")
print(f"   Prix final (aujourd'hui)    : ${prix_final:.2f}")
print(f"   Multiplication du capital   : x{(prix_final/prix_initial):.2f}")
print()
print(f"   Rendement total 15 ans      : {rendement_total_15ans:+,.2f}%")
if rendement_10ans is not None:
    print(f"   Rendement total 10 ans      : {rendement_10ans:+,.2f}%")
if rendement_5ans is not None:
    print(f"   Rendement total 5 ans       : {rendement_5ans:+,.2f}%")
if rendement_1an is not None:
    print(f"   Rendement total 1 an        : {rendement_1an:+,.2f}%")

print("\n   INTERPRÉTATION :")
if rendement_total_15ans > 200:
    print("   >> Performance EXCEPTIONNELLE : rendement supérieur à 200% sur 15 ans")
elif rendement_total_15ans > 100:
    print("   >> Performance EXCELLENTE : rendement supérieur à 100% sur 15 ans")
elif rendement_total_15ans > 50:
    print("   >> Performance BONNE : rendement supérieur à 50% sur 15 ans")
else:
    print("   >> Performance MODÉRÉE")

# Calcul du CAGR (Compound Annual Growth Rate)
print("\n" + "-"*40)
print("RENDEMENT ANNUALISÉ (CAGR)")
print("-"*40)

cagr_15ans = ((prix_final / prix_initial) ** (1 / nb_annees_total) - 1) * 100

if rendement_10ans is not None:
    nb_annees_10ans = len(data_10ans) / 252  # 252 jours de trading par an
    cagr_10ans = ((prix_final / prix_initial_10ans) ** (1 / nb_annees_10ans) - 1) * 100
else:
    cagr_10ans = None

if rendement_5ans is not None:
    nb_annees_5ans = len(data_5ans) / 252
    cagr_5ans = ((prix_final / prix_initial_5ans) ** (1 / nb_annees_5ans) - 1) * 100
else:
    cagr_5ans = None

print(f"\n   CAGR 15 ans                 : {cagr_15ans:+.2f}% par an")
if cagr_10ans is not None:
    print(f"   CAGR 10 ans                 : {cagr_10ans:+.2f}% par an")
if cagr_5ans is not None:
    print(f"   CAGR 5 ans                  : {cagr_5ans:+.2f}% par an")

print("\n   COMPARAISON AVEC LES BENCHMARKS :")
print(f"   Taux sans risque            : ~3% par an")
print(f"   S&P 500 historique          : ~10% par an")
print(f"   Obligations                 : ~5% par an")
print()

if cagr_15ans > 10:
    print(f"   >> Microsoft (CAGR {cagr_15ans:.2f}%) SURPERFORME le S&P 500 (~10%)")
    performance = "EXCELLENTE"
elif cagr_15ans > 5:
    print(f"   >> Microsoft (CAGR {cagr_15ans:.2f}%) surperforme les obligations")
    performance = "BONNE"
else:
    print(f"   >> Microsoft (CAGR {cagr_15ans:.2f}%) sous-performe")
    performance = "MODÉRÉE"

# Graphique de l'évolution du portefeuille
print("\n   ÉVOLUTION D'UN INVESTISSEMENT DE 10 000 EUR :\n")

investissement_initial = 10000
data['Valeur_Portefeuille'] = investissement_initial * (data['Close'] / prix_initial)
valeur_finale = data['Valeur_Portefeuille'].iloc[-1]
gain_absolu = valeur_finale - investissement_initial

print(f"   Investissement initial      : {investissement_initial:,.0f} EUR")
print(f"   Valeur finale               : {valeur_finale:,.0f} EUR")
print(f"   Gain absolu                 : {gain_absolu:+,.0f} EUR")
print(f"   Performance                 : {rendement_total_15ans:+.2f}%")

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Valeur_Portefeuille'], linewidth=2, color='#2E86AB', label='Portefeuille Microsoft')
plt.axhline(y=investissement_initial, color='red', linestyle='--', linewidth=1.5, label='Investissement initial', alpha=0.7)
plt.fill_between(data.index, investissement_initial, data['Valeur_Portefeuille'], 
                 where=(data['Valeur_Portefeuille'] >= investissement_initial), 
                 interpolate=True, alpha=0.3, color='green', label='Gains')
plt.title('Évolution d\'un investissement de 10 000 EUR dans Microsoft (2010-2025)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Valeur (EUR)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5.2 KPI DE RISQUE
print("\n" + "-"*80)
print("5.2 KPI DE RISQUE")
print("-"*80)

print("\nVOLATILITÉ :\n")

# Calcul de la volatilité
rendements = data['Rendement_Quotidien'].dropna()
volatilite_quotidienne = rendements.std()
volatilite_annualisee = volatilite_quotidienne * np.sqrt(252)  # 252 jours de trading

print(f"   Volatilité quotidienne      : {volatilite_quotidienne:.2f}%")
print(f"   Volatilité annualisée       : {volatilite_annualisee:.2f}%")

print("\n   INTERPRÉTATION DU NIVEAU DE RISQUE :")
if volatilite_annualisee < 15:
    niveau_risque = "FAIBLE"
    print(f"   >> Volatilité < 15% : Risque {niveau_risque}")
elif volatilite_annualisee < 25:
    niveau_risque = "MODÉRÉ"
    print(f"   >> Volatilité entre 15% et 25% : Risque {niveau_risque}")
else:
    niveau_risque = "ÉLEVÉ"
    print(f"   >> Volatilité > 25% : Risque {niveau_risque}")

print(f"\n   Niveau de risque : {niveau_risque} ({volatilite_annualisee:.2f}%)")

# Drawdown maximum
print("\n" + "-"*40)
print("DRAWDOWN MAXIMUM")
print("-"*40)

drawdown_max = data['Drawdown'].min()
idx_drawdown_max = data['Drawdown'].idxmin()
prix_au_pic = data.loc[idx_drawdown_max, 'Max_Historique']
prix_au_creux = data.loc[idx_drawdown_max, 'Close']
date_pic = data[data['Close'] == prix_au_pic].index[0]

print(f"\n   Date du pic historique      : {date_pic.strftime('%d/%m/%Y')}")
print(f"   Prix au pic                 : ${prix_au_pic:.2f}")
print(f"   Date du creux               : {idx_drawdown_max.strftime('%d/%m/%Y')}")
print(f"   Prix au creux               : ${prix_au_creux:.2f}")
print(f"   Drawdown maximum            : {drawdown_max:.2f}%")

# Calcul du temps de récupération
data_apres_creux = data.loc[idx_drawdown_max:]
recuperation = data_apres_creux[data_apres_creux['Drawdown'] >= -0.5]

if len(recuperation) > 0:
    date_recuperation = recuperation.index[0]
    jours_recuperation = (date_recuperation - idx_drawdown_max).days
    mois_recuperation = jours_recuperation / 30
    print(f"   Date de récupération        : {date_recuperation.strftime('%d/%m/%Y')}")
    print(f"   Temps de récupération       : {jours_recuperation} jours ({mois_recuperation:.1f} mois)")
else:
    print(f"   Statut                      : Non encore récupéré du drawdown maximum")

print("\n   INTERPRÉTATION :")
print(f"   >> Un investisseur qui aurait acheté au plus haut ({date_pic.strftime('%d/%m/%Y')})")
print(f"      aurait subi une perte maximale de {abs(drawdown_max):.2f}%")
print(f"   >> Cela représente le pire scénario d'investissement possible sur la période")

# Graphique du drawdown
plt.figure(figsize=(14, 6))
plt.fill_between(data.index, data['Drawdown'], 0, color='#D62246', alpha=0.5)
plt.plot(data.index, data['Drawdown'], color='#8B0000', linewidth=1.5)
plt.axhline(y=drawdown_max, color='red', linestyle='--', linewidth=1.5, label=f'Drawdown max ({drawdown_max:.2f}%)', alpha=0.7)
plt.title('Drawdown - Perte maximale depuis le pic historique', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown (%)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# Sharpe Ratio
print("\n" + "-"*40)
print("SHARPE RATIO")
print("-"*40)

taux_sans_risque = 3.0  # Hypothèse : 3% annuel
rendement_moyen_annuel = rendements.mean() * 252
excess_return = rendement_moyen_annuel - taux_sans_risque
sharpe_ratio = excess_return / volatilite_annualisee

print(f"\n   Rendement moyen annuel      : {rendement_moyen_annuel:.2f}%")
print(f"   Taux sans risque (hypothèse): {taux_sans_risque:.2f}%")
print(f"   Rendement excédentaire      : {excess_return:.2f}%")
print(f"   Volatilité annualisée       : {volatilite_annualisee:.2f}%")
print(f"\n   SHARPE RATIO                : {sharpe_ratio:.3f}")

print("\n   INTERPRÉTATION :")
if sharpe_ratio < 0:
    evaluation_sharpe = "MAUVAIS"
    print(f"   >> Sharpe < 0 : {evaluation_sharpe} - Rendement inférieur au taux sans risque")
elif sharpe_ratio < 1:
    evaluation_sharpe = "ACCEPTABLE"
    print(f"   >> Sharpe entre 0 et 1 : {evaluation_sharpe}")
elif sharpe_ratio < 2:
    evaluation_sharpe = "BON"
    print(f"   >> Sharpe entre 1 et 2 : {evaluation_sharpe}")
else:
    evaluation_sharpe = "EXCELLENT"
    print(f"   >> Sharpe > 2 : {evaluation_sharpe}")

print(f"\n   Signification : Pour chaque unité de risque pris, l'investisseur obtient")
print(f"                   {sharpe_ratio:.3f} unité(s) de rendement excédentaire")

# Ratio Rendement/Risque
print("\n" + "-"*40)
print("RATIO RENDEMENT/RISQUE")
print("-"*40)

ratio_rdt_risque = cagr_15ans / volatilite_annualisee

print(f"\n   CAGR                        : {cagr_15ans:.2f}%")
print(f"   Volatilité annualisée       : {volatilite_annualisee:.2f}%")
print(f"   Ratio Rendement/Risque      : {ratio_rdt_risque:.3f}")

print("\n   INTERPRÉTATION :")
if ratio_rdt_risque > 1:
    print(f"   >> Ratio > 1 : Le rendement ({cagr_15ans:.2f}%) est SUPÉRIEUR au risque ({volatilite_annualisee:.2f}%)")
    print(f"      Bon équilibre rendement/risque")
else:
    print(f"   >> Ratio < 1 : Le risque ({volatilite_annualisee:.2f}%) est SUPÉRIEUR au rendement ({cagr_15ans:.2f}%)")
    print(f"      Équilibre défavorable")

# Value at Risk (VaR)
print("\n" + "-"*40)
print("VALUE AT RISK (VaR)")
print("-"*40)

var_95_quotidien = np.percentile(rendements, 5)
var_95_mensuel = var_95_quotidien * np.sqrt(21)  # 21 jours de trading par mois

print(f"\n   VaR 95% (1 jour)            : {var_95_quotidien:.2f}%")
print(f"   VaR 95% (1 mois)            : {var_95_mensuel:.2f}%")

print("\n   INTERPRÉTATION :")
print(f"   >> Dans 95% des cas, la perte quotidienne ne dépassera pas {abs(var_95_quotidien):.2f}%")
print(f"   >> Dans 95% des cas, la perte mensuelle ne dépassera pas {abs(var_95_mensuel):.2f}%")
print(f"   >> Dans 5% des cas, la perte peut être supérieure (événements extrêmes)")

# 5.3 KPI DE TENDANCE (INDICATEURS TECHNIQUES)
print("\n" + "-"*80)
print("5.3 KPI DE TENDANCE - INDICATEURS TECHNIQUES")
print("-"*80)

print("\nPOSITION ACTUELLE VS MOYENNES MOBILES :\n")

prix_actuel = data['Close'].iloc[-1]
sma20_actuel = data['SMA_20'].iloc[-1]
sma50_actuel = data['SMA_50'].iloc[-1]
sma200_actuel = data['SMA_200'].iloc[-1]

ecart_sma20 = ((prix_actuel - sma20_actuel) / sma20_actuel) * 100
ecart_sma50 = ((prix_actuel - sma50_actuel) / sma50_actuel) * 100
ecart_sma200 = ((prix_actuel - sma200_actuel) / sma200_actuel) * 100

print(f"   Prix actuel                 : ${prix_actuel:.2f}")
print(f"   SMA 20 jours                : ${sma20_actuel:.2f} ({ecart_sma20:+.2f}%)")
print(f"   SMA 50 jours                : ${sma50_actuel:.2f} ({ecart_sma50:+.2f}%)")
print(f"   SMA 200 jours               : ${sma200_actuel:.2f} ({ecart_sma200:+.2f}%)")

print("\n   INTERPRÉTATION :")

if prix_actuel > sma200_actuel:
    tendance_200 = "HAUSSIÈRE"
    signal_200 = "Prix au-dessus de la SMA 200 : tendance haussière long terme"
else:
    tendance_200 = "BAISSIÈRE"
    signal_200 = "Prix en-dessous de la SMA 200 : tendance baissière long terme"

if prix_actuel > sma50_actuel:
    tendance_50 = "HAUSSIÈRE"
    signal_50 = "Prix au-dessus de la SMA 50 : tendance haussière moyen terme"
else:
    tendance_50 = "BAISSIÈRE"
    signal_50 = "Prix en-dessous de la SMA 50 : tendance baissière moyen terme"

print(f"   >> {signal_200}")
print(f"   >> {signal_50}")

# Signal de tendance Golden Cross / Death Cross
print("\n" + "-"*40)
print("SIGNAL DE TENDANCE LONG TERME")
print("-"*40)

print(f"\n   SMA 50 jours                : ${sma50_actuel:.2f}")
print(f"   SMA 200 jours               : ${sma200_actuel:.2f}")

if sma50_actuel > sma200_actuel:
    signal_cross = "GOLDEN CROSS"
    interpretation_cross = "Signal HAUSSIER long terme"
    ecart_cross = ((sma50_actuel - sma200_actuel) / sma200_actuel) * 100
    print(f"\n   Configuration               : {signal_cross}")
    print(f"   Écart SMA 50/200            : +{ecart_cross:.2f}%")
    print(f"   Interprétation              : {interpretation_cross}")
else:
    signal_cross = "DEATH CROSS"
    interpretation_cross = "Signal BAISSIER long terme"
    ecart_cross = ((sma200_actuel - sma50_actuel) / sma200_actuel) * 100
    print(f"\n   Configuration               : {signal_cross}")
    print(f"   Écart SMA 200/50            : +{ecart_cross:.2f}%")
    print(f"   Interprétation              : {interpretation_cross}")

# Recherche du dernier croisement
data['SMA_50_prev'] = data['SMA_50'].shift(1)
data['SMA_200_prev'] = data['SMA_200'].shift(1)
data['Croisement'] = ((data['SMA_50'] > data['SMA_200']) & (data['SMA_50_prev'] <= data['SMA_200_prev'])) | \
                     ((data['SMA_50'] < data['SMA_200']) & (data['SMA_50_prev'] >= data['SMA_200_prev']))

croisements = data[data['Croisement']].tail(1)
if len(croisements) > 0:
    dernier_croisement = croisements.index[0]
    type_croisement = "Golden Cross" if croisements['SMA_50'].iloc[0] > croisements['SMA_200'].iloc[0] else "Death Cross"
    jours_depuis = (data.index[-1] - dernier_croisement).days
    print(f"\n   Dernier croisement          : {type_croisement}")
    print(f"   Date                        : {dernier_croisement.strftime('%d/%m/%Y')}")
    print(f"   Jours écoulés               : {jours_depuis} jours ({jours_depuis/30:.1f} mois)")

# Distance au plus haut historique
print("\n" + "-"*40)
print("DISTANCE AU PLUS HAUT HISTORIQUE")
print("-"*40)

prix_max_historique = data['Close'].max()
date_max_historique = data['Close'].idxmax()
distance_max = ((prix_actuel - prix_max_historique) / prix_max_historique) * 100

print(f"\n   Plus haut historique        : ${prix_max_historique:.2f}")
print(f"   Date                        : {date_max_historique.strftime('%d/%m/%Y')}")
print(f"   Prix actuel                 : ${prix_actuel:.2f}")
print(f"   Distance                    : {distance_max:+.2f}%")

if distance_max >= -1:
    print(f"\n   >> Le prix actuel est proche du plus haut historique")
    print(f"      Potentiel résistance - surveiller pour confirmation de cassure")
elif distance_max >= -5:
    print(f"\n   >> Le prix est légèrement en-dessous du plus haut historique")
    print(f"      Zone de potentielle reprise haussière")
else:
    print(f"\n   >> Le prix est significativement en-dessous du plus haut historique")
    print(f"      Potentiel de rattrapage important")

# 5.4 SYNTHÈSE DES INDICATEURS
print("\n" + "-"*80)
print("5.4 TABLEAU DE BORD DES KPI")
print("-"*80)

print("\n" + "="*80)
print(" "*25 + "TABLEAU DE BORD MICROSOFT")
print("="*80)

print("\n[PERFORMANCE]")
print("-" * 80)
print(f"  Rendement total 15 ans       : {rendement_total_15ans:>+10,.2f}%  | {performance}")
print(f"  CAGR (rendement annualisé)   : {cagr_15ans:>+10.2f}%  | {performance}")
print(f"  Multiplication du capital    : {(prix_final/prix_initial):>10.2f}x   | {performance}")
if rendement_1an is not None:
    perf_1an = "Positive" if rendement_1an > 0 else "Négative"
    print(f"  Performance 1 an             : {rendement_1an:>+10.2f}%  | {perf_1an}")

print("\n[RISQUE]")
print("-" * 80)
print(f"  Volatilité annualisée        : {volatilite_annualisee:>10.2f}%  | {niveau_risque}")
print(f"  Drawdown maximum             : {drawdown_max:>10.2f}%  | Important")
if len(recuperation) > 0:
    print(f"  Temps de récupération        : {mois_recuperation:>10.1f} mois | Rapide" if mois_recuperation < 12 else f"  Temps de récupération        : {mois_recuperation:>10.1f} mois | Long")
print(f"  Sharpe Ratio                 : {sharpe_ratio:>10.3f}   | {evaluation_sharpe}")
print(f"  Ratio Rendement/Risque       : {ratio_rdt_risque:>10.3f}   | {'Favorable' if ratio_rdt_risque > 1 else 'Défavorable'}")
print(f"  VaR 95% (1 jour)             : {var_95_quotidien:>10.2f}%  | Risque quotidien")

print("\n[TECHNIQUE]")
print("-" * 80)
print(f"  Prix actuel                  : ${prix_actuel:>9.2f}   | Suivi")
print(f"  Position vs SMA 20           : {ecart_sma20:>+10.2f}%  | {tendance_50}")
print(f"  Position vs SMA 50           : {ecart_sma50:>+10.2f}%  | {tendance_50}")
print(f"  Position vs SMA 200          : {ecart_sma200:>+10.2f}%  | {tendance_200}")
print(f"  Tendance long terme          : {signal_cross:>13}  | {interpretation_cross}")
print(f"  Distance au max historique   : {distance_max:>+10.2f}%  | {'Proche' if distance_max > -5 else 'Éloigné'}")

print("\n" + "="*80)

# Système de scoring
print("\n" + "-"*40)
print("SCORE GLOBAL")
print("-"*40)

# Score Performance (0-10)
if cagr_15ans > 15:
    score_performance = 10
elif cagr_15ans > 12:
    score_performance = 8
elif cagr_15ans > 10:
    score_performance = 7
elif cagr_15ans > 7:
    score_performance = 5
else:
    score_performance = 3

# Score Risque (0-10) - inversé car moins de risque = meilleur score
if volatilite_annualisee < 15:
    score_risque = 10
elif volatilite_annualisee < 20:
    score_risque = 8
elif volatilite_annualisee < 25:
    score_risque = 6
elif volatilite_annualisee < 30:
    score_risque = 4
else:
    score_risque = 2

# Ajustement pour drawdown
if abs(drawdown_max) < 20:
    score_risque += 0
elif abs(drawdown_max) < 30:
    score_risque -= 1
else:
    score_risque -= 2

score_risque = max(0, min(10, score_risque))  # Limiter entre 0 et 10

# Score Technique (0-10)
score_technique = 0
if prix_actuel > sma200_actuel:
    score_technique += 3
if prix_actuel > sma50_actuel:
    score_technique += 3
if sma50_actuel > sma200_actuel:  # Golden Cross
    score_technique += 3
if distance_max > -10:  # Proche du max
    score_technique += 1

# Score final (moyenne pondérée)
poids_perf = 0.4
poids_risque = 0.3
# Poids des scores
poids_technique = 0.3

# Calcul du score global
score_global = (
    score_performance * poids_perf +
    score_risque * poids_risque +
    score_technique * poids_technique
)

print(f"\nScore Performance : {score_performance}/10")
print(f"Score Risque      : {score_risque}/10")
print(f"Score Technique   : {score_technique}/10")

print("\n" + "="*40)
print(f"SCORE GLOBAL MICROSOFT : {score_global:.2f} / 10")
print("="*40)

# Interprétation du score global
print("\nINTERPRÉTATION GLOBALE :")
if score_global >= 8:
    conclusion = "TRÈS ATTRACTIF"
    print(">> Actif de très grande qualité")
    print(">> Excellent compromis rendement / risque")
    print(">> Tendance technique favorable")
elif score_global >= 6:
    conclusion = "ATTRACTIF"
    print(">> Actif solide avec de bonnes performances")
    print(">> Risque maîtrisé")
    print(">> Convient à un investisseur long terme")
elif score_global >= 4:
    conclusion = "NEUTRE"
    print(">> Performances correctes mais sans avantage marqué")
    print(">> À surveiller selon le contexte de marché")
else:
    conclusion = "PEU ATTRACTIF"
    print(">> Rendement ou profil de risque peu favorable")
    print(">> Prudence recommandée")

print(f"\nConclusion finale : MICROSOFT est jugé **{conclusion}** selon les KPI analysés.")
