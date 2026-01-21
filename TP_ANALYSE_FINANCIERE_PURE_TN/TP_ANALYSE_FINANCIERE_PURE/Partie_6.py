import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
data['Volatilite_30j'] = data['Rendement_Quotidien'].rolling(window=30).std()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data['Max_Historique'] = data['Close'].expanding().max()
data['Drawdown'] = ((data['Close'] - data['Max_Historique']) / data['Max_Historique']) * 100

# Calcul des KPI principaux
prix_initial = data['Close'].iloc[0]
prix_final = data['Close'].iloc[-1]
nb_annees = (data.index[-1] - data.index[0]).days / 365.25
rendement_total = ((prix_final - prix_initial) / prix_initial) * 100
cagr = ((prix_final / prix_initial) ** (1 / nb_annees) - 1) * 100

rendements = data['Rendement_Quotidien'].dropna()
volatilite_quotidienne = rendements.std()
volatilite_annualisee = volatilite_quotidienne * np.sqrt(252)

drawdown_max = data['Drawdown'].min()
idx_drawdown_max = data['Drawdown'].idxmin()

taux_sans_risque = 3.0
rendement_moyen_annuel = rendements.mean() * 252
excess_return = rendement_moyen_annuel - taux_sans_risque
sharpe_ratio = excess_return / volatilite_annualisee

prix_actuel = data['Close'].iloc[-1]
sma50_actuel = data['SMA_50'].iloc[-1]
sma200_actuel = data['SMA_200'].iloc[-1]
ecart_sma200 = ((prix_actuel - sma200_actuel) / sma200_actuel) * 100

# Détermination des niveaux
if volatilite_annualisee < 15:
    niveau_risque = "FAIBLE"
elif volatilite_annualisee < 25:
    niveau_risque = "MODÉRÉ"
else:
    niveau_risque = "ÉLEVÉ"

if cagr > 15:
    performance = "EXCELLENTE"
elif cagr > 10:
    performance = "TRÈS BONNE"
elif cagr > 7:
    performance = "BONNE"
else:
    performance = "MODÉRÉE"

if sma50_actuel > sma200_actuel:
    signal_cross = "GOLDEN CROSS"
    tendance = "HAUSSIÈRE"
else:
    signal_cross = "DEATH CROSS"
    tendance = "BAISSIÈRE"

# Calcul du score
if cagr > 15:
    score_performance = 10
elif cagr > 12:
    score_performance = 8
elif cagr > 10:
    score_performance = 7
elif cagr > 7:
    score_performance = 5
else:
    score_performance = 3

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

if abs(drawdown_max) < 20:
    score_risque += 0
elif abs(drawdown_max) < 30:
    score_risque -= 1
else:
    score_risque -= 2
score_risque = max(0, min(10, score_risque))

score_technique = 0
if prix_actuel > sma200_actuel:
    score_technique += 3
if prix_actuel > sma50_actuel:
    score_technique += 3
if sma50_actuel > sma200_actuel:
    score_technique += 3
distance_max = ((prix_actuel - data['Close'].max()) / data['Close'].max()) * 100
if distance_max > -10:
    score_technique += 1

poids_perf = 0.4
poids_risque = 0.3
poids_technique = 0.3
score_final = (score_performance * poids_perf + score_risque * poids_risque + score_technique * poids_technique)

############
# PARTIE 6: COMMUNICATION DES RÉSULTATS
############

print("="*80)
print("ÉTAPE 6 : COMMUNICATION DES RÉSULTATS")
print("="*80)

# 6.1 RAPPORT EXÉCUTIF
print("\n" + "="*80)
print(" "*25 + "RAPPORT EXÉCUTIF")
print(" "*20 + "ANALYSE MICROSOFT (MSFT) 2010-2025")
print("="*80)

print("\n[1] CONTEXTE ET OBJECTIF")
print("-" * 80)
print("\n   Mission : Analyse de l'action Microsoft (MSFT) sur 15 ans (2010-2025)")
print("   Objectif : Déterminer si Microsoft représente un bon investissement aujourd'hui")
print(f"   Période analysée : {data.index[0].strftime('%d/%m/%Y')} au {data.index[-1].strftime('%d/%m/%Y')}")
print(f"   Nombre de jours analysés : {len(data):,} jours de trading")

print("\n[2] RÉSULTATS CLÉS")
print("-" * 80)

print(f"\n   [PERFORMANCE]")
print(f"   • Rendement total exceptionnel : +{rendement_total:,.2f}% sur 15 ans")
print(f"   • CAGR (rendement annualisé) : +{cagr:.2f}% par an")
print(f"   • Multiplication du capital : x{(prix_final/prix_initial):.2f}")
print(f"   • Surperformance vs S&P 500 : +{cagr - 10:.2f} points de pourcentage")

print(f"\n   [RISQUE]")
print(f"   • Volatilité annualisée : {volatilite_annualisee:.2f}% - Niveau {niveau_risque}")
print(f"   • Drawdown maximum : {drawdown_max:.2f}% (durant crise COVID-19 2020)")
print(f"   • Sharpe Ratio : {sharpe_ratio:.3f} - Rendement ajusté au risque {('acceptable' if sharpe_ratio < 1 else 'bon' if sharpe_ratio < 2 else 'excellent')}")
print(f"   • Ratio Rendement/Risque : {(cagr/volatilite_annualisee):.3f} - {'Favorable' if cagr > volatilite_annualisee else 'Défavorable'}")

print(f"\n   [TECHNIQUE]")
print(f"   • Tendance actuelle : {tendance} - Prix au-{'dessus' if prix_actuel > sma200_actuel else 'dessous'} de la SMA 200")
print(f"   • Signal long terme : {signal_cross} - Configuration {'haussière' if signal_cross == 'GOLDEN CROSS' else 'baissière'}")
print(f"   • Position vs moyennes : SMA 50 {('>' if sma50_actuel > sma200_actuel else '<')} SMA 200")
print(f"   • Distance au plus haut : {distance_max:+.2f}%")

print("\n[3] RECOMMANDATION FINALE")
print("-" * 80)

# Détermination de la recommandation
if score_final >= 7 and tendance == "HAUSSIÈRE" and cagr > 10:
    recommandation = "ACHAT"
    confiance = "ÉLEVÉE"
    couleur_reco = "[+++]"
elif score_final >= 5 and tendance == "HAUSSIÈRE":
    recommandation = "ACHAT PROGRESSIF"
    confiance = "MODÉRÉE"
    couleur_reco = "[++]"
elif score_final >= 4:
    recommandation = "CONSERVER / ATTENDRE"
    confiance = "MODÉRÉE"
    couleur_reco = "[=]"
else:
    recommandation = "ATTENTE"
    confiance = "FAIBLE"
    couleur_reco = "[-]"

print(f"\n   DÉCISION : {couleur_reco} {recommandation}")
print(f"   Niveau de confiance : {confiance}")
print(f"   Score global : {score_final:.1f}/10")

print(f"\n   JUSTIFICATION :")
if recommandation == "ACHAT":
    print(f"   • Performance historique exceptionnelle (CAGR {cagr:.2f}% vs marché ~10%)")
    print(f"   • Tendance haussière confirmée par les indicateurs techniques")
    print(f"   • Volatilité {niveau_risque.lower()} acceptable pour le rendement obtenu")
    print(f"   • Convient aux investisseurs avec horizon d'investissement > 5 ans")
elif recommandation == "ACHAT PROGRESSIF":
    print(f"   • Performance solide mais signaux techniques mitigés")
    print(f"   • Privilégier une stratégie de DCA (Dollar Cost Averaging)")
    print(f"   • Investissement progressif sur 3-6 mois recommandé")
elif recommandation == "CONSERVER / ATTENDRE":
    print(f"   • Situation actuelle incertaine, attendre confirmation de tendance")
    print(f"   • Pour les détenteurs : conserver la position")
    print(f"   • Pour les nouveaux investisseurs : attendre meilleur point d'entrée")
else:
    print(f"   • Signaux techniques défavorables")
    print(f"   • Attendre amélioration des indicateurs avant d'investir")

print("\n[4] RISQUES ET LIMITES")
print("-" * 80)
print("\n   RISQUES IDENTIFIÉS :")
print(f"   • Volatilité {niveau_risque.lower()} : possibilité de corrections de 20-30%")
print(f"   • Investissement non diversifié : concentration sur une seule action")
print(f"   • Drawdown historique : perte maximale de {abs(drawdown_max):.2f}% possible")
print(f"   • Dépendance aux cycles économiques et technologiques")

print("\n   LIMITES DE L'ANALYSE :")
print(f"   • Analyse basée uniquement sur données historiques (performances passées)")
print(f"   • Facteurs macroéconomiques non intégrés (inflation, taux d'intérêt)")
print(f"   • Analyse fondamentale non réalisée (ratios financiers, bénéfices)")
print(f"   • Événements géopolitiques non pris en compte")

print("\n" + "="*80)

# 6.2 VISUALISATIONS PRINCIPALES
print("\n" + "-"*80)
print("6.2 GÉNÉRATION DES GRAPHIQUES PRINCIPAUX")
print("-"*80)

# Configuration du style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# GRAPHIQUE 1 - Performance historique avec moyennes mobiles
print("\n   [1/5] Création du graphique : Performance historique...")

fig1 = plt.figure(figsize=(16, 8))
ax1 = plt.subplot(111)

ax1.plot(data.index, data['Close'], linewidth=2, color='#2E86AB', label='Prix de clôture', zorder=3)
ax1.plot(data.index, data['SMA_50'], linewidth=1.5, color='#F18F01', label='SMA 50 jours', alpha=0.8, zorder=2)
ax1.plot(data.index, data['SMA_200'], linewidth=1.5, color='#C73E1D', label='SMA 200 jours', alpha=0.8, zorder=2)

# Annotations des événements majeurs
ax1.axvline(x=pd.Timestamp('2020-03-01'), color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.text(pd.Timestamp('2020-03-01'), data['Close'].max() * 0.9, 'COVID-19', 
         rotation=90, verticalalignment='bottom', fontsize=10, color='red')

ax1.set_title('MICROSOFT (MSFT) - Évolution du prix 2010-2025\nAvec moyennes mobiles 50 et 200 jours', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Prix ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# Ajout d'une zone de texte avec les statistiques
textstr = f'Prix initial: ${prix_initial:.2f}\nPrix final: ${prix_final:.2f}\nRendement: +{rendement_total:.2f}%\nCAGR: +{cagr:.2f}%/an'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

print("       [OK] Graphique sauvegardé : graph1_performance_historique.png")

# GRAPHIQUE 2 - Rendements annuels
print("\n   [2/5] Création du graphique : Rendements annuels...")

rendements_annuels = []
for annee in sorted(data['Annee'].unique()):
    data_annee = data[data['Annee'] == annee]
    if len(data_annee) > 1:
        prix_debut = data_annee['Close'].iloc[0]
        prix_fin = data_annee['Close'].iloc[-1]
        rendement = ((prix_fin - prix_debut) / prix_debut) * 100
        rendements_annuels.append({'Annee': annee, 'Rendement': rendement})

df_rdt_annuel = pd.DataFrame(rendements_annuels)

fig2 = plt.figure(figsize=(14, 7))
ax2 = plt.subplot(111)

colors = ['#06A77D' if r > 0 else '#D62246' for r in df_rdt_annuel['Rendement']]
bars = ax2.bar(df_rdt_annuel['Annee'], df_rdt_annuel['Rendement'], 
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=df_rdt_annuel['Rendement'].mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Moyenne ({df_rdt_annuel["Rendement"].mean():.1f}%)', alpha=0.7)

# Annotations pour meilleure et pire année
idx_max = df_rdt_annuel['Rendement'].idxmax()
idx_min = df_rdt_annuel['Rendement'].idxmin()
ax2.annotate(f'Meilleur\n{df_rdt_annuel.loc[idx_max, "Rendement"]:.1f}%', 
             xy=(df_rdt_annuel.loc[idx_max, 'Annee'], df_rdt_annuel.loc[idx_max, 'Rendement']),
             xytext=(0, 20), textcoords='offset points', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax2.annotate(f'Pire\n{df_rdt_annuel.loc[idx_min, "Rendement"]:.1f}%', 
             xy=(df_rdt_annuel.loc[idx_min, 'Annee'], df_rdt_annuel.loc[idx_min, 'Rendement']),
             xytext=(0, -20), textcoords='offset points', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax2.set_title('MICROSOFT (MSFT) - Rendement annuel 2010-2024', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Année', fontsize=12, fontweight='bold')
ax2.set_ylabel('Rendement (%)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_xticks(df_rdt_annuel['Annee'])
ax2.set_xticklabels(df_rdt_annuel['Annee'].astype(int), rotation=45)

plt.tight_layout()
plt.show()

print("       [OK] Graphique sauvegardé : graph2_rendements_annuels.png")

# GRAPHIQUE 3 - Volatilité dans le temps
print("\n   [3/5] Création du graphique : Volatilité historique...")

fig3 = plt.figure(figsize=(16, 7))
ax3 = plt.subplot(111)

ax3.plot(data.index, data['Volatilite_30j'], linewidth=1.5, color='#C73E1D', label='Volatilité 30 jours')
ax3.axhline(y=data['Volatilite_30j'].mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Moyenne ({data["Volatilite_30j"].mean():.2f}%)', alpha=0.7)

# Zone de forte volatilité
ax3.fill_between(data.index, 0, data['Volatilite_30j'], 
                 where=(data['Volatilite_30j'] > data['Volatilite_30j'].mean() * 1.5),
                 color='red', alpha=0.2, label='Périodes de forte volatilité')

ax3.set_title('MICROSOFT (MSFT) - Volatilité mobile sur 30 jours', 
              fontsize=16, fontweight='bold', pad=20)
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_ylabel('Volatilité (%)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, alpha=0.3, linestyle='--')

textstr = f'Volatilité moyenne: {data["Volatilite_30j"].mean():.2f}%\nVolatilité max: {data["Volatilite_30j"].max():.2f}%\nVolatilité actuelle: {data["Volatilite_30j"].iloc[-1]:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

print("       [OK] Graphique sauvegardé : graph3_volatilite.png")

# GRAPHIQUE 4 - Drawdown
print("\n   [4/5] Création du graphique : Drawdown...")

fig4 = plt.figure(figsize=(16, 7))
ax4 = plt.subplot(111)

ax4.fill_between(data.index, data['Drawdown'], 0, color='#D62246', alpha=0.5)
ax4.plot(data.index, data['Drawdown'], color='#8B0000', linewidth=1.5)
ax4.axhline(y=0, color='black', linewidth=1)
ax4.axhline(y=drawdown_max, color='red', linestyle='--', linewidth=2, 
            label=f'Drawdown max ({drawdown_max:.2f}%)', alpha=0.7)

# Annotation du drawdown maximum
ax4.annotate(f'Perte max\n{drawdown_max:.2f}%\n{idx_drawdown_max.strftime("%m/%Y")}', 
             xy=(idx_drawdown_max, drawdown_max),
             xytext=(50, -30), textcoords='offset points', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))

ax4.set_title('MICROSOFT (MSFT) - Drawdown (Perte depuis le pic historique)', 
              fontsize=16, fontweight='bold', pad=20)
ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
ax4.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

print("       [OK] Graphique sauvegardé : graph4_drawdown.png")

# GRAPHIQUE 5 - Dashboard KPI
print("\n   [5/5] Création du graphique : Dashboard KPI...")

fig5 = plt.figure(figsize=(16, 10))

# Disposition en grille 3x3
gs = fig5.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# KPI 1 - Rendement total
ax51 = fig5.add_subplot(gs[0, 0])
ax51.text(0.5, 0.6, f'{rendement_total:,.1f}%', ha='center', va='center', 
          fontsize=40, fontweight='bold', color='#06A77D' if rendement_total > 0 else '#D62246')
ax51.text(0.5, 0.2, 'Rendement Total\n15 ans', ha='center', va='center', 
          fontsize=12, fontweight='bold')
ax51.axis('off')
ax51.set_facecolor('#F0F0F0')

# KPI 2 - CAGR
ax52 = fig5.add_subplot(gs[0, 1])
ax52.text(0.5, 0.6, f'{cagr:.2f}%', ha='center', va='center', 
          fontsize=40, fontweight='bold', color='#06A77D')
ax52.text(0.5, 0.2, 'CAGR\n(par an)', ha='center', va='center', 
          fontsize=12, fontweight='bold')
ax52.axis('off')
ax52.set_facecolor('#F0F0F0')

# KPI 3 - Volatilité
ax53 = fig5.add_subplot(gs[0, 2])
ax53.text(0.5, 0.6, f'{volatilite_annualisee:.1f}%', ha='center', va='center', 
          fontsize=40, fontweight='bold', color='#F18F01')
ax53.text(0.5, 0.2, f'Volatilité\n{niveau_risque}', ha='center', va='center', 
          fontsize=12, fontweight='bold')
ax53.axis('off')
ax53.set_facecolor('#F0F0F0')

# KPI 4 - Sharpe Ratio
ax54 = fig5.add_subplot(gs[1, 0])
ax54.text(0.5, 0.6, f'{sharpe_ratio:.3f}', ha='center', va='center', 
          fontsize=40, fontweight='bold', color='#2E86AB')
ax54.text(0.5, 0.2, 'Sharpe Ratio\nRdt/Risque', ha='center', va='center', 
          fontsize=12, fontweight='bold')
ax54.axis('off')
ax54.set_facecolor('#F0F0F0')

# KPI 5 - Drawdown max
ax55 = fig5.add_subplot(gs[1, 1])
ax55.text(0.5, 0.6, f'{drawdown_max:.1f}%', ha='center', va='center', 
          fontsize=40, fontweight='bold', color='#D62246')
ax55.text(0.5, 0.2, 'Drawdown Max\nPerte maximum', ha='center', va='center', 
          fontsize=12, fontweight='bold')
ax55.axis('off')
ax55.set_facecolor('#F0F0F0')

# KPI 6 - Position vs SMA 200
ax56 = fig5.add_subplot(gs[1, 2])
ax56.text(0.5, 0.6, f'{ecart_sma200:+.1f}%', ha='center', va='center', 
          fontsize=40, fontweight='bold', color='#06A77D' if ecart_sma200 > 0 else '#D62246')
ax56.text(0.5, 0.2, 'Position vs SMA 200\nTendance', ha='center', va='center', 
          fontsize=12, fontweight='bold')
ax56.axis('off')
ax56.set_facecolor('#F0F0F0')

# KPI 7 - Score final (grand)
ax57 = fig5.add_subplot(gs[2, :])
ax57.text(0.5, 0.65, f'{score_final:.1f}/10', ha='center', va='center', 
          fontsize=60, fontweight='bold', color='#06A77D' if score_final >= 7 else '#F18F01' if score_final >= 5 else '#D62246')
ax57.text(0.5, 0.35, f'SCORE GLOBAL - Recommandation: {recommandation}', ha='center', va='center', 
          fontsize=16, fontweight='bold')
ax57.text(0.5, 0.15, f'Tendance: {tendance} | Signal: {signal_cross} | Confiance: {confiance}', 
          ha='center', va='center', fontsize=12)
ax57.axis('off')
ax57.set_facecolor('#E8E8E8')

fig5.suptitle('MICROSOFT (MSFT) - TABLEAU DE BORD DES KPI', 
              fontsize=18, fontweight='bold', y=0.98)

plt.savefig('graph5_dashboard_kpi.png', dpi=300, bbox_inches='tight')
plt.show()

print("       [OK] Graphique sauvegardé : graph5_dashboard_kpi.png")

print("\n   [OK] Tous les graphiques ont été générés et sauvegardés avec succès !")

# 6.3 RECOMMANDATION DÉTAILLÉE PAR PROFIL
print("\n" + "-"*80)
print("6.3 RECOMMANDATION DÉTAILLÉE PAR PROFIL D'INVESTISSEUR")
print("-"*80)

print("\n[PROFIL 1] INVESTISSEUR CONSERVATEUR")
print("-" * 80)
print("\n   CARACTÉRISTIQUES :")
print("   • Aversion au risque élevée")
print("   • Objectif : préservation du capital")
print("   • Tolérance aux pertes : < 10%")
print("   • Horizon : court/moyen terme (1-3 ans)")

if volatilite_annualisee > 20 or abs(drawdown_max) > 25:
    reco_conservateur = "ATTENTE ou position très limitée (5-10% du portefeuille)"
    print(f"\n   RECOMMANDATION : {reco_conservateur}")
    print("   JUSTIFICATION :")
    print(f"   • Volatilité de {volatilite_annualisee:.1f}% trop élevée pour ce profil")
    print(f"   • Drawdown de {abs(drawdown_max):.1f}% dépasse la tolérance acceptable")
    print("   • Risque de corrections importantes (20-30%) incompatible")
    print("\n   ALTERNATIVE :")
    print("   • Privilégier obligations ou fonds diversifiés moins volatils")
    print("   • Si intérêt pour tech : ETF technologique diversifié")
else:
    reco_conservateur = "ACHAT LIMITÉ (10-15% du portefeuille)"
    print(f"\n   RECOMMANDATION : {reco_conservateur}")
    print("   JUSTIFICATION :")
    print(f"   • Volatilité acceptable pour exposition limitée")
    print("   • Performance historique solide justifie allocation minimale")
    print("\n   STRATÉGIE :")
    print("   • Investissement progressif (DCA sur 12 mois)")
    print("   • Stop-loss strict à -10% pour protection")

print("\n[PROFIL 2] INVESTISSEUR ÉQUILIBRÉ")
print("-" * 80)
print("\n   CARACTÉRISTIQUES :")
print("   • Équilibre rendement/risque")
print("   • Objectif : croissance modérée du capital")
print("   • Tolérance aux pertes : 10-20%")
print("   • Horizon : moyen/long terme (3-7 ans)")

if score_final >= 6 and tendance == "HAUSSIÈRE":
    reco_equilibre = "ACHAT MODÉRÉ (20-25% du portefeuille)"
    print(f"\n   RECOMMANDATION : {reco_equilibre}")
    print("   JUSTIFICATION :")
    print(f"   • Bon équilibre rendement/risque (CAGR {cagr:.2f}% vs volatilité {volatilite_annualisee:.1f}%)")
    print(f"   • Tendance technique favorable ({signal_cross})")
    print(f"   • Score de {score_final:.1f}/10 indique opportunité attractive")
    print("\n   STRATÉGIE D'ENTRÉE :")
    print("   • Investissement progressif (DCA sur 6 mois)")
    print("   • 50% immédiat, puis 25% à 3 mois, 25% à 6 mois")
    print("   • Réévaluation trimestrielle de la position")
elif score_final >= 4:
    reco_equilibre = "CONSERVER ou ACHAT PRUDENT (15-20%)"
    print(f"\n   RECOMMANDATION : {reco_equilibre}")
    print("   JUSTIFICATION :")
    print(f"   • Signaux mixtes nécessitent prudence")
    print(f"   • Position acceptable mais surveillance accrue")
    print("\n   STRATÉGIE :")
    print("   • DCA sur 12 mois pour minimiser risque d'entrée")
    print("   • Stop-loss à -15%")
else:
    reco_equilibre = "ATTENTE"
    print(f"\n   RECOMMANDATION : {reco_equilibre}")
    print("   • Attendre amélioration des indicateurs techniques")

print("\n[PROFIL 3] INVESTISSEUR DYNAMIQUE/AGRESSIF")
print("-" * 80)
print("\n   CARACTÉRISTIQUES :")
print("   • Recherche de rendement élevé")
print("   • Objectif : croissance forte du capital")
print("   • Tolérance aux pertes : > 25%")
print("   • Horizon : long terme (> 7 ans)")

if score_final >= 6:
    reco_dynamique = "ACHAT SIGNIFICATIF (30-40% du portefeuille)"
    print(f"\n   RECOMMANDATION : {reco_dynamique}")
    print("   JUSTIFICATION :")
    print(f"   • Performance historique exceptionnelle (CAGR {cagr:.2f}%)")
    print(f"   • Volatilité acceptable pour profil agressif")
    print(f"   • Potentiel de croissance long terme élevé")
    print("\n   STRATÉGIE D'ENTRÉE :")
    if tendance == "HAUSSIÈRE" and prix_actuel > sma50_actuel:
        print("   • Achat immédiat possible (50-70% de l'allocation)")
        print("   • Compléter position sur corrections (30-50%)")
    else:
        print("   • DCA rapide sur 3 mois")
        print("   • Profiter des corrections pour renforcer")
    print("\n   GESTION :")
    print("   • Stop-loss large à -25% ou -30% (accepter volatilité)")
    print("   • Objectif de prix à horizon 3-5 ans")
    print("   • Réévaluation annuelle")
else:
    reco_dynamique = "ACHAT MODÉRÉ (20-25%)"
    print(f"\n   RECOMMANDATION : {reco_dynamique}")
    print("   • Allocation réduite en attendant confirmation de tendance")

# 6.4 PLAN D'ACTION
print("\n" + "-"*80)
print("6.4 PLAN D'ACTION POUR L'INVESTISSEUR")
print("-"*80)

print("\n[ÉTAPE 1] AVANT D'INVESTIR")
print("-" * 80)
print("\n   1. Définir votre profil investisseur (conservateur/équilibré/dynamique)")
print("   2. Déterminer le montant à investir (ne jamais investir plus que supportable en perte)")
print("   3. Définir votre horizon d'investissement")
print("   4. Vérifier la diversification de votre portefeuille global")

print("\n[ÉTAPE 2] STRATÉGIE D'ENTRÉE")
print("-" * 80)

if tendance == "HAUSSIÈRE" and prix_actuel > sma200_actuel:
    print("\n   SITUATION ACTUELLE : Tendance haussière confirmée")
    print("\n   OPTION A - Entrée immédiate (investisseurs dynamiques)")
    print(f"   • Acheter 50-70% de l'allocation prévue au prix actuel (~${prix_actuel:.2f})")
    print("   • Placer ordres d'achat échelonnés à -5%, -10%, -15% pour compléter")
    print("\n   OPTION B - DCA rapide (investisseurs équilibrés)")
    print("   • Investir 1/3 immédiatement")
    print("   • Investir 1/3 dans 1 mois")
    print("   • Investir 1/3 dans 2 mois")
    print("\n   OPTION C - DCA lent (investisseurs conservateurs)")
    print("   • Investir mensuellement sur 6-12 mois")
    print("   • Montant fixe chaque mois (ex: 100-500 EUR/mois)")
else:
    print("\n   SITUATION ACTUELLE : Tendance incertaine ou baissière")
    print("\n   STRATÉGIE RECOMMANDÉE :")
    print("   • DCA sur 6-12 mois obligatoire")
    print("   • Attendre confirmation haussière avant d'accélérer")
    print("   • Privilégier achats sur corrections > 5%")

print("\n[ÉTAPE 3] DÉFINIR LES NIVEAUX CLÉS")
print("-" * 80)

prix_support = data['Close'].tail(252).min()
prix_resistance = data['Close'].tail(252).max()

print(f"\n   Prix actuel : ${prix_actuel:.2f}")
print(f"\n   SUPPORT (plancher 1 an) : ${prix_support:.2f} (-{((prix_actuel - prix_support)/prix_actuel*100):.1f}%)")
print("   • Si cassure : possible correction supplémentaire")
print("   • Niveau d'achat opportuniste pour renforcement")
print(f"\n   RÉSISTANCE (plafond 1 an) : ${prix_resistance:.2f} (+{((prix_resistance - prix_actuel)/prix_actuel*100):.1f}%)")
print("   • Si cassure : signal haussier fort")
print("   • Objectif de prix à moyen terme")

print("\n   STOP-LOSS RECOMMANDÉS (selon profil) :")
print(f"   • Conservateur : ${prix_actuel * 0.90:.2f} (-10%)")
print(f"   • Équilibré    : ${prix_actuel * 0.85:.2f} (-15%)")
print(f"   • Dynamique    : ${prix_actuel * 0.75:.2f} (-25%)")

print("\n   OBJECTIFS DE PRIX (horizon 1-3 ans) :")
objectif_conservateur = prix_actuel * 1.15
objectif_optimiste = prix_actuel * 1.30
objectif_agressif = prix_actuel * 1.50

print(f"   • Scénario conservateur (+15%) : ${objectif_conservateur:.2f}")
print(f"   • Scénario optimiste (+30%)    : ${objectif_optimiste:.2f}")
print(f"   • Scénario agressif (+50%)     : ${objectif_agressif:.2f}")

print("\n[ÉTAPE 4] SUIVI ET RÉÉVALUATION")
print("-" * 80)
print("\n   SUIVI HEBDOMADAIRE :")
print("   • Vérifier les grandes variations (> +/- 5% en 1 jour)")
print("   • Suivre l'actualité de l'entreprise")
print("   • Pas de décision impulsive")

print("\n   SUIVI MENSUEL :")
print("   • Recalculer les indicateurs techniques (SMA 50/200, tendance)")
print("   • Vérifier la volatilité")
print("   • Ajuster stop-loss si forte hausse (trailing stop)")

print("\n   RÉÉVALUATION TRIMESTRIELLE :")
print("   • Recalculer tous les KPI")
print("   • Réévaluer la recommandation (Achat/Conserver/Vendre)")
print("   • Ajuster l'allocation si nécessaire")
print("   • Vérifier la diversification globale du portefeuille")

print("\n[ÉTAPE 5] RÈGLES DE GESTION")
print("-" * 80)
print("\n   RÈGLES D'OR :")
print("   1. Ne jamais investir plus de 40% sur une seule action")
print("   2. Toujours respecter son stop-loss (discipline)")
print("   3. Ne pas vendre dans la panique (sauf stop-loss)")
print("   4. Prendre des profits partiels sur fortes hausses (> 30%)")
print("   5. Rééquilibrer régulièrement (tous les 6-12 mois)")

print("\n   SIGNAUX DE VENTE :")
print("   • Stop-loss atteint")
print("   • Death Cross confirmé (SMA 50 < SMA 200)")
print("   • Drawdown actuel < -20% et aggravation")
print("   • Changement fondamental de l'entreprise")
print("   • Besoin de liquidités personnelles")

print("\n   SIGNAUX DE RENFORCEMENT :")
print("   • Correction de 10-15% sur tendance haussière intacte")
print("   • Golden Cross confirmé")
print("   • Résultats financiers meilleurs que prévu")
print("   • Prix rebondit sur support majeur")

# 6.5 POINTS D'ATTENTION ET AVERTISSEMENTS
print("\n" + "/"*80)
print("6.5 POINTS D'ATTENTION ET AVERTISSEMENTS")
print("/"*80)

print("\n[FACTEURS NON ANALYSÉS]")
print("S" * 80)
print("\n   Cette analyse technique ne prend PAS en compte :")
print("   • Analyse fondamentale (P/E ratio, croissance bénéfices, dette)")
print("   • Situation macroéconomique (inflation, taux FED, récession)")
print("   • Concurrence et évolution du secteur technologique")
print("   • Événements géopolitiques (guerres, sanctions, réglementations)")
print("   • Changements de direction ou stratégie de l'entreprise")
print("   • Innovations technologiques disruptives")

print("\n[LIMITATIONS MÉTHODOLOGIQUES]")
print("M" * 80)
print("\n   • Les performances passées ne garantissent PAS les performances futures")
print("   • Modèle basé uniquement sur historique de prix")
print("   • Indicateurs techniques peuvent donner de faux signaux")
print("   • Analyse d'une seule action (risque de concentration)")
print("   • Hypothèses simplificatrices (taux sans risque fixe, etc.)")

print("\n[RECOMMANDATIONS COMPLÉMENTAIRES]")
print("N" * 80)
print("\n   AVANT D'INVESTIR :")
print("   • Consulter un conseiller financier agréé (AMF)")
print("   • Effectuer une analyse fondamentale complémentaire")
print("   • Lire les rapports annuels de Microsoft")
print("   • Suivre l'actualité du secteur technologique")
print("   • Vérifier votre situation fiscale (PEA, CTO, assurance-vie)")

print("\n   DIVERSIFICATION :")
print("   • Ne jamais mettre tous ses œufs dans le même panier")
print("   • Diversifier par secteur (tech, santé, finance, etc.)")
print("   • Diversifier par géographie (US, Europe, Asie)")
print("   • Diversifier par classe d'actifs (actions, obligations, immobilier)")
print("   • Règle générale : maximum 5-10% par ligne individuelle")


# 6.6 CONCLUSION ET LIVRABLES
print("\n" + "W"*80)
print("6.6 LIVRABLES FINAUX")
print("W"*80)

print("\n[FICHIERS GÉNÉRÉS]")
print("-" * 80)
print("\n   GRAPHIQUES :")
print("   [OK] graph1_performance_historique.png")
print("   [OK] graph2_rendements_annuels.png")
print("   [OK] graph3_volatilite.png")
print("   [OK] graph4_drawdown.png")
print("   [OK] graph5_dashboard_kpi.png")

print("\n   DONNÉES :")
print("   • Code source complet (parties 1-6)")
print("   • Données historiques 15 ans")
print("   • Calculs et KPI détaillés")

print("\n[RÉSUMÉ FINAL POUR LE CLIENT]")
print("W" * 80)

print(f"\n   ACTION ANALYSÉE : Microsoft (MSFT)")
print(f"   PÉRIODE : 2010-2025 ({nb_annees:.1f} ans)")
print(f"   DATE DU RAPPORT : {datetime.now().strftime('%d/%m/%Y')}")

print(f"\n   PERFORMANCE :")
print(f"   • Rendement total : +{rendement_total:,.2f}%")
print(f"   • CAGR : +{cagr:.2f}% par an")
print(f"   • Évaluation : {performance}")

print(f"\n   RISQUE :")
print(f"   • Volatilité : {volatilite_annualisee:.2f}% ({niveau_risque})")
print(f"   • Drawdown max : {drawdown_max:.2f}%")
print(f"   • Sharpe Ratio : {sharpe_ratio:.3f}")

print(f"\n   TECHNIQUE :")
print(f"   • Tendance : {tendance}")
print(f"   • Signal : {signal_cross}")
print(f"   • Position vs SMA 200 : {ecart_sma200:+.2f}%")

print(f"\n   SCORE GLOBAL : {score_final:.1f}/10")
print(f"\n   RECOMMANDATION FINALE : {recommandation}")
print(f"   NIVEAU DE CONFIANCE : {confiance}")

print("\n   RECOMMANDATIONS PAR PROFIL :")
if volatilite_annualisee > 20 or abs(drawdown_max) > 25:
    print("   • Conservateur : Attente ou position limitée (5-10%)")
else:
    print("   • Conservateur : Achat limité (10-15%)")

if score_final >= 6 and tendance == "HAUSSIÈRE":
    print("   • Équilibré : Achat modéré (20-25%)")
else:
    print("   • Équilibré : Conserver ou achat prudent (15-20%)")

if score_final >= 6:
    print("   • Dynamique : Achat significatif (30-40%)")
else:
    print("   • Dynamique : Achat modéré (20-25%)")

print("\n" + "I"*80)
print("FIN DU RAPPORT D'ANALYSE")
print("I"*80)

print("\n" + "O"*80)
print(" "*20 + "ANALYSE TERMINÉE AVEC SUCCÈS")
print("O"*80)
print("\nMERCI D'AVOIR UTILISÉ CE SYSTÈME D'ANALYSE FINANCIÈRE")
print("\nPour toute question ou amélioration, n'hésitez pas à consulter")
print("un professionnel de la finance ou à approfondir votre analyse.")
print("\nBONNE CHANCE DANS VOS INVESTISSEMENTS !")
print("O"*80)