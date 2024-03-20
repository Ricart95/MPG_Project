import warnings
import streamlit as st
import sklearn
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from sys import displayhook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics

import joblib
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingClassifier
import plotly.figure_factory as ff
import plotly.express as px

ligue1 = pd.read_csv("data_ligue1_new.csv", sep=";",
                     index_col=0, encoding='latin-1')
premier_league = pd.read_csv("premier_league.csv", sep=";",
                             index_col=0, encoding='latin')
serieA = pd.read_csv("serieA.csv", sep=";",
                     index_col=0, encoding='latin-1')
liga = pd.read_csv("liga.csv", sep=";",
                   index_col=0, encoding='latin-1')
ligue2 = pd.read_csv("ligue2.csv", sep=";",
                     index_col=0, encoding='latin-1')

ligue1 = ligue1.drop(["j17", "j16", "j15", 'j18', 'j19'], axis=1)
ligue2 = ligue2.drop(["j17", "j16", "j15", 'j18', 'j19'], axis=1)
serieA = serieA.drop(["j17", "j16", "j15", 'j18', 'j19'], axis=1)
liga = liga.drop(["j17", "j16", "j15", 'j18', 'j19'], axis=1)
premier_league = premier_league.drop(
    ["j17", "j16", "j15", 'j18', 'j19'], axis=1)

df = pd.concat([ligue1, premier_league, serieA, liga, ligue2])
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

colonnes_a_supprimer = ['Dispo@MPGLaurent?',
                        'Victoire probable', 'Date', 'Prochain opposant', 'Club']

df2 = df.drop(columns=colonnes_a_supprimer, axis=1)

liste_colonnes2 = df2.columns.tolist()

df3 = df2.fillna(0)

df3 = df3.replace(',', '.', regex=True)
df3["Note"] = df3["Note"].astype(float)
data_0 = df3[df3['Note'] > 0]['Note']
top_10_dc = df3[df3["Poste"] == "DC"].nlargest(10, "Note")
top_10_mo = df3[df3["Poste"] == "MO"].nlargest(10, "Note")
top_10_md = df3[df3["Poste"] == "MD"].nlargest(10, "Note")
top_10_a = df3[df3["Poste"] == "A"].nlargest(10, "Note")
top_10_dl = df3[df3["Poste"] == "DL"].nlargest(10, "Note")
top_10_g = df3[df3["Poste"] == "G"].nlargest(10, "Note")


bins = [0, 0.1, 4.9, 5.5, 9]
labels = ['D', 'C', 'B', 'A']
df3['j14'] = pd.to_numeric(df3['j14'], errors='coerce')
df3['j14'] = pd.cut(df3['j14'], bins=bins, labels=labels, include_lowest=True)


df_attaquants = df3[df3['Poste'] == 'A']
df_milieux = df3[df3['Poste'].isin(['MO', 'MD'])]
df_défenseurs = df3[df3['Poste'].isin(['DC', 'DL'])]
df_gardiens = df3[df3['Poste'] == 'G']

warnings.filterwarnings('ignore')

# Conversion des valeurs de la colonne "Note" en type numérique
df_attaquants['Note'] = pd.to_numeric(df_attaquants['Note'], errors='coerce')
df_milieux['Note'] = pd.to_numeric(df_milieux['Note'], errors='coerce')
df_défenseurs['Note'] = pd.to_numeric(df_défenseurs['Note'], errors='coerce')
df_gardiens['Note'] = pd.to_numeric(df_gardiens['Note'], errors='coerce')
# Calculer le 80 ème quantile en vue de ne filtrer que les joueurs faisant partis des 20% les mieux notés à leurs postes
noteQ80A = df_attaquants['Note'].quantile(0.8)
noteQ80M = df_milieux['Note'].quantile(0.8)
noteQ80D = df_défenseurs['Note'].quantile(0.8)
noteQ80G = df_gardiens['Note'].quantile(0.8)

st.sidebar.title("Sommaire")
pages = ["Contexte du projet", "Exploration des données",
         "Analyse de données", "Modélisation", "Stratégies", "Remerciements"]

selected_option = st.sidebar.radio('Sélectionnez une page', pages)

if selected_option == "Contexte du projet":
    st.title(" Contexte du projet")

    st.image('guide-mpg-mon-petit-gazon.jpg')

    st.subheader("MPG, qu’est ce que c’est ?")
    st.write("MPG est l'acronyme de 'Mon Petit Gazon', un jeu de fantasy football en ligne populaire en France. Dans ce jeu, les participants forment leur propre équipe de football en sélectionnant des joueurs réels de différentes ligues de football européennes. Les performances des joueurs dans les matchs réels sont ensuite converties en points en fonction de critères prédéfinis (comme les buts marqués, les passes décisives, les clean sheets pour les défenseurs et les gardiens, etc.). Les participants du jeu compétitionnent entre eux en fonction des performances de leurs joueurs dans les matchs réels.")
    st.image("mpg_site.PNG", caption="Site MPG")
    st.text("")
    st.text("")
    st.header("LES MEMBRES DE LA TEAM")
    st.subheader("Coach : YAZID")

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        st.subheader('LIONEL')

    with col_2:
        st.subheader('MATTHIEU')

    with col_3:
        st.subheader('BENOIT')
    st.markdown(
        "<h1 style='text-align: center;</h1>", unsafe_allow_html=True)

    st.image("ekip.gif", use_column_width=True)


elif selected_option == "Exploration des données":
    st.title("Exploration des données")
    st.header("1. Objectif")
    st.markdown("""
Notre projet MPG se base donc sur le jeu de simulation de fantasy league Mon Petit Gazon. Nous avons accès aux différentes données des ligues principales de football européen. Notre but est donc d’analyser les différentes données en notre possession pour prédire les performances des joueurs sur un match donné à travers leurs performances lors des matchs précédents.

Grâce aux prédictions de performances, nous pourrons donc développer des fonctions permettant d’adopter différentes stratégies de composition pour les matchs, mais aussi des stratégies de recrutement bien précises lors du mercato.""")

    st.header("2. Cadre de l'exploration")
    st.markdown("Le jeu de données que nous avons utilisé est une fusion des jeu de données des 5 ligues suivantes :\n-	Ligue 1\n-	Ligue 2\n-	Premier league\n-	SerieA\n-	Liga")
    image_liga = "laliga_logo.jfif"
    image_ligue2 = "logo-ligue-2.jpg"
    image_epl = "epl.png"
    image_ligue1 = "Ligue-1-Logo.png"
    image_seriea = "serie-a-logo.png"

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(image_ligue1, use_column_width=True)

    with col2:
        st.image(image_ligue2, use_column_width=True)

    with col3:
        st.image(image_epl, use_column_width=True)

    with col4:
        st.image(image_seriea, use_column_width=True)

    with col5:
        st.image(image_liga, use_column_width=True)

    st.text("")
    st.subheader("Mise a jour des données")
    st.markdown("""
Nous avons initialement choisi de nous concentrer sur le jeu de données de la Ligue 1 dans MPG, mais nous avons rapidement réalisé que cela ne fournirait pas suffisamment de données pour nos futurs algorithmes de Machine Learning, avec seulement environ 500 entrées correspondant au nombre de joueurs de la ligue. Nous avons donc importé les données de toutes les autres ligues et les avons fusionnées. Cependant, cette fusion a introduit de nouvelles colonnes en raison des matchs de football joués entre-temps, nécessitant la suppression des colonnes supplémentaires pour aligner les données avec la dernière journée de la Ligue 1, soit la "j14".
""")
    st.text("")
    st.markdown("Les données que nous récupérons directement depuis le site de MPG se mettent à jours toutes les semaines mais nous avons choisi de garder uniquement les données à partir de notre première utilisation de ces dernières.\n Il y a évidemment eu plusieurs étapes de préparation pour parvenir à notre jeu de données final :\n-	Fusion des Dataframes de toutes les ligues\n-	Transformation des virgules en points pour tous les chiffres à virgules\n-	Changement des types de colonnes\n-	Remplacement de tous les NaNs en 0\n-	Etablissement d'intervalles de notes en variables explicatives A, B, C et D")
    st.write("")

    st.write("")
    st.write(
        "Le Dataframe suivant est celui que nous avons obtenu à la suite de ces actions :")
    st.subheader("Dataframe Final")
    st.dataframe(df3.head(10))
    st.write("Taille du Dataframe :", df3.shape)
    st.write("")
    st.subheader("Informations importantes au sujet du Dataframe")
    st.dataframe(df3.describe())

    if st.checkbox("Afficher les NA du Dataframe de base"):
        st.dataframe(df.isna().sum())


elif selected_option == "Analyse de données":
    st.title(" Analyse des données")
    st.markdown(
        "Dans le cadre de l'analyse de nos données, nous avons décidé de créer de la Dataviz afin de pouvoir en tirer des potentielles tendances nous permettant d'adapter nos futurs Stratégies.")

    st.subheader("Top 10 des meilleurs joueurs par poste")
    expander = st.expander("Voir les explications")
    expander.write(
        "Un rapide aperçu des meilleurs joueurs que pourront être amené à recruter ou à surveiller lors de notre phase de mercato. On peut aussi voir rapidement quelles notes ont les meilleurs joueurs MPG.")
    options = ["TOP 10 Meilleurs Attaquants", "TOP 10 Meilleurs Defenseur", "TOP 10 Meilleurs Millieux Offensifs",
               "TOP 10 Meilleurs Millieux Defensifs",
               "TOP 10 Meilleurs Lateraux", "TOP 10 Meilleurs Gardiens"]

    selected_option = st.selectbox("Sélectionnez un classement :", options)

    fig, ax = plt.subplots(figsize=(35, 20))

    if selected_option == "TOP 10 Meilleurs Attaquants":
        ax.barh(top_10_a.index, top_10_a["Note"])
        ax.set_xlim(5, 7.6)
        ax.set_title(selected_option)
        ax.set_yticks(top_10_a.index)
        ax.set_yticklabels(top_10_a.index, fontsize=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.set_xlabel("Notes", fontsize=40)
        ax.set_ylabel("Joueurs", fontsize=40)
        fig.tight_layout()
        st.pyplot(fig)
        st.image("mbaporshe.gif")
    elif selected_option == "TOP 10 Meilleurs Defenseur":
        ax.barh(top_10_dc.index, top_10_dc["Note"])
        ax.set_xlim(5, 7.6)
        ax.set_title(selected_option)
        ax.set_yticks(top_10_dc.index)
        ax.set_yticklabels(top_10_dc.index, fontsize=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.set_xlabel("Notes", fontsize=40)
        ax.set_ylabel("Joueurs", fontsize=40)
        fig.tight_layout()
        st.pyplot(fig)
        st.image("marqui.gif")
    elif selected_option == "TOP 10 Meilleurs Millieux Offensifs":
        ax.barh(top_10_mo.index, top_10_mo["Note"])
        ax.set_xlim(5, 7.6)
        ax.set_title(selected_option)
        ax.set_yticks(top_10_mo.index)
        ax.set_yticklabels(top_10_mo.index, fontsize=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.set_xlabel("Notes", fontsize=40)
        ax.set_ylabel("Joueurs", fontsize=40)
        fig.tight_layout()
        st.pyplot(fig)
        st.image("bellingham.gif")
    elif selected_option == "TOP 10 Meilleurs Millieux Defensifs":
        ax.barh(top_10_md.index, top_10_md["Note"])
        ax.set_xlim(5, 7.6)
        ax.set_title(selected_option)
        ax.set_yticks(top_10_md.index)
        ax.set_yticklabels(top_10_md.index, fontsize=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.set_xlabel("Notes", fontsize=40)
        ax.set_ylabel("Joueurs", fontsize=40)
        fig.tight_layout()
        st.pyplot(fig)
        st.image("rodri.gif")
    elif selected_option == "TOP 10 Meilleurs Lateraux":
        ax.barh(top_10_dl.index, top_10_dl["Note"])
        ax.set_xlim(5, 7.6)
        ax.set_title(selected_option)
        ax.set_yticks(top_10_dl.index)
        ax.set_yticklabels(top_10_dl.index, fontsize=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.set_xlabel("Notes", fontsize=40)
        ax.set_ylabel("Joueurs", fontsize=40)
        fig.tight_layout()
        st.pyplot(fig)
        st.image("federico.gif")

    elif selected_option == "TOP 10 Meilleurs Gardiens":
        ax.barh(top_10_g.index, top_10_g["Note"])
        ax.set_xlim(5, 7.6)
        ax.tick_params(axis='x', labelsize=40)
        ax.set_xlabel("Notes", fontsize=40)
        ax.set_ylabel("Joueurs", fontsize=40)
        ax.set_title(selected_option)
        ax.set_yticks(top_10_g.index)
        ax.set_yticklabels(top_10_g.index, fontsize=40)
        fig.tight_layout()
        st.pyplot(fig)
        st.image("sakho.gif")

    st.subheader("Distribution des notes par postes")
    expander = st.expander("Voir les explications")
    expander.write(
        "À première vue, la distribution semble suivre une loi normale. En effet, la distribution est quasiment symétrique et se concentre autour de la note de 5. Nous remarquons également un grand nombre de zéros dans notre jeu de données, mais cela ne remet pas en cause notre hypothèse.")
    st.image('frequence_notes_postes.png')
    st.text("")

    st.subheader("Q-QPlot de la distribution des notes")
    expander = st.expander("Voir les explications")
    expander.write(
        "On constate que les variables suivent bien la loi normale, car les quantiles des ensembles théoriques et empiriques semblent suivre la même distribution. On peut clairement voir que les points sont alignés avec la courbe.")
    st.image('Q-QPlot.png')
    st.text("")

    st.subheader("Répartition des notes par postes")
    expander = st.expander("Voir les explications")
    expander.write("On constate ici que, en fonction des postes, la répartition des quartiles et le nombre de valeurs extrêmes seront différents. On pourrait penser qu'il est plus facile pour les gardiens d'obtenir une bonne note, mais cela n'est pas forcément vrai, car leur nombre étant inférieur à celui des attaquants, par exemple, il est plus facile pour ce groupe de s'éloigner de la médiane globale.")
    st.image('notes_par_postes.png')
    st.text("")

    st.subheader(
        "Pourcentage d'achat par rapport au ratio Note/Cote par poste")
    expander = st.expander("Voir les explications")
    expander.write(

        "De manière intuitive, on pourrait penser qu'il y a une corrélation entre la note d'un joueur et sa cote. Plus un joueur est fort, plus son prix sera élevé. C'est bien la tendance qui se dégage de ce graphique ; cependant, on peut voir qu'en fonction du poste, cela ne se vérifie pas de manière systématique. Sur le poste des gardiens notamment, il y a plusieurs joueurs avec de bonnes notes et des cotes faibles. Par ailleurs, le pourcentage d'achat suit la même logique : plus un joueur a une bonne note, plus il sera souvent acheté, mais il y a encore une fois des exceptions. On peut donc penser qu'il y a des joueurs très bien notés mais, du fait de leur faible notoriété, qui sont peu achetés et peu côtés. Il faudra donc surveiller ces joueurs lors du mercato afin de pouvoir potentiellement créer une bonne équipe homogène sans trop dépenser.")
    st.image('cote_note_achat.png')
    st.text("")

    st.subheader(
        'Nuage de points représentant la note des joueurs en fonction de la variation de leur note')
    expander = st.expander("Voir les explications")
    expander.write(
        "Nous avons estimé qu’à note équivalente, il valait mieux prendre un joueur avec des performances plus régulières et donc une variation de note moins importante. Les joueurs qui nous intéresse sur ce graphique sont donc ceux situé dans la partie supérieure gauche. ")
    fig = px.scatter(df3, x="Variation", y="Note", color="% achat", facet_col="Poste", facet_col_wrap=3,
                     labels={"Variation": "Variation", "Note": "Note"})
    fig.update_xaxes(title_text="Variation")
    fig.update_yaxes(title_text="Note")
    st.plotly_chart(fig)

    df5 = df3[df3["Note"] > 0]
    st.subheader("Pourcentage d'achat des joueurs en fonction de leur note")
    fig = px.scatter(df5, x="Note", y="% achat", trendline="ols",
                     title="Pourcentage d'achat des joueurs en fonction de leur note")
    st.plotly_chart(fig)

    cols = ["Cote", "But", "Tps moy", "Cleansheet", "Pass decis.", "Tirs", "Tirs cadrés", "Corner gagné",
            "%Passes", "Ballons", "Interceptions", "Tacles", "%Duel", "Fautes", "Centres", "Centres ratés",
            "Ballon perdu", "Passe>Tir", "Diff de buts", "Grosse occas manquée", "Balle non rattrapée", "Note"]

    st.subheader("Matrice de corrélation")
    expander = st.expander("Voir les explications")
    expander.write(
        "Une matrice de corrélation permet de voir rapidement les liens qui existent entre les différentes variables et permet de valider ou de réfuter certaines de hypothèses et nous aide ainsi à préparer d’autres graphiques.")
    st.image('heatmap.png')
    st.text("")

    st.subheader(
        "Répartition des notes sur la Nième journée de championnat")
    expander = st.expander("Voir les explications")
    expander.write(
        "Ce countplot permet de vérifier rapidement si en enlevant nos valeurs aberrantes (les 0) la distribution des notes suit une loi distribution normale, nous observons rapidement que cela à l’air d’être le cas.")
    df3["j12"] = pd.to_numeric(df3["j12"], errors="coerce")
    dfj12 = df3[df3["j12"] > 0]
    fig = px.histogram(
        dfj12, x="j12")
    fig.update_xaxes(categoryorder="total descending")

    st.plotly_chart(fig)


elif selected_option == "Modélisation":
    st.title(" Modélisation")
    st.header("1. Étapes du processus de Modélisation")
    st.markdown("""

    Initialement, nous avons exploré des algorithmes de régression, tels que DecisionTreeRegressor(), pour prédire les notes des joueurs à la prochaine journée j15. Cependant, cette approche s'est révélée peu fructueuse, produisant des scores tendant vers 0 voire négatifs. Nous avons alors opté pour des algorithmes de classification. Pour ce faire, nous avons défini des intervalles de notes, représentés par des lettres, créant ainsi des classes :

    - A = [6,9]
    - B = [5,6]
    - C = [4,5]
    - D = [2.5,4]
    - E = [0,2.5]

    Malgré le bon fonctionnement initial de nos algorithmes de classification, nous avons remarqué des déséquilibres dans les classes lors de l'affichage du classification report.

    Après analyse, nous avons réduit nos intervalles à 4 :

    - A = [5.5,9]
    - B = [4.9,5.5]
    - C = [0.1,4.9]
    - D = [0,0.1]
                
    La présence importante de joueurs avec une note de 0 a conduit à une surreprésentation de l'intervalle D.

    La réduction à 4 intervalles a permis d'atténuer cet écart entre les classes A, B, C et D, privilégiant ainsi le recall comme métrique de performance pour notre projet, en raison de sa capacité à détecter les vrais positifs.
                    """)
    st.header("2. Modélisation")

    feats = df3.drop('j14', axis=1)
    target = df3['j14']
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.25)
    X_train['Poste'] = X_train['Poste'].astype(str)
    X_test['Poste'] = X_test['Poste'].astype(str)
    le = LabelEncoder()
    X_train['Poste'] = le.fit_transform(X_train['Poste'])
    X_test['Poste'] = le.transform(X_test['Poste'])

    def train_model(classifier, X_train, y_train):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'Gradient Boosting':
            clf = GradientBoostingClassifier()
        elif classifier == 'KNN':
            clf = KNeighborsClassifier()
        elif classifier == "Decision Tree":
            clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(X_train, y_train)
        return clf

    def evaluate_model(clf, X_test, y_test, choice):
        if choice == "Accuracy":
            return clf.score(X_test, y_test)
        elif choice == "Confusion matrix":
            return confusion_matrix(y_test, clf.predict(X_test))

    choix = ["Random Forest", "Gradient Boosting", 'KNN', "Decision Tree"]
    option = st.selectbox("Choix du modèle", choix, key='unique_key_1')
    st.write("Le modèle choisi est:", option)

    if option:
        clf = train_model(option, X_train, y_train)

        display = st.radio('Que souhaitez-vous montrer ?',
                           ('Accuracy', 'Confusion matrix', "Classification report"), key='unique_key_2')
        if display == 'Accuracy':
            st.write(evaluate_model(clf, X_test, y_test, display))
        elif display == 'Confusion matrix':
            st.dataframe(evaluate_model(clf, X_test, y_test, display))
        elif display == 'Classification report':
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred)
            st.text(report)

    st.text("")

    st.markdown(
        "Aux vues de nos résultats, le modèle que nous allons retenir est le Gradient Boosting Classifier.")
    st.markdown(
        "En appliquant cette méthode, nous obtenons les résultats suivants : ")
    st.code("Score du modèle :  0.6781437125748503")
    st.code("""Matrice de Confusion :
[[ 42  28  20   6]
 [ 27  83  27  24]
 [ 16  28  49  25]
 [  3  20  11 259]]""")

    st.code("""precision    recall  f1-score   support

           A       0.48      0.44      0.46        96
           B       0.52      0.52      0.52       161
           C       0.46      0.42      0.44       118
           D       0.82      0.88      0.85       293

    accuracy                           0.65       668
   macro avg       0.57      0.56      0.57       668
weighted avg       0.64      0.65      0.64       668""")

elif selected_option == "Stratégies":
    st.title(" Stratégies")
    st.markdown("Le moment du Mercato est arrivé, nous vous proposons d'utiliser nos différents outils afin de créer la meilleure des équipes du championnat 💪👑")

    def afficher_dataframe_selon_selection(noms_dataframes, dataframes):
        dataframe_selectionne = st.selectbox(
            "Sélectionnez un DataFrame :", noms_dataframes)
        if dataframe_selectionne:
            st.write(f"Affichage du DataFrame '{dataframe_selectionne}' :")
            st.dataframe(dataframes[dataframe_selectionne])

    dataframes = {"Ligue 1": ligue1, "Ligue 2": ligue2,
                  "Premier League": premier_league, "Serie A": serieA, "Liga": liga}
    noms_dataframes = list(dataframes.keys())
    afficher_dataframe_selon_selection(noms_dataframes, dataframes)

    gbc = joblib.load("model_gbs")
    df4 = df3.drop("j14", axis=1)
    df4['Poste'] = df4['Poste'].astype(str)
    le = LabelEncoder()
    df4['Poste'] = le.fit_transform(df4['Poste'])
    df4 = df4.replace(',', '.', regex=True)
    y_pred_gbs_all = gbc.predict(df4)
    len(y_pred_gbs_all)

    def classement_prédictif(df3, y_pred_gbs_all, joueurs):
        classement = {poste: {} for poste in df3['Poste'].unique()}

        for joueur in joueurs:
            if joueur in df3.index:
                prédiction = y_pred_gbs_all[df3.index.get_loc(joueur)]
                poste_joueur = df3.loc[joueur, 'Poste']
                classement[poste_joueur][joueur] = prédiction
            else:
                st.warning(
                    f"Le joueur {joueur} n'est pas présent dans l'index du DataFrame.")

        for poste, joueurs_poste in classement.items():
            classement[poste] = dict(
                sorted(joueurs_poste.items(), key=lambda item: item[1]))

        return classement

    def main():
        st.subheader("Classement Prédictif des Joueurs")
        st.markdown(
            "Nous avons développé un outil capable de prédire, à l'aide de notre algorithme, le classement de joueurs sélectionnés.")
        joueurs_test = st.text_input(
            "Entrez les noms des joueurs séparés par des virgules:")
        joueurs_test = [j.strip() for j in joueurs_test.split(',')]

        if st.button("Obtenir le Classement"):
            classement = classement_prédictif(
                df3, y_pred_gbs_all, joueurs_test)
            st.subheader("Classement Trié par Prédiction")
            for poste, joueurs_poste in classement.items():
                st.write(f"Poste: {poste}")
                for joueur, prédiction in joueurs_poste.items():
                    st.write(f"Joueur : {joueur}, Prédiction : {prédiction}")

    if __name__ == "__main__":
        main()

    df3['Note'] = df3['Note'].astype(float)

    def mercato_homogène(nb_joueurs, championnat, nb_participants):
        #  nb_joueurs = Choix du nombre de joueurs à acheter # Choisir entre 18, 20 et 22
        # championnat = Choix du championnat # Choisir entre "Ligue 1", "Ligue 2", "Premier League", "Liga" et "SerieA"
        # nb_participants = Nombre de participants à la ligue MPG # Chosiir entre 6, 8 et 10
        joueurs_a_acheter = []
        joueurs_retires = []
        if nb_joueurs == 18:
            if championnat == "Ligue 1":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calcul de la somme initiale des valeurs de la colonne "Q3 à 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 à 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calcul de la somme initiale des valeurs de la colonne "Q3 à 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Liga":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "SerieA":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Premier League":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Ligue 2":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:6])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:4])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            else:
                print("Championnat invalide")

        elif nb_joueurs == 20:
            if championnat == "Ligue 1":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Liga":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "SerieA":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Premier League":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Ligue 2":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:2])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:6])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            else:
                print("Championnat invalide")

        elif nb_joueurs == 22:
            if championnat == "Ligue 1":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue1.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue1.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue1.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue1.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Liga":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(liga.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(liga.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(liga.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(liga.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "SerieA":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(serieA.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(serieA.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(serieA.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(serieA.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Premier League":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(premier_league.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(premier_league.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(premier_league.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(premier_league.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            elif championnat == "Ligue 2":
                if nb_participants == 6:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 6" pour les joueurs sélectionnés

                    somme_Q3_a_6 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 6 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_6 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 6" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 6 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 6" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_6 = df3.loc[joueurs_a_acheter, 'Q3 à 6 joueurs'].sum(
                            )

                elif nb_participants == 8:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 8" pour les joueurs sélectionnés

                    somme_Q3_a_8 = df3.loc[joueurs_a_acheter,
                                           'Q3 à 8 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_8 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 8" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 8 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 8" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_8 = df3.loc[joueurs_a_acheter, 'Q3 à 8 joueurs'].sum(
                            )

                elif nb_participants == 10:
                    # Récupération des joueurs communs avec df_gardiens ayant une note supérieure à NoteQ80G
                    gardiens_communs = gardien_communs = df3.index.intersection(
                        df_gardiens.index).intersection(ligue2.index)
                    gardiens_qualifies = df3.loc[gardiens_communs][(
                        df3['Note'] > noteQ80G) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(gardiens_qualifies[:3])

                    # Récupération des joueurs communs avec df_défenseurs ayant une note supérieure à NoteQ80D
                    defenseurs_communs = df3.index.intersection(
                        df_défenseurs.index).intersection(ligue2.index)
                    defenseurs_qualifies = df3.loc[defenseurs_communs][(
                        df3['Note'] > noteQ80D) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(defenseurs_qualifies[:7])

                    # Récupération des joueurs communs avec df_milieux ayant une note supérieure à NoteQ80M
                    milieux_communs = df3.index.intersection(
                        df_milieux.index).intersection(ligue2.index)
                    milieux_qualifies = df3.loc[milieux_communs][(
                        df3['Note'] > noteQ80M) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(milieux_qualifies[:7])

                    # Récupération des joueurs communs avec df_attaquants ayant une note supérieure à NoteQ80A
                    attaquants_communs = df3.index.intersection(
                        df_attaquants.index).intersection(ligue2.index)
                    attaquants_qualifies = df3.loc[attaquants_communs][(
                        df3['Note'] > noteQ80A) & (df3['Nb match'] >= 9)].index.tolist()
                    joueurs_a_acheter.extend(attaquants_qualifies[:5])

                    # Calculer la somme initiale des valeurs de la colonne "Q3 a 10" pour les joueurs sélectionnés

                    somme_Q3_a_10 = df3.loc[joueurs_a_acheter,
                                            'Q3 à 10 joueurs'].sum()

                    # Tant que la somme est supérieure à 500, on remplace des joueurs
                    while somme_Q3_a_10 > 500:
                        # Trouver l'index du joueur avec la valeur de "Q3 a 10" la plus élevée parmi les joueurs sélectionnés
                        joueur_max_index = max(
                            joueurs_a_acheter, key=lambda x: df3.loc[x, 'Q3 à 10 joueurs'])

                        # Ajouter le joueur retiré à la liste des joueurs retirés
                        joueurs_retires.append(joueur_max_index)

                        joueurs_disponibles = df3.index.difference(
                            joueurs_a_acheter + joueurs_retires)

                        if joueur_max_index in gardiens_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_gardien = df3.loc[gardiens_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_gardien[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in defenseurs_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_defenseur = df3.loc[defenseurs_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_defenseur[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        elif joueur_max_index in milieux_communs:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_milieu = df3.loc[milieux_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_milieu[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                        else:
                            # Trouver les joueurs disponibles qui répondent aux critères des gardiens
                            joueur_remplacement_attaquant = df3.loc[attaquants_qualifies].index.intersection(
                                joueurs_disponibles).tolist()

                            # Retirer le joueur avec la valeur de "Q3 a 10" la plus élevée et ajouter le joueur de remplacement
                            joueurs_a_acheter.remove(joueur_max_index)
                            joueurs_a_acheter.extend(
                                joueur_remplacement_attaquant[:1])

                            # Mettre à jour la somme des valeurs
                            somme_Q3_a_10 = df3.loc[joueurs_a_acheter, 'Q3 à 10 joueurs'].sum(
                            )

                else:
                    print("Nombre de participants incorrect")

            else:
                print("Championnat invalide")

        else:
            print("Le nombre de joueurs à acheter est invalide.")

        # Définition d'une fonction pour obtenir le poste d'un joueur
        def get_poste(joueur):
            return df3.loc[df3.index == joueur, 'Poste'].values[0]

        # Définition d'une fonction pour obtenir le score d'un poste
        def get_poste_score(poste):
            poste_ordre = ["G", "DC", "DL", "MD", "MO", "A"]
            try:
                return poste_ordre.index(poste)
            except ValueError:
                # Utilisation d'une valeur infinie si le poste n'est pas dans la liste
                return float('inf')

        # Définition d'une fonction de clé pour trier les joueurs par poste
        def trier_par_poste(joueur):
            poste_joueur = get_poste(joueur)
            return get_poste_score(poste_joueur)

        # Trie des joueurs par poste
        joueurs_tries = sorted(joueurs_a_acheter, key=trier_par_poste)
        liste_mercato = []

        # Affichage des joueurs triés avec leurs postes et prix d'achat conseillé
        for joueur in joueurs_tries:
            if nb_participants == 6:
                prix_conseille = round(
                    df3.loc[df3.index == joueur, 'Q3 à 6 joueurs'].values[0])
            elif nb_participants == 8:
                prix_conseille = round(
                    df3.loc[df3.index == joueur, 'Q3 à 8 joueurs'].values[0])
            elif nb_participants == 10:
                prix_conseille = round(
                    df3.loc[df3.index == joueur, 'Q3 à 10 joueurs'].values[0])
            poste_joueur = get_poste(joueur)
            joueurs_mercato = (
                f"Joueur : {joueur}, Poste : {poste_joueur}, Prix d'achat conseillé : {prix_conseille}")
            liste_mercato.append(joueurs_mercato)
        return liste_mercato

    st.title("Application d'achat de mercato")
    st.markdown("Nous allons établir une liste des meilleurs joueurs à recruter, avec un budget de 500M€, en fonction de la ligue, du nombre de participants, ainsi que du nombre de joueurs que l'on souhaite recruter.")
    nb_joueurs = st.select_slider(
        "Nombre de joueurs à recruter :", options=[18, 20, 22], value=18)
    championnat = st.selectbox("Choisissez le championnat :", [
                               "Ligue 1", "Ligue 2", "Premier League", "SerieA", "Liga"])
    nb_participants = st.select_slider(
        "Nombre de participants dans le championnat :", options=[6, 8, 10], value=6)

    if st.button("Exécuter"):
        resultats_mercato = mercato_homogène(
            nb_joueurs, championnat, nb_participants)
        st.write("Joueurs achetés :", resultats_mercato)


elif selected_option == "Remerciements":
    st.title(" Merci pour votre attention !")
    st.write(
        "Un grand merci aux équipes de Datascientest, notamment à Roseline, Aïda et Yazid qui nous ont suivit sur ces 3 derniers moi !")
    st.balloons()
    st.image("thankyou.gif", width=700)
