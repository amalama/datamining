import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

class HousingPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Prédiction de Prix Immobiliers")
        
        self.label = tk.Label(self.master, text="Choisissez un fichier CSV pour la base de données Housing:")
        self.label.pack(pady=10)
        
        self.button = tk.Button(self.master, text="Parcourir", command=self.load_housing_data)
        self.button.pack(pady=10)
        
        self.text = scrolledtext.ScrolledText(self.master, height=10, width=50)
        self.text.pack(pady=10)
        
        self.predict_button = tk.Button(self.master, text="Prédire les prix", command=self.predict_prices)
        self.predict_button.pack(pady=10)
        
        self.result_text = scrolledtext.ScrolledText(self.master, height=5, width=50)
        self.result_text.pack(pady=10)

    def load_housing_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.text.insert(tk.END, f"Fichier chargé: {file_path}\n")
            self.display_database_info()

    def preprocess_data(self):
        # Supprimer les lignes avec des valeurs manquantes
        self.df = self.df.dropna()

    def apply_clustering(self):
        # Sélectionner les caractéristiques pour le clustering
        X_cluster = self.df[['area', 'bedrooms']]

        # Appliquer l'algorithme de K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X_cluster)

    def predict_prices(self):
        if hasattr(self, 'df'):
            # Appliquer le prétraitement des données
            self.preprocess_data()

            # Appliquer le clustering
            self.apply_clustering()

            X = self.df[['area', 'bedrooms']]
            y = self.df['price']

            # Entraîner le modèle de régression linéaire
            model = LinearRegression()
            model.fit(X, y)

            # Prédire les prix pour les données existantes
            self.df['predicted_price'] = model.predict(X)

            # Afficher les résultats
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Prédiction de Prix Immobiliers terminée.\n")
            self.result_text.insert(tk.END, f"Modèle entraîné avec succès.\n")

            # Afficher les prédictions
            self.result_text.insert(tk.END, "\nPrédictions de Prix pour les Données Existantes:\n")
            self.result_text.insert(tk.END, self.df[['area', 'bedrooms', 'predicted_price', 'cluster']])

            # Visualiser les prédictions avec un graphique
            self.plot_predictions()

            # Afficher les résultats du clustering
            self.result_text.insert(tk.END, "\nRésultats du Clustering (K-Means):\n")
            self.result_text.insert(tk.END, self.df[['area', 'bedrooms', 'cluster']])
        else:
            self.text.insert(tk.END, "Veuillez charger un fichier CSV Housing d'abord.\n")

    def display_database_info(self):
        if hasattr(self, 'df'):
            self.text.insert(tk.END, f"Colonnes de la base de données: {', '.join(self.df.columns)}\n")
            self.text.insert(tk.END, f"Nombre de lignes dans la base de données: {len(self.df)}\n")
        else:
            self.text.insert(tk.END, "Aucune base de données Housing chargée\n")

    def plot_predictions(self):
        if hasattr(self, 'df'):
            fig, ax = plt.subplots()
            scatter = ax.scatter(self.df['area'], self.df['price'], c=self.df['cluster'], label='Vraies valeurs')
            ax.scatter(self.df['area'], self.df['predicted_price'], label='Prédictions', marker='x', color='r')
            ax.set_xlabel('Surface')
            ax.set_ylabel('Prix')
            ax.set_title('Prédiction de Prix Immobiliers avec Clustering')
            ax.legend()

            # Ajouter un paragraphe explicatif
            explanation = (
                "Le graphique ci-dessus compare les vraies valeurs des prix immobiliers avec les prédictions "
                "obtenues à partir du modèle de régression linéaire. Les points rouges représentent les "
                "prédictions, et les points bleus représentent les vraies valeurs. Un alignement étroit entre "
                "les deux ensembles de points indique une bonne performance du modèle, tandis qu'un écart important "
                "peut signaler des zones où le modèle peut être amélioré. Les couleurs différentes indiquent les clusters "
                "obtenus à partir de l'algorithme de K-Means."
            )
            self.result_text.insert(tk.END, f"\n{explanation}\n")

            # Ajouter une légende pour les clusters
            legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
            ax.add_artist(legend)

            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = HousingPricePredictionApp(root)
    root.mainloop()
