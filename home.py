import streamlit as st

def show():
    st.title("🧠 Classifieur d'Émotions - Projet IA")
    st.markdown("""
    Bienvenue dans cette application permettant de classifier des émotions ressenties par une personne à partir de phrases.
    
    ### 🧪 Modèles disponibles :
    - **CNN** : Convolutional Neural Network (classification par motifs locaux)
    - **RNN** : Recurrent Neural Network (mémoire séquentielle simple)
    - **LSTM** : Long Short-Term Memory (meilleure gestion des dépendances à long terme)

    ### 📁 Chargement personnalisé :
    Vous pouvez charger votre propre fichier CSV contenant deux colonnes :
    - `text` : le texte exprimant une émotion
    - `label` : l’étiquette correspondante
    
    > 💡 Les fichiers doivent être au format `.csv`, avec encodage UTF-8.
    
    Sélectionnez un modèle dans la barre de navigation à gauche pour commencer.
    """)

    st.info("👉 Utilisez le menu de gauche pour naviguer entre les modèles.")