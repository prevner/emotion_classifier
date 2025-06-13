import streamlit as st
import home
import models.rnn as rnn
import models.lstm as lstm
import models.cnn as cnn

# Configuration de la page
st.set_page_config(
    page_title="Classifieur d'Émotions",
    page_icon="🧠",
    layout="wide"
)

# Barre latérale pour la navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/05/Factorio_neural_network_icon.png ", use_column_width=True)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "CNN", "RNN", "LSTM"])

# Affichage conditionnel des pages
if page == "Accueil":
    home.show()
elif page == "CNN":
    cnn.show()
elif page == "RNN":
    rnn.show()
elif page == "LSTM":
    lstm.show()