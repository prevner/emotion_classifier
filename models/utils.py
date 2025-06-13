import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

@st.cache_data
def load_data(file_path=None):
    """
    Charge les données depuis un fichier CSV ou Kaggle.
    Retourne deux listes : textes (X) et labels (y)
    """
    if file_path is not None:
        # Charger depuis le fichier téléchargé
        df = pd.read_csv(file_path)
    else:
        try:
            # Charger depuis Kaggle
            from kagglehub import dataset_download
            path = dataset_download("nelgiriyewithana/emotions")
            df = pd.read_csv(f"{path}/text.csv")
        except Exception as e:
            st.error("Impossible de charger les données depuis Kaggle. Téléchargez un fichier CSV valide.")
            return None, None

    # Supprimer la colonne 'Unnamed: 0' si elle existe
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # Vérifier que les colonnes nécessaires existent
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Le fichier manque les colonnes suivantes : {missing_columns}")
        return None, None

    X = df['text'].astype(str)
    y = df['label']

    return X, y


def preprocess_data(X, y, vocab_size=10000, max_len=200):
    """
    Prépare les données : tokenisation, padding, encodage des labels,
    et division en ensembles d'entraînement/test.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, maxlen=max_len)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return (X_train, X_test, y_train, y_test), le.classes_, tokenizer