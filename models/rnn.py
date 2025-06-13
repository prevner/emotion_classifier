import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from models.utils import load_data, preprocess_data

def build_rnn_model(input_length, vocab_size, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=input_length))
    model.add(SimpleRNN(units=32, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def show():
    st.header("🔁 Modèle RNN")
    uploaded_file = st.file_uploader("Télécharger un fichier CSV (colonne 'text' et 'label')", type="csv")

    if uploaded_file is not None or st.checkbox("Utiliser les données par défaut", key="rnn_use_default"):
        X, y = load_data(uploaded_file)
        if X is not None:
            data, classes, tokenizer = preprocess_data(X, y)
            X_train, X_test, y_train, y_test = data

            model = build_rnn_model(X_train.shape[1], vocab_size=10000, num_classes=len(classes))

            if st.button("Entraîner le modèle RNN"):
                with st.spinner("Entraînement du RNN en cours..."):
                    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
                    loss, acc = model.evaluate(X_test, y_test)
                    st.success(f"Modèle RNN entraîné ! Précision sur les tests : {acc:.2f}")
        else:
            st.warning("Erreur lors du chargement des données.")
    else:
        st.info("Veuillez télécharger un fichier ou utiliser les données par défaut.")