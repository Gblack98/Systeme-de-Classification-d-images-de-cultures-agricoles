import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
import plotly.express as px

# Charger le modèle Keras
@st.cache_resource
def load_keras_model():
    model = load_model("model/Gabar_agriculture_crop_classifier.keras")
    return model

model = load_keras_model()

# Définir les classes cibles
CLASSES = [
    'Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 'Lemon', 'Olive-tree', 
    'Pearl_millet(bajra)', 'Tobacco-plant', 'almond', 'banana', 'cardamom', 'chilli',
    'clove', 'coconut', 'cotton', 'gram', 'jowar', 'jute', 'maize', 'mustard-oil',
    'papaya', 'pineapple', 'rice', 'soyabean', 'sugarcane', 'sunflower', 'tea', 
    'tomato', 'vigna-radiati(Mung)', 'wheat'
]

# Fonction de prédiction avec probabilités
def predict_with_probabilities(image, model):
    # Prétraitement de l'image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    # Prédictions
    predictions = model.predict(img_array)[0]
    probabilities = {CLASSES[i]: predictions[i] * 100 for i in range(len(CLASSES))}

    # Top 5 des prédictions
    top_5 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    predicted_class = top_5[0][0]
    confidence = top_5[0][1]

    return predicted_class, confidence, probabilities, top_5

# Interface utilisateur avec Streamlit
st.title("Classificateur de cultures agricoles 🌱")
st.write("Chargez une image pour identifier la culture agricole. 🖼️")

# Charger une image
uploaded_file = st.file_uploader("Chargez une image (formats acceptés : jpg, jpeg, png,webp)", type=["jpg", "jpeg", "png","webp"])

if uploaded_file:
    # Afficher l'image chargée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_container_width=True)

    # Prédiction
    with st.spinner("Analyse de l'image..."):
        predicted_class, confidence, probabilities, top_5 = predict_with_probabilities(image, model)

    # Résultats
    st.subheader("Résultat de la classification :")
    st.write(f"**Culture prédite** : {predicted_class}")
    st.write(f"**Confiance** : {confidence:.2f}%")

    # Graphique des probabilités (toutes les classes)
    st.subheader("Probabilités pour toutes les classes :")
    df_probabilities = pd.DataFrame(list(probabilities.items()), columns=["Classe", "Probabilité"])
    fig_all = px.bar(df_probabilities.sort_values(by="Probabilité", ascending=False),
                     x="Probabilité", y="Classe", orientation="h", 
                     title="Distribution des probabilités", height=800)
    st.plotly_chart(fig_all)

    # Graphique des Top 5 classes
    st.subheader("Top 5 des prédictions :")
    df_top5 = pd.DataFrame(top_5, columns=["Classe", "Probabilité"])
    fig_top5 = px.bar(df_top5, x="Probabilité", y="Classe", orientation="h", 
                      title="Top 5 des prédictions", color="Classe", text_auto=".2f")
    st.plotly_chart(fig_top5)

    # Explications supplémentaires
    st.subheader("Informations complémentaires 📚")
    st.write(f"Les 5 classes les plus probables avec leurs scores :")
    for rank, (cls, prob) in enumerate(top_5, start=1):
        st.write(f"{rank}. **{cls}** : {prob:.2f}%")

    # Réinitialiser en réinitialisant les variables de session
    if st.button("Réinitialiser"):
        st.session_state.clear()  # Efface toutes les variables de session
        st.query_params()  # Efface les paramètres de l'URL


else:
    st.info("Veuillez charger une image pour commencer. 📤")
