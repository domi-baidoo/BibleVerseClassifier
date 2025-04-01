import streamlit as st
import joblib

# Load the saved TF-IDF vectorizer and KNN model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
knn = joblib.load('knn_model.pkl')

st.title("Bible Verse Classifier")
st.write("Enter a Bible verse below to see which category it belongs to based on text similarity.")

# Text area for user input
verse_input = st.text_area("Bible Verse:")

if st.button("Classify"):
    if verse_input:
        # Transform the input verse using the loaded vectorizer
        verse_vector = vectorizer.transform([verse_input])
        # Predict the category using the loaded KNN model
        prediction = knn.predict(verse_vector)
        st.write("The verse belongs to:", prediction[0])
    else:
        st.write("Please enter a verse to classify.")