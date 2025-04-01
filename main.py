import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib


# Create a simple dataframe of Bible verses with their respective categories
data = {
    'verse': [
        "For God so loved the world that he gave his one and only Son.",
        "I can do all things through Christ who strengthens me.",
        "The fear of the Lord is the beginning of wisdom.",
        "Love is patient, love is kind. It does not envy, it does not boast.",
        "Trust in the Lord with all your heart and lean not on your own understanding.",
        "For I know the plans I have for you, declares the Lord.",
        "Be strong and courageous. Do not be afraid; do not be discouraged.",
        "But those who hope in the Lord will renew their strength.",
        "In the beginning, God created the heavens and the earth.",
        "Come to me, all you who are weary and burdened, and I will give you rest.",
        "Blessed are the meek, for they shall inherit the earth.",
        "The Lord is my shepherd; I shall not want.",
        "Cast all your anxiety on him because he cares for you.",
        "Ask, and it will be given to you; seek, and you will find; knock, and it will be opened to you.",
        "Jesus wept.",
        "I am the way, and the truth, and the life.",
        "But seek first his kingdom and his righteousness, and all these things will be given to you as well.",
        "Your word is a lamp for my feet, a light on my path.",
        "The steadfast love of the Lord never ceases; his mercies never come to an end.",
        "For the wages of sin is death, but the gift of God is eternal life in Christ Jesus our Lord.",
        "He heals the brokenhearted and binds up their wounds.",
        "My grace is sufficient for you, for my power is made perfect in weakness.",
        "For we walk by faith, not by sight.",
        "And now these three remain: faith, hope and love. But the greatest of these is love.",
        "Let all that you do be done in love.",
        "Do not be overcome by evil, but overcome evil with good.",
        "Be kind and compassionate to one another, forgiving each other, just as in Christ God forgave you.",
        "Even though I walk through the darkest valley, I will fear no evil.",
        "For where two or three gather in my name, there am I with them.",
        "I have fought the good fight, I have finished the race, I have kept the faith.",
        "The harvest is plentiful but the workers are few.",
        "And he said to them, 'Go into all the world and preach the gospel to all creation.'",
        "In my Father's house are many rooms.",
        "Peace I leave with you; my peace I give you.",
        "Do not worry about tomorrow, for tomorrow will worry about itself.",
        "Whoever believes in me, as Scripture has said, rivers of living water will flow from within them."
    ],
    'category': [
        "Love",           # For God so loved the world...
        "Encouragement",  # I can do all things...
        "Wisdom",         # The fear of the Lord is the beginning of wisdom.
        "Love",           # Love is patient, love is kind...
        "Faith",          # Trust in the Lord with all your heart...
        "Prophecy",       # For I know the plans I have for you...
        "Encouragement",  # Be strong and courageous...
        "Faith",          # But those who hope in the Lord will renew their strength.
        "Creation",       # In the beginning, God created the heavens...
        "Comfort",        # Come to me, all you who are weary...
        "Humility",       # Blessed are the meek...
        "Comfort",        # The Lord is my shepherd...
        "Comfort",        # Cast all your anxiety on him...
        "Prayer",         # Ask, and it will be given to you...
        "Compassion",     # Jesus wept.
        "Salvation",      # I am the way, and the truth, and the life.
        "Guidance",       # But seek first his kingdom...
        "Guidance",       # Your word is a lamp for my feet...
        "Grace",          # The steadfast love of the Lord never ceases...
        "Salvation",      # For the wages of sin is death...
        "Comfort",        # He heals the brokenhearted...
        "Grace",          # My grace is sufficient for you...
        "Faith",          # For we walk by faith...
        "Love",           # And now these three remain...
        "Love",           # Let all that you do be done in love.
        "Encouragement",  # Do not be overcome by evil...
        "Forgiveness",    # Be kind and compassionate to one another...
        "Encouragement",  # Even though I walk through the darkest valley...
        "Community",      # For where two or three gather in my name...
        "Perseverance",   # I have fought the good fight...
        "Mission",        # The harvest is plentiful but the workers are few.
        "Mission",        # Go into all the world and preach the gospel...
        "Hope",           # In my Father's house are many rooms.
        "Peace",          # Peace I leave with you...
        "Encouragement",  # Do not worry about tomorrow...
        "Spiritual Renewal"  # Whoever believes in me...
    ]
}


df = pd.DataFrame(data)
print(df.head())

# Vectorize the verses
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['verse'])

# Labels
y = df['category']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using 3 neighbors to start with
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model
knn.fit(X_train, y_train)

# Save the trained vectorizer and model to disk
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(knn, 'knn_model.pkl')

print("Model training complete. Files saved as 'tfidf_vectorizer.pkl' and 'knn_model.pkl'.")