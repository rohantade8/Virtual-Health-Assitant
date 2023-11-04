from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # Import joblib

app = Flask(__name__)

# Load the CSV data and perform preprocessing
df = pd.read_csv("C:\\Users\\rohan\\OneDrive\\Desktop\\Health assitant\\Virtual Health Assistant\\Virtual-Health-Assitant\\Dataset\\medical data.csv")
df.dropna(subset=['Symptoms', 'Medicine'], inplace=True)
X = df['Symptoms']
y = df['Medicine']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Load the pre-trained classifier model from the pickle file
model_path = r"C:\Users\rohan\OneDrive\Desktop\Health assitant\Virtual Health Assistant\Virtual-Health-Assitant\Training\medical_classifier.pkl"
clf = joblib.load(model_path)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('symptoms')
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = clf.predict(user_input_vectorized)
    return render_template('result.html', condition=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
