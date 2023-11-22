from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import TextCleaner

app = Flask(__name__)

# Load the TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('tfidf_encoder.pkl')
model = joblib.load('models/rfc.pkl')  # Adjust the path to your model file
label_encoder = joblib.load('lable_encoder.pkl')  # Adjust the path to your label encoder file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']

        # Read the content of the file in binary mode
        content = file.read()

        # Transform input text using TF-IDF vectorizer
        
        text_cleaner= TextCleaner()
        cleaned_text= text_cleaner.clean([content.decode('utf-8', 'ignore')])
        print(cleaned_text)
        
        input_text_tfidf = tfidf_vectorizer.transform(cleaned_text)

        # Make predictions
        prediction_encoded = model.predict(input_text_tfidf)

        # Decode predictions using LabelEncoder
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        # Return the prediction as JSON
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
