from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Define a dummy route for the root URL to avoid 404 error
@app.route('/')
def index():
    return 'Welcome to the API!'

@app.route('/predict', methods=['POST','GET'])
def predict():
    # Get data from the request
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid JSON format or missing "text" key'}), 400

    # Load LabelEncoder, TF-IDF vectorizer, and the model
    label_encoder = joblib.load('label_encoder.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('your_model.pkl')

    # Transform input text using TF-IDF vectorizer
    input_text_tfidf = tfidf_vectorizer.transform([data['text']])

    # Make predictions
    predictions_encoded = model.predict(input_text_tfidf)

    # Decode predictions using LabelEncoder
    predictions = label_encoder.inverse_transform(predictions_encoded)

    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port=5000)
