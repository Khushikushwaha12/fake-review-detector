from flask import Flask, render_template, request
import joblib

# 1. Sabse pehle app define karna zaroori hai
app = Flask(__name__)

# 2. Model aur Vectorizer load karein
model = joblib.load('review_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        
        # Text ko transform aur predict karna
        data = vectorizer.transform([review_text])
        prediction = model.predict(data)[0]
        
        # Confidence nikalne ke liye
        probs = model.predict_proba(data)[0]
        confidence = round(max(probs) * 100, 2)
        
        result = "FAKE" if prediction == 1 else "REAL"
        
        return render_template('index.html', prediction=result, confidence=confidence, text=review_text)

if __name__ == '__main__':
    # Isse aap network par bhi chala payenge
    app.run(host='0.0.0.0', port=5000, debug=True)