from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Backend API URL
BACKEND_API = "http://localhost:8000/api"

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['DEBUG'] = True

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/predict')
def predict():
    """Eligibility prediction page"""
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
