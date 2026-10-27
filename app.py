import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

MODEL_DIR = "model/cnn_lstm_model"

print("Loading model and scalers...")
model = load_model(f"{MODEL_DIR}/final_cnn_lstm_model.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])

x_scaler = joblib.load(f"{MODEL_DIR}/x_scaler.pkl")
y_scaler = joblib.load(f"{MODEL_DIR}/y_scaler.pkl")

with open(f"{MODEL_DIR}/meta.json", "r") as f:
    meta = json.load(f)

FEATURES = meta["features"]
WINDOW_SIZE = meta["window_size"]



app = Flask(__name__)
app.secret_key = 'unmyeong'

DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_data TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        user_check = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if user_check:
            flash('Username or email already exists. Please choose a different one.', 'error')
            conn.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)

        try:
            conn.execute(
                'INSERT INTO users (username, email, phone_number, password) VALUES (?, ?, ?, ?)',
                (username, email, phone_number, hashed_password)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT username, email, phone_number FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()
    conn.close()

    return render_template('home.html', user=user)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Require login
    if 'username' not in session:
        flash("Please log in to make a prediction.", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # 1️⃣ Gather input values from the form
            user_input = {f: float(request.form.get(f, 0)) for f in FEATURES}

            # 2️⃣ Prepare model input
            input_vector = np.array([[user_input[f] for f in FEATURES]], dtype=float)
            input_scaled = x_scaler.transform(input_vector)
            input_window = np.repeat(input_scaled, WINDOW_SIZE, axis=0).reshape(1, WINDOW_SIZE, len(FEATURES))

            # 3️⃣ Run prediction
            pred_scaled = model.predict(input_window)
            predicted_value = y_scaler.inverse_transform(pred_scaled)[0, 0]

            # 4️⃣ Save to DB (automatically linked to the session user)
            conn = get_db_connection()
            user = conn.execute(
                'SELECT id FROM users WHERE username = ?',
                (session['username'],)
            ).fetchone()

            if user:
                conn.execute(
                    'INSERT INTO predictions (user_id, input_data, predicted_value) VALUES (?, ?, ?)',
                    (user['id'], json.dumps(user_input), float(predicted_value))
                )
                conn.commit()

            conn.close()

            # 5️⃣ Show success + result
            flash(f"Prediction successful! Predicted close price: {predicted_value:.4f}", "success")
            return render_template('predict.html', features=FEATURES, prediction=predicted_value, values=user_input)

        except Exception as e:
            print("Prediction error:", e)
            flash(f"Error during prediction: {e}", "error")

    # For GET requests — show empty form
    return render_template('predict.html', features=FEATURES)

@app.route('/history')
def history():
    # Require login
    if 'username' not in session:
        flash("Please log in to view your history.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT id FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()

    predictions = []
    if user:
        predictions = conn.execute(
            'SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC',
            (user['id'],)
        ).fetchall()

    conn.close()

    return render_template('history.html', username=session['username'], predictions=predictions)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/datascience')
def datascience():
    return render_template('datascience.html')

@app.route('/exsisting')
def exsisting():
    return render_template('exsisting.html')

@app.route('/proposed')
def proposed():
    return render_template('proposed.html')

if __name__ == '__main__':
    app.run(debug=True)
