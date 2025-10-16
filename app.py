# ======================================
# 🌐 Flask Web App untuk Prediksi Risiko Kredit
# ======================================
from flask import Flask, render_template, request
import pickle
import pandas as pd

# Inisialisasi Flask
app = Flask(__name__)

# ======================================
# 🔹 Load Model, Scaler, dan Columns
# ======================================
with open("credit_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
model_columns = bundle["columns"]

# ======================================
# 🔹 Route Halaman Utama
# ======================================
@app.route("/")
def home():
    return render_template("index.html")

# ======================================
# 🔹 Route Prediksi
# ======================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        input_data = {k: float(v) for k, v in request.form.items()}
        df_input = pd.DataFrame([input_data])

        # One-hot encoding (harus sama seperti training)
        cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
        df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

        # Tambahkan kolom yang hilang agar sesuai dengan model
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Pastikan urutan kolom sama
        df_input = df_input[model_columns]

        # Scaling
        df_scaled = scaler.transform(df_input)

        # Prediksi probabilitas default
        prob = model.predict_proba(df_scaled)[0][1] * 100

        # Terapkan threshold 40%
        threshold = 40
        pred = 1 if prob >= threshold else 0

        hasil = "⚠️ Risiko Tinggi Gagal Bayar" if pred == 1 else "✅ Aman - Pembayaran Lancar"

        return render_template(
            "index.html",
            prediction_text=hasil,
            probability_text=f"{prob:.2f}%"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Terjadi Kesalahan: {str(e)}")

# ======================================
# 🔹 Jalankan Aplikasi
# ======================================
if __name__ == "__main__":
    app.run(debug=True)
