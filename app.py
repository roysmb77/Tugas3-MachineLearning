# ======================================
# 🌐 Flask Web App untuk Prediksi Risiko Kredit
# ======================================
from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# ======================================
# 🔹 Inisialisasi Flask
# ======================================
app = Flask(__name__)

# ======================================
# 🔹 Load Model, Scaler, dan Columns
# ======================================
try:
    with open("credit_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    model_columns = bundle["columns"]

    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model, scaler, model_columns = None, None, None

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
        # Ambil semua input dari form
        input_data = {k: float(v) for k, v in request.form.items()}
        df_input = pd.DataFrame([input_data])

        # Pastikan kolom sesuai dengan urutan saat training
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0  # tambahkan kolom yang hilang

        df_input = df_input[model_columns]

        # Scaling (menggunakan scaler dari training)
        df_scaled = scaler.transform(df_input)

        # Prediksi probabilitas gagal bayar
        prob = model.predict_proba(df_scaled)[0][1] * 100

        # Threshold 40%
        pred = 1 if prob >= 40 else 0

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
