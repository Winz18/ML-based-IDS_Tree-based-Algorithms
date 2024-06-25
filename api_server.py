import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Tải mô hình và các bộ mã hóa đã lưu
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json  # Nhận dữ liệu từ client dưới dạng JSON
    df = pd.DataFrame(data)  # Chuyển đổi dữ liệu JSON thành DataFrame

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_features] = scaler.transform(df[numeric_features])  # Chuẩn hóa dữ liệu

    # Dự đoán với mô hình đã huấn luyện
    predictions = model.predict(df)
    predicted_labels = label_encoder.inverse_transform(predictions)  # Chuyển đổi nhãn dự đoán thành nhãn gốc

    results = {
        'predictions': predicted_labels.tolist()
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
