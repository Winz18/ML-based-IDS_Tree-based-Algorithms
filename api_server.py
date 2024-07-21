import numpy as np
from flask import Flask, request, jsonify, abort
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Tải mô hình và các bộ mã hóa đã lưu
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Thiết lập logging
logging.basicConfig(level=logging.INFO)


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json  # Nhận dữ liệu từ client dưới dạng JSON
        if not data:
            abort(400, description="No input data provided")

        df = pd.DataFrame(data)  # Chuyển đổi dữ liệu JSON thành DataFrame

        # Kiểm tra xem tất cả các thuộc tính cần thiết có trong dữ liệu đầu vào hay không
        required_columns = scaler.feature_names_in_
        if not all(col in df.columns for col in required_columns):
            abort(400, description="Invalid input data: missing required columns")

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_features] = scaler.transform(df[numeric_features])  # Chuẩn hóa dữ liệu

        # Dự đoán với mô hình đã huấn luyện
        predictions = model.predict(df)
        predicted_labels = label_encoder.inverse_transform(predictions)  # Chuyển đổi nhãn dự đoán thành nhãn gốc

        results = {
            'predictions': predicted_labels.tolist()
        }

        return jsonify(results)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        abort(500, description="Internal server error")


if __name__ == '__main__':
    app.run(debug=False)
