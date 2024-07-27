from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

MODEL_PATH = 'voting_model.joblib'
SCALER_PATH = 'scaler.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

# Tải mô hình, scaler và label encoder
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    traceback.print_exc()

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        # Nhận dữ liệu JSON từ client
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(data)

        # Kiểm tra xem các thuộc tính có khớp không
        expected_columns = [
            ' Destination Port', ' Init_Win_bytes_backward', ' Average Packet Size',
            ' Bwd Packet Length Std', 'Init_Win_bytes_forward', 'Flow Bytes/s',
            ' PSH Flag Count', 'Fwd IAT Total', 'Bwd IAT Total', ' min_seg_size_forward',
            'Total Length of Fwd Packets', ' Flow Duration', ' Packet Length Mean',
            ' Avg Bwd Segment Size', ' Bwd Packet Length Mean', ' Subflow Fwd Bytes',
            ' Fwd Packet Length Max', ' Fwd IAT Max', ' Flow IAT Max', ' ACK Flag Count',
            ' Bwd Packet Length Min', ' Max Packet Length', ' Fwd Packet Length Mean',
            ' Fwd IAT Std', ' Bwd IAT Min', ' Bwd Header Length', ' Total Backward Packets',
            'Bwd Packet Length Max', ' Packet Length Std', ' Total Fwd Packets',
            ' Subflow Bwd Packets', ' Min Packet Length', ' Subflow Bwd Bytes',
            ' Packet Length Variance', ' Fwd IAT Mean', ' act_data_pkt_fwd',
            ' URG Flag Count', 'Fwd PSH Flags', ' Flow IAT Std', ' Fwd Header Length.1',
            ' SYN Flag Count', ' Bwd Packets/s'
        ]
        if not all(column in df.columns for column in expected_columns):
            return jsonify({"error": "Incorrect data format"}), 400

        # Chuẩn hóa dữ liệu
        df = df[expected_columns]
        df_scaled = scaler.transform(df)

        # Dự đoán với mô hình
        predictions = model.predict(df_scaled)
        prediction_labels = label_encoder.inverse_transform(predictions)

        # Trả về kết quả dưới dạng JSON
        return jsonify(prediction_labels.tolist()), 200

    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
