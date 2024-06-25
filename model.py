import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib


''' Đọc dữ liệu từ tệp CSV và kết nối các tệp CSV '''

file_path1 = "/content/drive/MyDrive/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
file_path2 = "/content/drive/MyDrive/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
file_path5 = "/content/drive/MyDrive/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
file_path6 = "/content/drive/MyDrive/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
file_path3 = "/content/drive/MyDrive/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
file_path4 = "/content/drive/MyDrive/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv"
file_path7 = "/content/drive/MyDrive/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"
file_path8 = "/content/drive/MyDrive/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
data5 = pd.read_csv(file_path5)
data6 = pd.read_csv(file_path6)
data3 = pd.read_csv(file_path3)
data4 = pd.read_csv(file_path4)
data7 = pd.read_csv(file_path7)
data8 = pd.read_csv(file_path8)

data1 = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8])

important_columns = [
    " Subflow Fwd Bytes",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Fwd Packet Length Mean",
    " Avg Bwd Segment Size",
    "Total Length of Fwd Packets",
    " Fwd Packet Length Max",
    "Bwd Packet Length Max",
    " Max Packet Length",
    "Init_Win_bytes_forward",
    " Total Length of Bwd Packets",
    " Bwd Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    " ACK Flag Count",
    " Packet Length Mean",
    " Bwd Packet Length Std",
    " act_data_pkt_fwd",
    " Bwd Header Length",
    " Bwd Packets/s",
    "Subflow Fwd Packets",
    " Bwd Packet Length Min",
    " PSH Flag Count",
    " Total Fwd Packets",
    " Subflow Bwd Bytes",
    " Fwd Header Length.1",
    " Fwd Header Length",
    " Fwd IAT Std",
    " Subflow Bwd Packets",
    " Fwd Packet Length Min",
    " Flow IAT Mean",
    " Label"
]

data1 = data1[important_columns]


''' Xử lý dữ liệu '''
print("Before data processing:")
print(data1.shape)
print(data1.info())
print(data1.head())

# Loại bỏ các hàng có giá trị thiếu lần 1
data1 = data1.dropna()

# Lấy danh sách các cột số (78 cột trừ Label)
numeric_features = data1.select_dtypes(include=[np.number]).columns.tolist()

# Loại bỏ các hàng có giá trị vô hạn và số quá lớn
data1 = data1[~data1[numeric_features].applymap(np.isinf).any(axis=1)]
data1 = data1[(data1[numeric_features] <= np.finfo(np.float64).max).all(axis=1)]

# Loại bỏ các hàng có giá trị thiếu lần 2
data1 = data1.dropna()

# Chuẩn hóa các thuộc tính số
scaler = StandardScaler()
data1[numeric_features] = scaler.fit_transform(data1[numeric_features])

# Loại bỏ các hàng trùng lặp
data1 = data1.drop_duplicates()

# Mã hóa nhãn
le = LabelEncoder()
data1[' Label'] = le.fit_transform(data1[' Label'])


''' Huấn luyện mô hình Random Forest '''

# Tạo X (thuộc tính) và y (nhãn)
X = data1.drop(columns=[' Label'], axis=1)
y = data1[' Label']

# Chia data theo tỷ lệ 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng RandomForestClassifier để phân loại
rf = RandomForestClassifier(random_state=42, oob_score=True, n_estimators=300,
                            verbose=1, n_jobs=-1, class_weight="balanced_subsample",
                            max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


''' Đánh giá mô hình '''

# Hiển thị kết quả về độ chính xác
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("OOB Score:")
print(rf.oob_score_)

# Lưu mô hình và các bộ mã hóa
joblib.dump(rf, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')
