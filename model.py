import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print(df[' Label'].value_counts())

    # Kiểm tra giá trị thiếu
    print("Missing values per column:")
    print(df.isnull().sum())

    # Loại bỏ các hàng có giá trị thiếu lần 1
    df = df.dropna()

    # Lấy danh sách các cột số (78 cột trừ Label)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Loại bỏ các hàng có giá trị vô hạn và số quá lớn
    df = df[~df[numeric_features].applymap(np.isinf).any(axis=1)]
    df = df[(df[numeric_features] <= np.finfo(np.float64).max).all(axis=1)]

    # Loại bỏ các hàng có giá trị thiếu lần 2
    df = df.dropna()

    # Loại bỏ các hàng trùng lặp
    df = df.drop_duplicates()

    return df, numeric_features


def preprocess_and_split_data(df, numeric_features):
    # Chuẩn hóa các thuộc tính số
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Mã hóa nhãn
    le = LabelEncoder()
    df[' Label'] = le.fit_transform(df[' Label'])

    # Tạo X (thuộc tính) và y (nhãn)
    X = df.drop(columns=[' Label'], axis=1)
    y = df[' Label']

    # Chia data theo tỷ lệ 8:2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, scaler, le


def train_and_evaluate_models(X_train, y_train, X_test, y_test, le):
    # Tạo các mô hình riêng lẻ
    rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1, class_weight="balanced_subsample")
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    et = ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample")

    # Tạo voting ensemble
    voting_model = VotingClassifier(estimators=[
        ('rf', rf),
        ('dt', dt),
        ('et', et)
    ], voting='hard')

    # Huấn luyện voting ensemble
    voting_model.fit(X_train, y_train)
    y_pred = voting_model.predict(X_test)

    # Hiển thị kết quả về độ chính xác
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return voting_model


def save_models(voting_model, scaler, le):
    # Lưu mô hình và các bộ mã hóa
    joblib.dump(voting_model, 'voting_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')


# Đường dẫn tới file dữ liệu
file_path = '/content/drive/MyDrive/MachineLearningCVE/data_processed.csv'

# Load và làm sạch dữ liệu
df, numeric_features = load_and_clean_data(file_path)

# Tiền xử lý và chia dữ liệu
X_train, X_test, y_train, y_test, scaler, le = preprocess_and_split_data(df, numeric_features)

# Huấn luyện và đánh giá các mô hình
voting_model = train_and_evaluate_models(X_train, y_train, X_test, y_test, le)

# Lưu mô hình
save_models(voting_model, scaler, le)
