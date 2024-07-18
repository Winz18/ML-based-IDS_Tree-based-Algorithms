import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

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
    print(y_train.value_counts())

    # Sử dụng Smote để cân bằng dữ liệu
    smote = SMOTE(random_state=42, n_jobs=-1, sampling_strategy={4: 1500})
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test, scaler, le


def train_and_evaluate_models(X_train, y_train, X_test, y_test, le):
    # Tạo các mô hình riêng lẻ
    rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1, class_weight="balanced_subsample",
                                n_estimators=500)
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    et = ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample", n_estimators=500)

    # Tạo voting ensemble
    voting_model = VotingClassifier(estimators=[
        ('rf', rf),
        ('dt', dt),
        ('et', et)
    ], voting='hard', n_jobs=-1, verbose=True)

    # Huấn luyện voting ensemble
    voting_model.fit(X_train, y_train)
    y_pred = voting_model.predict(X_test)

    # Hiển thị kết quả về độ chính xác
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return voting_model


def feature_importance_optimized(voting_model, X_train):
    # Calculate mean feature importances using numpy for efficiency
    feature_importances = np.mean([est.feature_importances_ for est in voting_model.estimators_], axis=0)

    # Create a DataFrame for easier sorting and manipulation
    features_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})
    features_df.sort_values(by='importance', ascending=False, inplace=True)

    # Calculate cumulative importance and select features until 90% importance is reached
    features_df['cumulative_importance'] = features_df['importance'].cumsum()
    selected_features = features_df[features_df['cumulative_importance'] <= 0.9]['feature'].tolist()

    # Optionally, print the DataFrame for review
    print(features_df)
    print("Total number of features:", len(features_df))
    print("Selected features:", selected_features)

    return selected_features


def save_models(voting_model, scaler, le):
    # Lưu mô hình và các bộ mã hóa
    joblib.dump(voting_model, 'voting_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')


# Đường dẫn tới file dữ liệu
file_path = 'MachineLearningCSV/MachineLearningCVE/data_processed.csv'

# Load và làm sạch dữ liệu
df, numeric_features = load_and_clean_data(file_path)

# Tiền xử lý và chia dữ liệu
X_train, X_test, y_train, y_test, scaler, le = preprocess_and_split_data(df, numeric_features)

# Huấn luyện và đánh giá các mô hình
voting_model = train_and_evaluate_models(X_train, y_train, X_test, y_test, le)

# Tính toán độ quan trọng của các thuộc tính
selected_features = feature_importance_optimized(voting_model, X_train)

# Huấn luyện lại mô hình với các thuộc tính được chọn
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[' Label'], test_size=0.2, random_state=42,
                                                    stratify=df[' Label'])
voting_model = train_and_evaluate_models(X_train, y_train, X_test, y_test, le)

# Lưu mô hình
save_models(voting_model, scaler, le)
