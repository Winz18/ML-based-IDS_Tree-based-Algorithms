import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os
import logging

# Cài đặt logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hằng số
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_SAMPLING_STRATEGY = {4: 1500}
N_ESTIMATORS = 500
DATA_FILE_PATH = 'MachineLearningCSV/MachineLearningCVE/data_processed.csv'


def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None, None

    df = pd.read_csv(file_path)

    # Loại bỏ các hàng có giá trị thiếu và trùng lặp
    df = df.dropna().drop_duplicates()

    # Lấy danh sách các cột số
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Loại bỏ các hàng có giá trị vô hạn và số quá lớn
    df = df[~df[numeric_features].applymap(np.isinf).any(axis=1)]
    df = df[(df[numeric_features] <= np.finfo(np.float64).max).all(axis=1)]

    return df, numeric_features


def preprocess_and_split_data(df, numeric_features):
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    le = LabelEncoder()
    df[' Label'] = le.fit_transform(df[' Label'])

    X = df.drop(columns=[' Label'], axis=1)
    y = df[' Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                                        stratify=y)
    logging.info(f"Original training set class distribution: {y_train.value_counts()}")

    smote = SMOTE(random_state=RANDOM_STATE, n_jobs=-1, sampling_strategy=SMOTE_SAMPLING_STRATEGY)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logging.info(f"Resampled training set class distribution: {y_train.value_counts()}")

    return X_train, X_test, y_train, y_test, scaler, le


def train_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=RANDOM_STATE, oob_score=True, n_jobs=-1, class_weight="balanced_subsample",
                                n_estimators=N_ESTIMATORS)
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    et = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample",
                              n_estimators=N_ESTIMATORS)

    voting_model = VotingClassifier(estimators=[('rf', rf), ('dt', dt), ('et', et)], voting='hard', n_jobs=-1,
                                    verbose=True)
    voting_model.fit(X_train, y_train)

    return voting_model


def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    logging.info("Confusion Matrix:")
    logging.info(confusion_matrix(y_test, y_pred))
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=le.classes_))


def feature_importance_optimized(model, X_train):
    feature_importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)

    features_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})
    features_df.sort_values(by='importance', ascending=False, inplace=True)

    features_df['cumulative_importance'] = features_df['importance'].cumsum()
    selected_features = features_df[features_df['cumulative_importance'] <= 0.9]['feature'].tolist()

    logging.info(features_df)
    logging.info(f"Total number of features: {len(features_df)}")
    logging.info(f"Selected features: {selected_features}")

    return selected_features


def save_models(model, scaler, le):
    joblib.dump(model, 'voting_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')


if __name__ == '__main__':
    df, numeric_features = load_and_clean_data(DATA_FILE_PATH)
    if df is not None:
        X_train, X_test, y_train, y_test, scaler, le = preprocess_and_split_data(df, numeric_features)
        voting_model = train_model(X_train, y_train)
        evaluate_model(voting_model, X_test, y_test, le)

        selected_features = feature_importance_optimized(voting_model, X_train)
        X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[' Label'], test_size=TEST_SIZE,
                                                            random_state=RANDOM_STATE, stratify=df[' Label'])
        voting_model = train_model(X_train, y_train)
        evaluate_model(voting_model, X_test, y_test, le)

        save_models(voting_model, scaler, le)
