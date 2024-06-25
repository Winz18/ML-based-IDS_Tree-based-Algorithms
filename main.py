import requests
import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')


def send_data_to_server(data):
    url = 'http://127.0.0.1:5000/api/evaluate'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


if __name__ == "__main__":
    csv_file_path = 'extracted_features.csv'
    data_to_send = load_data(csv_file_path)
    result = send_data_to_server(data_to_send)

    if result is not None:
        print("Evaluation Result from IDS Server:")
        print(result)
    else:
        print("Failed to get evaluation result from IDS Server.")
