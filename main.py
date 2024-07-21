import requests

import packet_capture as pc


def capture_data(duration):
    """
    Capture network data for a specified duration.
    """
    pc.capture_packets(duration)
    return pc.df  # Assuming packet_capture.py saves data in a DataFrame called df


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
    # Duration for which to capture network packets (in seconds)
    capture_duration = 10

    print("Capturing data...")
    data_df = capture_data(capture_duration)

    if not data_df.empty:
        print("Data captured. Sending to server for evaluation...")
        data_to_send = data_df.to_dict(orient='records')
        result = send_data_to_server(data_to_send)

        if result is not None:
            print("Evaluation Result from IDS Server:")
            print(result)
        else:
            print("Failed to get evaluation result from IDS Server.")
    else:
        print("No data captured. Please check your network and try again.")
