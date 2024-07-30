import packet_capture
import requests
import json
import time
import pandas as pd


def send_data_to_server(data):
    url = 'http://localhost:5000/api/evaluate'
    headers = {'Content-Type': 'application/json'}

    try:
        # Ghi dữ liệu ra file JSON với timestamp
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        with open(f'data_{time_stamp}.json', 'w') as f:
            json.dump(data, f)

        # Gửi request tới server
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending data to the server: {e}")
        return None


def display_results_as_table(results):
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


def main():
    print("Welcome to the Intrusion Detection System (IDS)")

    while True:
        print("\n1. Start IDS")
        print("2. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            print("Starting packet capture...")

            try:
                packet_data = packet_capture.get_packet_data()

                if packet_data.empty:
                    print("No packets captured.")
                    continue

                print("Processing captured data...")

                data = packet_data.to_dict(orient='records')
                result = send_data_to_server(data)

                if result:
                    print("IDS Detection Results:")
                    display_results_as_table(result)
                else:
                    print("Failed to get results from server.")

            except Exception as e:
                print(f"An error occurred during packet capture or processing: {e}")

        elif choice == '2':
            print("Exiting...")
            break

        else:
            print("Invalid choice, please try again.")


if __name__ == '__main__':
    main()
