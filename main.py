import packet_capture
import requests
import json
import time
import pandas as pd


def send_data_to_server(data):
    url = 'http://localhost:5000/api/evaluate'
    headers = {'Content-Type': 'application/json'}

    try:
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        with open(f'data_{time_stamp}.json', 'w') as f:
            json.dump(data, f)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def main():
    print("Welcome to the Intrusion Detection System (IDS)")
    while True:
        print("1. Start IDS")
        print("2. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            print("Starting packet capture...")
            try:
                packet_data = packet_capture.get_packet_data()
                print("Processing captured data...")

                if packet_data.empty:
                    print("No packets captured.")
                    continue

                try:
                    data = packet_data.to_dict(orient='records')
                    result = send_data_to_server(data)

                    if result:
                        print("IDS Detection Results:")
                        print(result)
                    else:
                        print("Failed to get results from server.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            except Exception as e:
                print(f"An error occurred during packet capture: {e}")

        elif choice == '2':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == '__main__':
    main()
