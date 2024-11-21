import joblib
import numpy as np
import pandas as pd


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

print("Before data processing:")
print(data1.shape)
print(data1[' Label'].value_counts())

# Re-sampling
data1_BENIGN = data1[(data1[' Label'] == 'BENIGN')]
data1_Botnet = data1[(data1[' Label'] == 'Bot')]

data1_DoS = data1[data1[' Label'].isin(['DoS', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris'])]
data1_DoS[' Label'] = 'DoS/DDoS'

data1_PortScan = data1[(data1[' Label'] == 'PortScan')]
data1_PortScan[' Label'] = 'PortScan'

data1_BruteForce = data1[data1[' Label'].isin(['SSH-Patator', 'FTP-Patator'])]
data1_BruteForce[' Label'] = 'BruteForce'

data1_WebAttacks = data1[data1[' Label'].isin(['Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS'])]
data1_WebAttacks[' Label'] = 'WebAttacks'

data1_minor = data1[data1[' Label'].isin(['Infiltration', 'Heartbleed'])]
data1_minor[' Label'] = 'OtherAttacks'

df = pd.concat([data1_BENIGN, data1_DoS, data1_PortScan, data1_BruteForce,
                data1_WebAttacks, data1_Botnet, data1_minor])
df = df.sort_index()

print("After data processing:")
print(df.shape)
print(df[' Label'].value_counts())

df.to_csv('/content/drive/MyDrive/MachineLearningCVE/data_processed.csv', index=False)