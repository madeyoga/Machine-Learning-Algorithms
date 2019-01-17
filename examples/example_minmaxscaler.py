from mla.algos.preprocessing.scaler import MinMaxScaler

scaler = MinMaxScaler()
print(scaler.transform([3, 1, 5, 2, 1, 6, 7, 2, 1, 4]))
