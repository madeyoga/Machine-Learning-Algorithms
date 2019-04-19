from pymla.preprocessing.scaler import MinMaxScaler, DecimalScaler, ZScoreScaler

scaler = MinMaxScaler()
print(scaler.transform([3, 1, 5, 2, 1, 6, 7, 2, 1, 4]))
scaler = DecimalScaler()
print(scaler.transform([3, 1, 5, 2, 1, 6, 7, 2, 1, 4], 5))
scaler = ZScoreScaler()
print(scaler.transform([3, 1, 5, 2, 1, 6, 7, 2, 1, 4]))
