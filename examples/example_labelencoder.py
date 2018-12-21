from mla.algos.Preprocessing.LabelEncoder import LabelEncoder

le = LabelEncoder()
le.fit(['x', 'x', 'x', 'y', 'z'])
print(le.transform(['x', 'x', 'x', 'y', 'z']))
