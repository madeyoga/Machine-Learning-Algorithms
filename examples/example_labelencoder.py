if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from algos.Preprocessing.LabelEncoder import *
else:
    from ..algos.Preprocessing.LabelEncoder import LabelEncoder
    
le = LabelEncoder()
le.fit(['x', 'x', 'x', 'y', 'z'])
print(le.transform(['x', 'x', 'x', 'y', 'z']))
