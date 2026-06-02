import numpy as np


train_path = 'processed_data/train.npz'
test_path = 'processed_data/test.npz'


train_data = np.load(train_path, allow_pickle=True)
print(train_data)

print(train_data['feature_names'])

print(train_data['full_features'][0][0])
print(f"Lenght of first entry: {len(train_data['full_features'][0])}")