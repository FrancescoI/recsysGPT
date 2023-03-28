import pandas as pd
from torch.utils.data import Dataset

class TorchRecSysDataset(Dataset):
    def __init__(self, data_path, metadata_path):
        self.data = pd.read_csv(data_path)

        # Load metadata and preprocess
        metadata = pd.read_csv(metadata_path)
        self.user_metadata = metadata[['user_id', 'gender_id', 'age']].drop_duplicates().set_index('user_id')
        self.item_metadata = metadata[['product_id']].drop_duplicates().set_index('product_id')

        # Generate user and item mapping
        self.user2idx = {user_id: idx for idx, user_id in enumerate(self.user_metadata.index.unique())}
        self.idx2user = {idx: user_id for user_id, idx in self.user2idx.items()}

        self.item2idx = {item_id: idx for idx, item_id in enumerate(self.item_metadata.index.unique())}
        self.idx2item = {idx: item_id for item_id, idx in self.item2idx.items()}

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        timestamp = row['timestamp']
        user_id = self.user2idx[row['user_id']]
        product_id = self.item2idx[row['product_id']]
        action_id = row['action_id']

        # Get user and item metadata
        user_metadata = self.user_metadata.loc[row['user_id']].values
        item_metadata = self.item_metadata.loc[row['product_id']].values

        return {
            'timestamp': timestamp,
            'user_id': user_id,
            'product_id': product_id,
            'action_id': action_id,
            'user_metadata': user_metadata,
            'item_metadata': item_metadata
        }

    def __len__(self):
        return len(self.data)