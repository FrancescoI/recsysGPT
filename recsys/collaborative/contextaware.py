import torch
import torch.nn as nn

class ContextAwareCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, n_actions, n_factors, user_metadata_dim, item_metadata_dim, hidden_dim=32):
        super().__init__()

        # User, item, and action embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.action_factors = nn.Embedding(n_actions, n_factors)

        # User and item metadata embeddings
        self.user_metadata_layer = nn.Linear(user_metadata_dim, n_factors)
        self.item_metadata_layer = nn.Linear(item_metadata_dim, n_factors)

        # Hidden layers
        self.fc1 = nn.Linear(n_factors * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, action, user_metadata, item_metadata):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        action_embedding = self.action_factors(action)

        # Incorporate metadata
        user_metadata_embedding = self.user_metadata_layer(user_metadata)
        item_metadata_embedding = self.item_metadata_layer(item_metadata)

        # Combine embeddings
        user_embedding = user_embedding + user_metadata_embedding
        item_embedding = item_embedding + item_metadata_embedding

        # Concatenate user, item, and action embeddings
        x = torch.cat((user_embedding, item_embedding, action_embedding), dim=1)

        # Pass through hidden layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Output
        out = self.sigmoid(x)
        return out.squeeze()