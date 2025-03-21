import pandas as pd
from tqdm import tqdm
from os.path import join

import torch.nn as nn
from torch import long
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from gensim.models import Word2Vec
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define constants and hyperparameters
MODEL_NAME = 'myw2v.model'
DATASET_LOC = './datasets/cora'
NUM_CLASSES = 7
LEARN_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 10

# Load the Word2Vec model and read the dataset
model = Word2Vec.load(MODEL_NAME)
content_df = pd.read_csv(join(DATASET_LOC, "cora.content"), sep = '\t', header = None)[[0, 1434]]
content_df.rename({0: 'node', 1434: 'type'}, axis = 1, inplace = True)

# Create a DataFrame to map node IDs to vector indices
k2i = pd.DataFrame({'node': model.wv.key_to_index.keys(), 'vindex': model.wv.key_to_index.values()})

# Merge the content DataFrame with the vector indices
meta_df = pd.merge(content_df, k2i, on='node').sort_values('vindex')

# Encode the 'type' column to numerical labels
meta_df['y'] = LabelEncoder().fit_transform(meta_df['type'])

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(model.wv.vectors, 
                                                meta_df['y'], 
                                                train_size = 0.9, 
                                                stratify = meta_df['y'])

# Define a neural network classifier
class NNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(model.wv.vector_size, model.wv.vector_size // 2)
        self.dropout1 = nn.Dropout(p = 0.2)
        self.linear2 = nn.Linear(model.wv.vector_size // 2, NUM_CLASSES)
        self.dropout2 = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        probs = nn.functional.softmax(x, dim = -1)
        return probs

# Initialize the neural network classifier and optimizer
nnmodel = NNClassifier()
optim = AdamW(nnmodel.parameters(), lr = LEARN_RATE)

# Convert data to PyTorch tensors and create datasets and dataloaders
xtrain = Tensor(xtrain)
ytrain = Tensor(ytrain.values).type(long)
xtest = Tensor(xtest)
ytest = Tensor(ytest.values).type(long)

train_dataset = TensorDataset(xtrain, ytrain)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
test_dataset = TensorDataset(xtest, ytest)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)

# Lists to store training and testing loss
train_loss = []
test_loss = []

# Training loop
for epoch in tqdm(range(EPOCHS)):
    etrain_loss = []
    etest_loss = []

    # Set the model to training mode
    nnmodel.train()
    # Iterate through the training data
    for batch_x, batch_y in train_dataloader:
        probs = nnmodel(batch_x)
        loss = nn.functional.cross_entropy(probs, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        etrain_loss.append(loss.detach().numpy())

    # Set the model to evaluation mode
    nnmodel.eval()
    # Iterate through the testing data
    for batch_x, batch_y in test_dataloader:
        probs = nnmodel(batch_x)
        loss = nn.functional.cross_entropy(probs, batch_y)
        etest_loss.append(loss.detach().numpy())

    # Calculate and store average training and testing loss for this epoch
    train_loss.append(sum(etrain_loss) / len(etrain_loss))
    test_loss.append(sum(etest_loss) / len(etest_loss))

# Plot the training and testing loss
fig, ax = plt.subplots()
ax.plot(train_loss, label = 'train')
ax.plot(test_loss, label = 'test')
ax.legend(frameon = False, loc = 'lower center', ncol = 2)
ax.set_title('train and test losses')
plt.show()