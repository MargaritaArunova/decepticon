import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from metrics_utils import make_prediction
from model import Encoder, Decoder, TranslationModel
from train import train, set_model_weights

# fix random seed for reproducibility
RANDOM_SEED = 777

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# model and dataset config
DATASET_PREFIX = 'dataset/'
SUFFIX_FILE_EN = '.de-en.en'
SUFFIX_FILE_DE = '.de-en.de'

ENCODER_INPUT_DIM = 200  # train_loader.dataset.vocab_size_source
DECODER_INPUT_DIM = 200  # train_loader.dataset.vocab_size_target
EMBEDDING_DIM = 256
ENCODER_NUM_LAYERS = 3
DECODER_NUM_LAYERS = 3
ENCODER_NUM_HEADS = 8
DECODER_NUM_HEADS = 8
ENCODER_FEEDFORWARD_DIM = 512
DECODER_FEEDFORWARD_DIM = 512
ENCODER_DROPOUT = 0.1
DECODER_DROPOUT = 0.1
MAX_LENGTH = 200

OPTIMIZER_LEARNING_RATE = 0.0005

# get current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init dataset
train_set = TextDataset(
    data_file_en=DATASET_PREFIX + 'train' + SUFFIX_FILE_EN,
    data_file_de=DATASET_PREFIX + 'train' + SUFFIX_FILE_DE,
    sp_model_prefix=DATASET_PREFIX + 'experiment_1'
)
valid_set = TextDataset(
    data_file_en=DATASET_PREFIX + 'val' + SUFFIX_FILE_EN,
    data_file_de=DATASET_PREFIX + 'val' + SUFFIX_FILE_DE,
    sp_model_prefix=DATASET_PREFIX + 'experiment_1'
)
train_loader = DataLoader(train_set, num_workers=2, shuffle=False, batch_size=128)
valid_loader = DataLoader(valid_set, num_workers=2, shuffle=False, batch_size=128)

# init models
ENCODER_INPUT_DIM = train_loader.dataset.vocab_size_source
DECODER_INPUT_DIM = train_loader.dataset.vocab_size_target

encoder = Encoder(
    input_dim=ENCODER_INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_layers=ENCODER_NUM_LAYERS,
    num_heads=ENCODER_NUM_HEADS,
    feedforward_dim=ENCODER_FEEDFORWARD_DIM,
    dropout_prob=ENCODER_DROPOUT,
    max_length=MAX_LENGTH,
    device=device
)

decoder = Decoder(
    input_dim=DECODER_INPUT_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_layers=DECODER_NUM_LAYERS,
    num_heads=DECODER_NUM_HEADS,
    feedforward_dim=DECODER_FEEDFORWARD_DIM,
    dropout_prob=DECODER_DROPOUT,
    max_length=MAX_LENGTH,
    device=device
)

model = TranslationModel(encoder, decoder, train_set, device).to(device)
model.apply(set_model_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=train_set.pad_id)

# training process
NUM_EPOCH = 30

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=None,
    train_loader=train_loader,
    val_loader=valid_loader,
    num_epochs=NUM_EPOCH
)

# making prediction using best of trained models (according to CrossEntropy loss)
model.load_state_dict(torch.load('translation-model.pt'))
make_prediction(model)
