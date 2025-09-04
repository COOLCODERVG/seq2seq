import random
import torch

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Device (CUDA check)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_SAMPLES = 20000
MIN_LEN = 1
MAX_LEN = 8
BATCH_SIZE = 128
EMB_SIZE = 64
HIDDEN_SIZE = 128
NUM_EPOCHS = 20
LR = 0.001
TEACHER_FORCING_RATIO = 0.5
MAX_DECODING_STEPS = 30