
IMAGES= '/dccstor/aurmmaret1/sajini-code/Kaggle-Data/SRGAN/Training_set/*'
OUTPUT_DIR = '/dccstor/aurmmaret1/sajini-code/Kaggle-Data/SRGAN/Results/srgan/images/'
LOGS_DIR = '/dccstor/aurmmaret1/sajini-code/Kaggle-Data/SRGAN/Results/srgan/logs/'
CHECKPOINT = '/dccstor/aurmmaret1/sajini-code/Kaggle-Data/SRGAN/Results/srgan/checkpoints/checkpoint/'
USE_CHECKPOINT = True


#NUM_TRAIN_EPOCHS = 150
NUM_TRAIN_EPOCHS = 100
TRAIN_RATIO = .9
VAL_RATIO = .1 
PRETRAIN_ONLY = False

LEARNING_RATE = 1e-4
#LEARNING_RATE = 1e-5
AD_LOSS_WEIGHT = 10.
#AD_LOSS_WEIGHT = 0.001
BN_EPSILON = 0.001
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BETA_1 = 0.9
RANDOM_SEED = 1337 


HR_HEIGHT = 96
HR_WIDTH = 96
r = 16 #2
LR_HEIGHT = HR_HEIGHT // r
LR_WIDTH = HR_WIDTH // r
NUM_CHANNELS = 3
BATCH_SIZE = 32 #4
#NUM_PRETRAIN_EPOCHS = 200
NUM_PRETRAIN_EPOCHS = 50

MAX_FILES = None
SAVE_TRUTH = False
PREDICT_ONLY = False
PREDICT_2X = False
#PREDICT_2X = True
PREDICT_4X = False
WEIGHTS = None
MEM_FRAC = 0.9
