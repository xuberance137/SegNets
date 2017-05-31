#configuration parameters
IMAGE_DATA_SHAPE = (360,480,3)
KERNEL_SIZE = (3,3)
FILTER_SIZE = 64
PAD_SIZE = (1,1)
POOL_SIZE = (2,2)
CATEGORIES = 12
CREATE_MODEL = 0
ROOT_FOLDER = '../'
DATA_PATH = ROOT_FOLDER + 'data/CamVid/'
MODEL_PATH = ROOT_FOLDER + 'model/'
DATASET_FILE = 'dataset.h5'
NUM_EPOCH = 100
BATCH_SIZE = 6
CLASS_WEIGHTING= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

# classifier label colors
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]