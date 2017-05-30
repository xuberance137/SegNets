import os
import h5py
from keras.models import model_from_json
from config import *

label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

def visualize_prediction(model):
	with h5py.File(DATA_PATH+DATASET_FILE, 'r') as hf:
		train_data = hf['train_data'][:]

	gt = []
	with open(DataPath+'train.txt') as f:
	    txt = f.readlines()
	    txt = [line.split(' ') for line in txt]
	for i in range(len(txt)):
	    gt.append(cv2.imread(os.getcwd() + txt[i][0][7:]))


	output = model.predict_proba(train_data[23:24])
	pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)), False)
	plt.imshow(pred)
	plt.figure(2)
	plt.imshow(gt[23])

if __name__ == '__main__':
	segnet_model = model_from_json(MODEL_PATH+'basic_model.json')
	segnet_model.load_weights(MODEL_PATH+'basic_model.h5')
	visualize_prediction(segnet_model)