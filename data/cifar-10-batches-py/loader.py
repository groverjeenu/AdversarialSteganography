import numpy as np
import matplotlib.pyplot as plt
import os
 
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
def get_data(file):
	absFile = os.path.abspath(file)
	dict = unpickle(absFile)
	#for key in dict.keys():
	#	print(key)
	#print("Unpacking {}".format(dict[b'batch_label']))
	X = np.asarray(dict[b'data'].T).astype("uint8")
	Yraw = np.asarray(dict[b'labels'])
	Y = np.zeros((10,10000))
	for i in range(10000):
		Y[Yraw[i],i] = 1
	names = np.asarray(dict[b'filenames'])
	return X,Y,names
def visualize_image(X,Y,names,id):
	rgb = X[:,id]
	#print(rgb.shape)
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	#print(img.shape)
	plt.imshow(img)
	plt.title(names[id])
	#print(Y[id])
	#plt.show()
	dir = os.path.abspath("samples")
	plt.savefig(dir+"/"+names[id].decode('ascii'))



if __name__ == '__main__':
	X,Y,names = get_data('data_batch_1')
	visualize_image(X,Y,names,100)