import numpy as np
import tensorflow as tf
from config import *
import matplotlib.pyplot as plt
import os

counter = 0

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



data,_,_ = get_data('data/cifar-10-batches-py/data_batch_1')
data = np.asarray(data.T)


def load_data(BATCH_SIZE):
	global counter
	if counter+ BATCH_SIZE > len(data):
		counter = 0

	counter = counter + BATCH_SIZE
	#print np.shape(data)

	return (data[counter-BATCH_SIZE:counter]-127.0)/128.0

def convToInt(X):
	X = X*128.0 + 127.0
	X = np.asarray(X).astype("uint8")
	return X



def visualize_image(X,fn,BATCH_SIZE,steg=False):
	global counter

	for i in range(0,BATCH_SIZE):
		rgb = X[i]
		#print(rgb.shape)
		img = rgb.reshape(3,32,32).transpose([1, 2, 0])
		#print(img.shape)
		plt.imshow(img)

		ind = counter-BATCH_SIZE+i
		plt.title(str(ind))
		#print(Y[id])
		#plt.show()

		fn_ = str(ind)
		
		if not os.path.exists(fn):
			os.makedirs(fn)
		dir = os.path.abspath(fn)

		if(steg):
			plt.savefig(dir+"/"+(fn_+'a').decode('ascii'))
		else:
			plt.savefig(dir+"/"+(fn_+'b').decode('ascii'))




# Function to generate n random messages and keys
def gen_data(n=BATCH_SIZE, msg_len=MSG_LEN, key_len=KEY_LEN, secret_len=SECRET_LEN):
    # return (np.random.randint(0, 2, size=(n, msg_len))*2-1), \
    return (load_data(n)), \
    		(np.random.randint(0, 2, size=(n, secret_len))*2-1), \
           (np.random.randint(0, 2, size=(n, key_len))*2-1)


# Xavier Glotrot initialization of weights
def init_weights(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def encrypt(input,key):
	return input + key
    #return tf.cast( tf.logical_xor(tf.cast( (input+1)/2.0,tf.bool),tf.cast(key,tf.bool)),tf.float32)

def decrypt(input,key):
    return input - key
    #return 2*(tf.cast( tf.logical_xor(tf.cast(input,tf.bool),tf.cast(key,tf.bool)),tf.float32))-1

def lossFunctionForAlice(msg,msg_gen):
	
	bit_vec = tf.add(tf.zeros([MSG_LEN]),  tf.tile(tf.constant([2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 2**0],dtype=tf.float32), [MSG_LEN/8]))
	#print bit_vec.eval()



	loss = tf.reduce_mean(tf.square(tf.multiply(msg-msg_gen, bit_vec)))
	#print loss.eval()
	

	return loss








if __name__ == '__main__':
	#print lossFunctionForAlice(tf.constant([[1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1] , [1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0]],dtype=tf.float32), tf.constant([[1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0], [1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1]],dtype=tf.float32) )
	
	for i in range(1,100):
		print load_data(10)




