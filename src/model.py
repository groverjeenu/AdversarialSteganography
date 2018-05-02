import tensorflow as tf
import numpy as np

import matplotlib
# OSX fix
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from layers import conv_layer
from config import *
from utils import init_weights, gen_data, encrypt, decrypt, lossFunctionForAlice, convToInt, visualize_image
from datetime import datetime

class CryptoNet(object):
    def __init__(self, sess, msg_len=MSG_LEN, secret_len=SECRET_LEN, key_len=KEY_LEN, random_seed_len=RANDOM_SEED_LEN, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, eve_learning_rate=EVE_LEARNING_RATE):
        """
        Args:
            sess: TensorFlow session
            msg_len: The length of the input message to encrypt.
            key_len: Length of Alice and Bob's private key.
            batch_size: Minibatch size for each adversarial training
            epochs: Number of epochs in the adversarial training
            learning_rate: Learning Rate for Adam Optimizer
        """

        self.sess = sess
        self.msg_len = msg_len
        self.key_len = key_len
        self.random_seed_len = random_seed_len
        self.secret_len = secret_len

        self.N = self.msg_len


        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.eve_learning_rate = eve_learning_rate


        # Hyper-parameters
        self.alice_factor = 50
        self.bob_factor = 100
        self.eve_factor = 1
        self.eve_opt_factor = 100
        self.iterations = iterations = 128
        self.file_name_without_time = str(self.msg_len)+"msg"+str(self.secret_len)+"secret"+str(self.key_len)+"key"+str(self.random_seed_len)+"seed"+str(self.iterations)+"iters"+str(self.epochs)+"epochs"+str(self.learning_rate)+"learnrate"+str(self.eve_learning_rate)+"evelearnrate" +str(self.batch_size)+"batchsize"+str(self.alice_factor)+"timesAliceError" + str(self.bob_factor)+"timesBobError"+str(self.eve_factor)+"timesEveError"+ str(self.eve_opt_factor)+"timesEveOptError"
        self.file_name = str(datetime.now())+"time"+self.file_name_without_time
        

        self.build_model()

    def build_model(self):
        # Weights for fully connected layers
        self.w_alice = init_weights("alice_w", [self.msg_len + self.secret_len + self.key_len, 2*self.msg_len])
        self.w_bob = init_weights("bob_w", [self.msg_len + self.key_len, 2 * self.secret_len])
        self.w_keygen = init_weights("keygen_w",[self.random_seed_len, 2*self.key_len])
        self.w_eve1 = init_weights("eve_w1", [self.msg_len, 2 * self.msg_len])
        self.w_eve2 = init_weights("eve_w2", [2 * self.msg_len, 2 * self.secret_len])

        # Placeholder variables for Message and Key
        self.msg = tf.placeholder("float", [None, self.msg_len])
        self.secret = tf.placeholder("float", [None, self.secret_len])
        self.seed = tf.placeholder("float", [None, self.random_seed_len])


        # KeyGen's network
        # self.keygen_input = self.seed
        # self.keygen_hidden = tf.nn.tanh(tf.matmul(self.keygen_input,self.w_keygen))
        # self.keygen_hidden = tf.expand_dims(self.keygen_hidden, 2)
        # self.key = tf.sigmoid(tf.squeeze(conv_layer(self.keygen_hidden,"keygen")));
        self.key = self.seed


        # Alice's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.alice_input = tf.concat(axis=1, values=[self.msg, self.secret, self.key])
        self.alice_hidden = tf.nn.tanh(tf.matmul(self.alice_input, self.w_alice))
        self.alice_hidden = tf.expand_dims(self.alice_hidden, 2)
        self.alice_output = tf.squeeze(conv_layer(self.alice_hidden, "alice"))
        #self.alice_output = encrypt(self.msg,self.key)

        # Bob's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.bob_input = tf.concat(axis=1, values=[self.alice_output, self.key])
        self.bob_hidden = tf.nn.tanh(tf.matmul(self.bob_input, self.w_bob))
        self.bob_hidden = tf.expand_dims(self.bob_hidden, 2)
        self.bob_output = tf.squeeze(conv_layer(self.bob_hidden, "bob"))
        #self.bob_output = decrypt(self.alice_output,self.key)

        # Eve's network
        # FC layer -> FC layer -> Conv Layer (4 1-D convolutions)
        self.eve_input = self.alice_output
        self.eve_hidden1 = tf.nn.tanh(tf.matmul(self.eve_input, self.w_eve1))   #Sigmoid Earlier
        self.eve_hidden2 = tf.nn.tanh(tf.matmul(self.eve_hidden1, self.w_eve2)) #Sigmoid Earlier
        self.eve_hidden2 = tf.expand_dims(self.eve_hidden2, 2)
        self.eve_output = tf.squeeze(conv_layer(self.eve_hidden2, "eve"))


    def train(self):
        # Loss Functions
        self.decrypt_err_eve = tf.reduce_mean(tf.abs(self.secret - self.eve_output))
        self.decrypt_err_bob = tf.reduce_mean(tf.abs(self.secret - self.bob_output))
        self.decrypt_err_alice = tf.reduce_mean(tf.abs(self.msg - self.alice_output)) #lossFunctionForAlice(self.msg, self.alice_output)


        self.loss_bob = self.bob_factor*self.decrypt_err_bob + self.eve_factor*(1 - self.decrypt_err_eve) ** 2. +  self.alice_factor*self.decrypt_err_alice

        # Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice_' in var.name or 'bob_' in var.name or 'keygen_' in var.name]
        self.eve_vars = [var for var in self.t_vars if 'eve_' in var.name]

        # Build the optimizers
        self.bob_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_bob, var_list=self.alice_or_bob_vars)
        self.eve_optimizer = tf.train.AdamOptimizer(self.eve_learning_rate).minimize(
            self.eve_opt_factor*self.decrypt_err_eve, var_list=self.eve_vars)

        self.bob_errors, self.eve_errors, self.alice_errors = [], [], []

        # Begin Training
        tf.initialize_all_variables().run()
        for i in range(self.epochs):
            

            print 'Training Alice and Bob, Epoch:', i + 1
            bob_loss, _, alice_loss = self._train('bob', self.iterations)
            self.bob_errors.append(bob_loss)
            self.alice_errors.append(alice_loss)

            print 'Training Eve, Epoch:', i + 1
            _, eve_loss, _ = self._train('eve', self.iterations)
            self.eve_errors.append(eve_loss)

        self.plot_errors()

    def test(self):
        # Loss Functions
        self.decrypt_err_eve = tf.reduce_mean(tf.square(self.secret - self.eve_output))
        self.decrypt_err_bob = tf.reduce_mean(tf.square(self.secret - self.bob_output))
        self.decrypt_err_alice = tf.reduce_mean(tf.square(self.msg - self.alice_output))


        
        # Train Eve for two minibatches to give it a slight computational edge
        no_of_examples = 5
        alice_test_error = []
        bob_test_error = []
        eve_test_error = []
        for i in range(no_of_examples):
            alice_decrypt_error, bob_decrypt_error, eve_decrypt_error = 0.0, 0.0, 0.0
            bs = self.batch_size
            msg_in_val, secret_val, key_val = gen_data(n=bs, msg_len=self.msg_len, secret_len=self.secret_len, key_len=self.random_seed_len)
            
            decrypt_err_alice, decrypt_err_bob, decrypt_err_eve,alice,bob,eve = self.sess.run([self.decrypt_err_alice, self.decrypt_err_bob, self.decrypt_err_eve,self.alice_output,self.bob_output,self.eve_output],
                                           feed_dict={self.msg: msg_in_val, self.secret: secret_val, self.seed: key_val})
            #eve_decrypt_error = min(eve_decrypt_error, decrypt_err)
            eve_decrypt_error = eve_decrypt_error + decrypt_err_eve
            bob_decrypt_error = bob_decrypt_error + decrypt_err_bob
            alice_decrypt_error = alice_decrypt_error + decrypt_err_alice

            print msg_in_val[0], key_val[0], alice[0],bob[0],eve[0]
            bob_test_error.append(bob_decrypt_error)
            eve_test_error.append(eve_decrypt_error)
            alice_test_error.append(alice_decrypt_error)

            visualize_image(convToInt(msg_in_val),self.file_name,bs,steg=False)
            visualize_image(convToInt(alice),self.file_name,bs,steg=True)





        print "here"
    
        f = open(self.file_name+".out.txt" ,'w')
        print "here2"
        for i in bob_test_error:
            f.write(str(i)+",")
        f.write("\n")
        for i in eve_test_error:
            f.write(str(i)+",")
        f.write("\n")
        for i in alice_test_error:
            f.write(str(i)+",")
        f.write("\n")
        f.write(str(np.mean(bob_test_error)) +", "+str(np.std(bob_test_error)))
        f.write("\n")
        f.write(str(np.mean(eve_test_error)) + ", " + str(np.std(eve_test_error)))
        f.write("\n")
        f.write(str(np.mean(alice_test_error)) + ", " + str(np.std(alice_test_error)))
        f.write("\n")
        f.close()
        print np.mean(bob_test_error), np.std(bob_test_error)
        print np.mean(eve_test_error), np.std(eve_test_error)
        print np.mean(alice_test_error), np.std(alice_test_error)

        return bob_decrypt_error, eve_decrypt_error, alice_decrypt_error



    def test_(self):
        # Loss Functions
        self.decrypt_err_eve = tf.reduce_mean(tf.square(self.secret - self.eve_output))
        self.decrypt_err_bob = tf.reduce_mean(tf.square(self.secret - self.bob_output))
        self.decrypt_err_alice = tf.reduce_mean(tf.square(self.msg - self.alice_output))


        
        # Train Eve for two minibatches to give it a slight computational edge
        no_of_examples = 20
        alice_test_error = []
        bob_test_error = []
        eve_test_error = []
        for i in range(no_of_examples):
            #alice_decrypt_error, bob_decrypt_error, eve_decrypt_error = 0.0, 0.0, 0.0

            t1 = datetime.now()
            
            bs = self.batch_size
            msg_in_val, secret_val, key_val = gen_data(n=bs, msg_len=self.msg_len, secret_len=self.secret_len, key_len=self.random_seed_len)
            
            alice = self.sess.run(self.alice_output,
                                           feed_dict={self.msg: msg_in_val, self.secret: secret_val, self.seed: key_val})
            #eve_decrypt_error = min(eve_decrypt_error, decrypt_err)

            t2 = datetime.now()
            # eve_decrypt_error = eve_decrypt_error + decrypt_err_eve
            # bob_decrypt_error = bob_decrypt_error + decrypt_err_bob
            # alice_decrypt_error = alice_decrypt_error + decrypt_err_alice

            bob = self.sess.run(self.bob_output,
                                           feed_dict={self.alice_output:alice, self.seed: key_val})



            t3 = datetime.now()

           # print msg_in_val[0], key_val[0], alice[0],bob[0],eve[0]
            bob_test_error.append((t3-t2).microseconds*1.0 /bs)
            #eve_test_error.append(eve_decrypt_error)
            alice_test_error.append( (t2-t1).microseconds*1.0 /bs )

            #visualize_image(convToInt(msg_in_val),self.file_name,bs,steg=False)
            #visualize_image(convToInt(alice),self.file_name,bs,steg=True)


        print bob_test_error
        print alice_test_error



        
        print np.mean(bob_test_error), np.std(bob_test_error)
        #print np.mean(eve_test_error), np.std(eve_test_error)
        print np.mean(alice_test_error), np.std(alice_test_error)

        #return bob_decrypt_error, eve_decrypt_error, alice_decrypt_error



    def _train(self, network, iterations):
        bob_decrypt_error, eve_decrypt_error, alice_decrypt_error = 0.0 , 0.0 , 0.0

        bs = self.batch_size
        # Train Eve for two minibatches to give it a slight computational edge
        if network == 'eve':
            bs *= 2

        for i in range(iterations):
            msg_in_val, secret_val, key_val= gen_data(n=bs, msg_len=self.msg_len, secret_len=self.secret_len, key_len=self.random_seed_len)
            if network == 'bob':
                _, decrypt_err_alice, decrypt_err,alice,bob,eve = self.sess.run([self.bob_optimizer, self.decrypt_err_alice, self.decrypt_err_bob,self.alice_output,self.bob_output,self.eve_output],
                                               feed_dict={self.msg: msg_in_val, self.secret: secret_val, self.seed: key_val})
                # bob_decrypt_error = min(bob_decrypt_error, decrypt_err)
                bob_decrypt_error  = bob_decrypt_error + decrypt_err
                alice_decrypt_error = alice_decrypt_error + decrypt_err_alice 

            elif network == 'eve':
                _, decrypt_err,alice,bob,eve = self.sess.run([self.eve_optimizer, self.decrypt_err_eve,self.alice_output,self.bob_output,self.eve_output],
                                               feed_dict={self.msg: msg_in_val, self.secret: secret_val, self.seed: key_val})
                #eve_decrypt_error = min(eve_decrypt_error, decrypt_err)
                eve_decrypt_error = eve_decrypt_error + decrypt_err

            print msg_in_val[0], secret_val[0], alice[0], bob[0], eve[0]

        return bob_decrypt_error/iterations, eve_decrypt_error/iterations , alice_decrypt_error/iterations


    def postTrain(self):
        # Loss Functions
        self.decrypt_err_eve = tf.reduce_mean(tf.abs(self.msg - self.eve_output))
        self.decrypt_err_bob = tf.reduce_mean(tf.abs(self.msg - self.bob_output))
        self.loss_bob = self.bob_factor*self.decrypt_err_bob + self.eve_factor*(1 - self.decrypt_err_eve) ** 2.

        # Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice_' in var.name or 'bob_' in var.name or 'keygen_' in var.name]
        self.eve_vars = [var for var in self.t_vars if 'eve_' in var.name]

        self.eve_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.decrypt_err_eve, var_list=self.eve_vars)

        self.bob_errors, self.eve_errors = [], []

        self.reset_eve_vars = [var for var in tf.all_variables() if 'eve_' in var.name or 'beta' in var.name]
        print self.reset_eve_vars

        # Begin Training
        reset_vars = tf.variables_initializer(self.reset_eve_vars)#[var.initializer for var in self.eve_vars]
        self.sess.run(reset_vars)
        print self.sess.run(tf.report_uninitialized_variables())

        for i in range(self.epochs):
            

            print 'Training Eve, Epoch:', i + 1
            _, eve_loss = self._train('eve', self.iterations)
            self.eve_errors.append(eve_loss)

        self.plot_errors()


    def plot_errors(self):
        """
        Plot Lowest Decryption Errors achieved by Bob and Eve per epoch
        """
        
        print self.eve_errors
        print self.bob_errors
        print self.alice_errors
        f = open(self.file_name+".txt" ,'w')
        for i in self.eve_errors:
            f.write(str(i)+",")
        f.write("\n")
        for i in self.bob_errors:
            f.write(str(i)+",")
        f.write("\n")
        for i in self.alice_errors:
            f.write(str(i)+",")
        f.write("\n")
        f.close()


        sns.set_style("darkgrid")
        plt.plot(self.bob_errors)
        plt.plot(self.eve_errors)
        plt.plot(self.alice_errors)
        plt.legend(['bob', 'eve','alice'])
        plt.xlabel('Epoch')
        plt.ylabel('Decryption error')
        plt.savefig(self.file_name+".png")
        plt.show()

