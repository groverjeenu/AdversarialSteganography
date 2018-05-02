import tensorflow as tf

from argparse import ArgumentParser
from src.model import CryptoNet
from src.config import *


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='message length',
                        metavar='MSG_LEN', default=MSG_LEN)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)


    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)

        crypto_net.train()
        saver = tf.train.Saver()
        saver.save(sess,'models/' + crypto_net.file_name)

        #tf.graph().as_default()

        return crypto_net.file_name

def test(fn=""):
    parser = build_parser()
    options = parser.parse_args()

    

    with tf.Session() as sess:

        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)
        saver = tf.train.Saver()
        saver.restore(sess,'models/' + fn)
        crypto_net.test()
        fr =tf.summary.FileWriter('log',sess.graph)

def test_(fn=""):
    parser = build_parser()
    options = parser.parse_args()

    

    with tf.Session() as sess:

        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)
        saver = tf.train.Saver()
        saver.restore(sess,'models/' + fn)
        crypto_net.test_()
        fr =tf.summary.FileWriter('log',sess.graph)
                

def postTrain():
    parser = build_parser()
    options = parser.parse_args()

    with tf.Session() as sess:
        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)
        saver = tf.train.Saver()
        saver.restore(sess,'models/' + crypto_net.file_name_without_time+'2017-11-09 10:16:42.134883')
        crypto_net.postTrain()



if __name__ == '__main__':
    #fn = main()
    #fn = "2018-04-10 16:32:18.966697time3072msg400secret32key16seed128iters40epochs0.0005learnrate0.008evelearnrate512batchsize50timesAliceError100timesBobError1timesEveError100timesEveOptError"
    #fn = "2018-04-06 09:51:42.131772time3072msg100secret32key16seed128iters40epochs0.0005learnrate0.008evelearnrate512batchsize50timesAliceError100timesBobError1timesEveError100timesEveOptError"
    fn = "2018-04-17 18:30:05.063198time3072msg100secret32key32seed128iters40epochs0.0005learnrate0.008evelearnrate512batchsize50timesAliceError100timesBobError1timesEveError100timesEveOptError"
    test(fn)

