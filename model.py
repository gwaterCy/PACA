import tensorflow as tf
from layers import *
import numpy as np

class Model(object):
    def __init__(self):
        name = self.__class__.__name__.lower()
        self.name = name

        #define the vars, ploceholders, layers, activations in child class
        self.vars = []

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = None
        self.optimizer = None
        self.opt_op = None
        # self.global_step = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0)

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        variables=tf.trainable_variables()
        self.vars =variables
        # Build metrics
        self._loss()
        self._evaluate()
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class PACA(Model):
    def __init__(self,decay,penalization=True,lambda_lossL2=1e-3,windowSize=3,batch_size=512
                 ,emb_dim=100,num_item=41372,max_length=19,learning_rate=0.001,learningRate_decay=0.1):
        super(PACA, self).__init__()

        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.num_item = num_item
        self.stdv = np.sqrt(2.0/self.emb_dim)
        self.max_length = max_length
        self.learningRate_decay = learningRate_decay
        self.decay =decay
        self.windowSize =windowSize
        self.penalizaiton = penalization
        self.lambda_lossL2 = lambda_lossL2
        #optimal
        # self.learning_rate = learning_rate
        self.learning_rate = tf.train.exponential_decay(learning_rate, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=learningRate_decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        #evaluation metrics
        self.mrr={}
        self.recall={}
        self.ranks=None
        #define placeholder
        self.x = tf.placeholder(dtype=tf.int32, shape=[None,None],name='x') #step*bat
        self.mask = tf.placeholder(dtype=tf.float32,shape=[None,None],name='mask')#step*bat
        self.dropouts = tf.placeholder(dtype=tf.float32, shape=[3], name='drop_out_rates')
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None],name='labels')#bat
        self.inputs = self.x
        #define global varibles

        with tf.variable_scope(self.name ):
            with tf.variable_scope(self.name + '_vars'):
                self.embeddings = tf.get_variable(shape=[num_item, self.emb_dim], name='embeddings', dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            # print(self.embeddings)


        self.build()

    def _build(self):
        self.layers.append(LookingUp())

        self.layers.append(PositionAwareMeanAttention(max_length=self.max_length,
                                                  input_dim = self.emb_dim,
                                                  output_dim= self.emb_dim,
                                                  dropout=self.dropouts[1],
                                                  mask=self.mask))

        self.layers.append(BiLinear(input_dim = self.emb_dim,
                                    output_dim= self.emb_dim,
                                    dropout=self.dropouts[2]))
    def _loss(self):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.outputs))

        if self.penalizaiton:
            loss_layer = PenalizaitonLoss(self.num_step,self.mask)
            self.lossL2 = lossL2= loss_layer(self.activations[1])* self.lambda_lossL2

            # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars]) * self.lambda_lossL2

            self.loss += lossL2

        tf.summary.scalar('loss', self.loss)

    def _evaluate(self):
        tmp = tf.batch_gather(self.outputs,tf.reshape(self.labels,[-1,1]))
        over_tar = self.outputs > tmp # bat*num_item466
        ranks = tf.add(tf.reduce_sum(
            tf.cast(over_tar,dtype=tf.int32),axis=1),1)
        rank_top = [(ranks <= 20),(ranks <= 10),(ranks <= 5),(ranks <= 3)]
        for i in range(4):
            self.recall[i] = tf.reduce_sum(
                tf.cast(rank_top[i], dtype=tf.int32))
            self.mrr[i] = tf.reduce_sum(tf.div(1.0,
                                 tf.cast(tf.boolean_mask(ranks, rank_top[i]),dtype=tf.float32)))
        self.ranks = ranks











