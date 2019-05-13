import tensorflow as tf
import numpy as np

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
            Layers with common name share variables. (TODO)
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self):
        layer = self.__class__.__name__.lower()
        name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class LookingUp(Layer):
    def __init__(self,):
        super(LookingUp,self).__init__()

        with tf.variable_scope("paca_vars",reuse=True):
            self.vars['emb']= tf.get_variable(name='embeddings')

    def _call(self,inputs):
        x=inputs # step_num*batch_size
        session_emb = tf.nn.embedding_lookup(self.vars['emb'], x)
        return session_emb

class PositionAwareAttention(Layer):
    def __init__(self,max_length,input_dim,output_dim,dropout,mask,kernel_size=10,transfer=False):
        super(PositionAwareAttention,self).__init__()

        self.max_length = max_length
        self.input_dim =input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.mask = mask
        self.transfer =transfer
        self.kernel_size = kernel_size

        self.stdv=np.sqrt(2.0/self.input_dim)
        self.initializer = tf.random_uniform_initializer(-self.stdv, self.stdv)

        with tf.variable_scope(self.name+'_vars'):
            if transfer:
                self.vars['w_t'] = tf.get_variable(name="weights_transfer", dtype=tf.float32,shape=[self.input_dim, self.output_dim]
                                                   , initializer=self.initializer)
            else:
                self.output_dim = self.input_dim

            self.vars['w_p']=tf.get_variable(name="weights_position",dtype=tf.float32,shape=[self.kernel_size,self.max_length,self.output_dim]
                            ,initializer=self.initializer)





    def _call(self,inputs):
        session_emb = tf.nn.dropout(inputs, 1 - self.dropout) #step_num*batch_size*dim
        mask = self.mask
        num_step = tf.shape(session_emb)[0]

        if self.transfer:
            session_emb = tf.tensordot(session_emb,self.vars['w_t'],axes=[2,0])
        session_emb = tf.multiply(session_emb,
                                  tf.expand_dims(mask,axis=2))
        # tmp=self.vars['w_p'][self.max_length-num_step:,:]
        tmp_emb = tf.sigmoid(session_emb)

        kernel_emb =[]
        for i in range(self.kernel_size):
            tmp_wp = self.vars['w_p'][i,:num_step, :]
            tmp_wp = tf.expand_dims(tmp_wp, axis=2)
            sim_matrix = tf.matmul(tmp_emb,tmp_wp) #sp*bat*1
            sim_matrix  = tf.reduce_sum(sim_matrix,axis=2)#sp*bat
            kernel_emb.append(sim_matrix)

        sim_matrix = tf.stack(kernel_emb)
        sim_matrix = tf.reduce_max(sim_matrix,axis=0)




        tmp = tf.multiply(sim_matrix,mask) #step*bat
        tmp = tf.nn.softmax(tmp,axis=0)
        att = tf.multiply(tmp,mask)
        p = tf.reduce_sum(att,axis=0,keepdims=True)
        att_alpha = tf.div(att,p) # step*bat
        #att_alpha = tmp
        paa_matrix = tf.multiply(session_emb,tf.expand_dims(att_alpha,axis=2))
        paa_h  = tf.reduce_sum(paa_matrix,axis=0) #bat*dimte
        return paa_h

class BiLinear(Layer):
    def __init__(self,input_dim,output_dim,dropout):
        super(BiLinear,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stdv = np.sqrt(2.0/self.input_dim)
        self.dropout = dropout

        with tf.variable_scope(self.name + '_vars'):
            self.vars['w_b'] = tf.get_variable(name="weights_bilinear", dtype=tf.float32,
                                       shape=[self.input_dim, self.output_dim]
                                       , initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope("paca_vars",reuse=True):
            self.vars['emb']= tf.get_variable(name='embeddings')

    def _call(self,inputs):
        paa_h = tf.nn.dropout(inputs, 1 - self.dropout)
        final_state = tf.matmul(paa_h,self.vars['w_b'])
        # final_state =paa_h
        logits = tf.matmul(final_state,self.vars['emb'],transpose_b=True) #bat*num_item
        return logits

class PositionAwareMeanAttention(Layer):
    def __init__(self,max_length,hidden_dim,dropout,mask):
        super(PositionAwareMeanAttention,self).__init__()

        self.max_length = max_length
        self.hidden_dim  =hidden_dim
        self.stdv=np.sqrt(2.0/self.hidden_dim)
        self.dropout = dropout
        self.mask = mask

        with tf.variable_scope(self.name+'_vars'):
            self.vars['w_p']=tf.get_variable(name="weights_position",dtype=tf.float32,shape=[self.max_length,self.hidden_dim]
                            ,initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.vars['w_k'] = tf.get_variable(name="weights_key", dtype=tf.float32,
                                               shape=[self.hidden_dim, self.hidden_dim]
                                               , initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.vars['w_q'] = tf.get_variable(name="weights_query", dtype=tf.float32,
                                               shape=[self.hidden_dim, self.hidden_dim]
                                               , initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))


    def _call(self,inputs):
        session_emb = tf.nn.dropout(inputs, 1 - self.dropout) #step_num*batch_size*dim
        mask = self.mask
        num_step = tf.shape(session_emb)[0]

        session_emb = tf.multiply(session_emb,
                                  tf.reshape(mask,[num_step,-1,1]))
        session_mean = tf.reduce_sum(session_emb,axis=0,keepdims=True)


        key = tf.tensordot(session_emb,self.vars['w_k'],axes=[2,0])
        query = tf.tensordot(session_mean,self.vars['w_q'],axes=[2,0])

        tmp_emb = tf.nn.sigmoid(
                                tf.add(key,query) )

        tmp=self.vars['w_p'][:num_step,:]
        w_pr= tf.reshape(tmp,[num_step,-1,1])
        # session_emb = tf.nn.sigmoid(session_emb)
        sim_matrix = tf.matmul(tmp_emb,w_pr) #step*bat*1
        sim_matrix  = tf.reduce_sum(sim_matrix,axis=2)

        tmp = tf.multiply(sim_matrix,mask) #step*bat
        tmp = tf.nn.softmax(tmp,axis=0)
        att = tf.multiply(tmp,mask)
        p = tf.reduce_sum(att,axis=0,keepdims=True)
        att_alpha = tf.div(att,p) # step*bat
        #att_alpha = tmp
        paa_matrix = tf.multiply(session_emb,tf.reshape(att_alpha,[num_step,-1,1]))
        paa_h  = tf.reduce_sum(paa_matrix,axis=0) #bat*dimte
        return paa_h

class PenalizaitonLoss(Layer):
    def __init__(self,mask):
        super(PenalizaitonLoss,self).__init__()

        self.mask = mask


    def _call(self,inputs):
        session_emb =inputs
        mask = self.mask
        bat = tf.shape(session_emb)

        session_emb = tf.multiply(session_emb,tf.expand_dims(mask,2))
        session_mean = tf.reduce_sum(session_emb,axis=0,keepdims=True)

        delta_emb = tf.add(session_emb,-1.0*session_mean)
        delta_emb = tf.multiply(delta_emb,tf.expand_dims(mask,2))# n*bat*dim
        cluster_penalizaton = tf.div(tf.div(tf.nn.l2_loss(delta_emb),tf.cast(bat[1],dtype=tf.float32))
                                     ,tf.cast(bat[0],dtype=tf.float32))

        # cluster_penalizaton = tf.div(tf.nn.l2_loss(delta_emb), tf.reduce_sum(mask,axis=[0,1]) )

        # delta_emb = tf.multiply(session_emb,session_mean)
        # delta_emb = tf.reduce_sum(delta_emb,axis=2)
        # cluster_penalizaton = tf.div(tf.nn.l2_loss(delta_emb), tf.reduce_sum(mask,axis=[0,1]) )

        return cluster_penalizaton















