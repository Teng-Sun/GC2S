import tensorflow as tf
import numpy as np
import helper_sim as helper
from IPython import embed
print(tf.__version__)
from gensim import models
import random

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
tf.reset_default_graph()
#sess = tf.InteractiveSession()
BOS = 0
EOS = 1
PAD = 3
UNK = 2

class gc2s() :
    def __init__(self, batch_size):
        #settings
        #self.vocab_size = 6000
        self.batch_size = batch_size
        self.input_length = 9
        #self.decoder_length = 30
        self.embedding_size = 250
        #self.attr_col = 3
        self.attr_col = 2
        self.K = 11
        self.context_embed_size = 20
        self.decoder_hidden_units = 256
        #self.EPOCHS = 50
        self.FIT_ON = True
        self.TEMP = 0.7
        self.random_sampling = True
        self.max_length = 50

        self.LEARNING_RATE = 0.8
        self.DECAY = 0.7
        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5
        self.MODEL = models.Word2Vec.load("data/new_train_ckip.model.bin")
        #self.MODEL = models.Word2Vec.load("data/seg_data.model.bin")
        self.vocab_size = len(self.MODEL.wv.index2word)+4
        self.make_graph()

    def make_graph(self):
        self._init_placeholder()
        self._init_variables()
        self.connector_to_decoder()
        self._init_embedding()
        self._init_decoder()
        self._init_optimizer()

    def _init_placeholder(self):
        self.threshold = tf.placeholder(shape=(), dtype=tf.float32,name='threshold')

        self.temperature = tf.constant(self.TEMP, shape=[1])
        self.inputs = tf.placeholder(shape=(None, None), \
                                        dtype =tf.int32, name='inputs')

        self.decoder_targets =tf.placeholder(shape=(None, self.batch_size), \
                                        dtype = tf.int32, name ='decoder_targets')

        self.decoder_lengths = tf.placeholder(shape=(None, ), dtype=tf.int32,
                                                    name='decoder_lengths')

    def connector_to_decoder(self):

        #transform the semantic frame to semantic representation eor decoder
        '''
        E_list = []
        for _ in range(self.attr_col):
            E_list.append(
                tf.Variable(tf.random_uniform([self.K, self.context_embed_size],\
                                                -0.1, 0.1), dtype = tf.float32)
        )
        #Encoder
        input_matrix = tf.one_hot(self.inputs,depth=self.K, dtype=tf.float32)

        input_T = tf.transpose(input_matrix, [1, 0, 2])

        batch_size X attr_col * K
        |
        V
        attr_col X batch_size X K

        concat_E = tf.matmul(input_T[0], E_list[0])
        for i in range(1, self.attr_col):
            concat_E = tf.concat([concat_E, tf.matmul(input_T[i], E_list[i])], axis=1)
        '''

        input1 = tf.one_hot(tf.transpose(self.inputs)[0], depth=1200, dtype=tf.float32)
        w1 =  tf.Variable(tf.random_uniform([1200, self.context_embed_size],\
                                            -0.1, 0.1),dtype=tf.float32)
        #input1_T = tf.transpose(input1, [1, 0, 2])
        concat_E = tf.matmul(input1, w1)
        input2 = tf.one_hot(tf.transpose(self.inputs)[1], depth=11, dtype=tf.float32)
        w2 =  tf.Variable(tf.random_uniform([self.K, self.context_embed_size],\
                                            -0.1, 0.1),dtype=tf.float32)
        #input2_T = tf.transpose(input2, [1, 0, 2])
        concat_E = tf.concat([concat_E, tf.matmul(input2, w2)], axis=1)

        #concat_E = tf.reshape(tf.matmul(input_flat, encoder_E), (batch_size, -1))
        print('conat_e==>', concat_E)
        hc = tf.nn.tanh(tf.add(tf.matmul(concat_E, self.encoder_W), self.encoder_b))
        self.encoder_state = LSTMStateTuple(
            c=hc,
            h=hc
        )
        self.eos_step_slice = tf.ones([1,self.batch_size], dtype=tf.int32, name='EOS')
        self.bos_step_slice = tf.zeros([1,self.batch_size], dtype=tf.int32, name='BOS')

        self.decoder_inputs_train = tf.concat([self.bos_step_slice, self.decoder_targets], axis=0)
        self.decoder_targets_train = tf.concat([self.decoder_targets, self.eos_step_slice],axis=0)


        #loss_wights= tf.ones([batch_size, ])
    def _init_variables(self):
        self.encoder_W = tf.Variable(tf.random_uniform(
            [(self.context_embed_size*self.attr_col), self.decoder_hidden_units],
            -0.1, 0.1), dtype = tf.float32)
        self.encoder_b = tf.Variable(tf.zeros([self.decoder_hidden_units]), dtype = tf.float32)

        self.W = tf.Variable(tf.random_uniform([self.decoder_hidden_units, self.vocab_size], -0.1, 0.1) \
                                            ,dtype = tf.float32)
        self.b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32)

        self.V = tf.Variable(tf.random_uniform([self.decoder_hidden_units, self.decoder_hidden_units],
                                                    -0.1, 0.1), dtype=tf.float32)
        self.bv = tf.Variable(tf.zeros([self.decoder_hidden_units]), dtype=tf.float32)

    def _init_embedding(self):
        self.embedding = np.random.rand(self.vocab_size, self.embedding_size).astype(np.float32)
        #import pickle
        #with open('tokenizer.pkl', 'rb') as f:
        #    tokenizer = pickle.load(f)
        self.dict_ ={v:i+4 for i,v in enumerate(self.MODEL.wv.index2word)}
        self.dict_.update({'BOS':0,'EOS':1,'UNK':2, 'PAD':3})
        self.reverse = {v:i for i, v in self.dict_.items()}
        #self.reverse.update({0:'BOS'})
        for i in range(4, len(list(self.reverse))):
            if self.reverse[i] not in self.MODEL.wv :
                self.embedding[i] = self.embedding[2]
                embed()
            else :
                self.embedding[i] = self.MODEL.wv[self.reverse[i]].astype(np.float32)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs_train)
        self.eos_step_embedded = tf.nn.embedding_lookup(
            self.embedding, tf.ones(shape=[self.batch_size], dtype=tf.int32))
        self.bos_step_embedded = tf.nn.embedding_lookup(
            self.embedding, tf.zeros(shape=[self.batch_size], dtype=tf.int32))

    def to_zh(self, string):
        return ''.join([self.reverse[c] for c in string])

    def _init_decoder(self) :
        def projection(outputs):
            output_logits = tf.add(tf.matmul(outputs, self.W), self.b)
            return output_logits

        def attention(ht):
            mt = tf.sigmoid(tf.add(tf.matmul(ht.c, self.V), self.bv))
            #print('ht.c-->', ht.c)
            #print('hc-->', hc)
            #print('mt-->', mt)
            #assert tf.shape(mt) == tf.shape(hc)
            c = tf.add(ht.c, tf.multiply(mt, hc))
            return LSTMStateTuple(c=c, h=ht.h)
        with tf.variable_scope('Decoder') as scope :
            self.decoder_cell = LSTMCell(self.decoder_hidden_units)

            def loop_fn_initial():
                initial_element_finished = (0 > self.decoder_lengths)
                print(initial_element_finished)
                initial_input = self.bos_step_embedded
                initial_state = self.encoder_state
                initial_cell_output = None
                initial_loop_state = None
                return (initial_element_finished,
                                    initial_input,
                                    initial_state,
                                    initial_cell_output,
                                    initial_loop_state)
            def loop_fn_transition(time, cell_output, cell_state, loop_state) :
                def get_next():
                    def fn_force() :
                        tf.Print(time,[time])
                        time_ = tf.add(time,1)
                        return self.decoder_inputs_embedded[time_]
                    def fn_feed() :
                        tf.Print(time,[time],'Forcing')
                        output_logits = projection(cell_output)
                        '''
                        scale =tf.div(output_logits, self.temperature)
                        exp = tf.exp(tf.subtract(scale, tf.reduce_max(scale,axis=0, keep_dims=True)))
                        soft = tf.div(exp, tf.reduce_max(exp, axis=0))
                        samples =tf.multinomial(soft, 1)
                        prediction = tf.argmax(samples, 1)
                        '''
                        prediction = tf.argmax(output_logits, 1)
                        next_input = tf.nn.embedding_lookup(self.embedding, prediction)
                        return next_input
                    coin = tf.greater(x=tf.random_uniform((), 0, 1), y=self.threshold)
                    #coin_ = tf.Print(coin,[coin])
                    #return tf.cond(coin, fn_force, fn_feed)
                    return tf.cond(coin, fn_feed, fn_feed)

                element_finished = (time >= self.decoder_lengths)
                finished = tf.reduce_all(element_finished)
                loop_state = None
                input_ = tf.cond(finished, lambda: self.eos_step_embedded, get_next)
                #next_state = attention(cell_state)
                next_state = cell_state
                output = cell_output
                return (element_finished,
                                    input_,
                                    next_state,
                                    output,
                                    loop_state)

            def loop_fn(time, cell_output, cell_state, loop_state):
                if cell_state == None : #first time stamp
                    assert cell_output is None and cell_state is None
                    #assert time == 0
                    return loop_fn_initial()
                else :
                    return loop_fn_transition(time, cell_output, cell_state, loop_state)

            self.decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn)
            self.decoder_outputs = self.decoder_outputs_ta.stack()
            decoder_time_stamp, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs))
            decoder_output_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
            decoder_logit_flat = projection(decoder_output_flat)
            self.decoder_logit = tf.reshape(decoder_logit_flat,
                            (decoder_time_stamp, decoder_batch_size, self.vocab_size))
            self.decoder_prediction = tf.argmax(self.decoder_logit, -1)

            scope.reuse_variables()

            def loop_fn_transition_infer(time, cell_output, cell_state, loop_state) :
                def get_next():
                    output_logits = projection(cell_output)
                    scale =tf.div(output_logits, self.temperature)
                    exp = tf.exp(tf.subtract(scale, tf.reduce_max(scale,axis=0, keep_dims=True)))
                    soft = tf.div(exp, tf.reduce_max(exp, axis=0))
                    samples =tf.multinomial(soft, 1)
                    prediction = tf.argmax(samples, 1)
                    next_input = tf.nn.embedding_lookup(self.embedding, prediction)
                    return next_input

                element_finished = (time >= self.decoder_lengths)
                finished = tf.reduce_all(element_finished)
                loop_state = None
                input_ = tf.cond(finished, lambda: self.eos_step_embedded, get_next)
                #next_state = attention(cell_state)
                next_state = cell_state
                output = cell_output
                return (element_finished,
                                    input_,
                                    next_state,
                                    output,
                                    loop_state)

            def loop_fn_infer(time, cell_output, cell_state, loop_state):
                if cell_state == None : #first time stamp
                    assert cell_output is None and cell_state is None
                    return loop_fn_initial()
                else :
                    return loop_fn_transition_infer(time, cell_output, cell_state, loop_state)

            self.decoder_outputs_ta_infer, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn_infer)
            self.decoder_outputs_infer = self.decoder_outputs_ta_infer.stack()
            decoder_time_stamp, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs_infer))
            decoder_output_flat_infer = tf.reshape(self.decoder_outputs_infer, (-1, decoder_dim))
            decoder_logit_flat_infer = projection(decoder_output_flat_infer)
            self.decoder_logit_infer = tf.reshape(decoder_logit_flat_infer,
                            (decoder_time_stamp, decoder_batch_size, self.vocab_size))
            self.decoder_prediction_infer = tf.argmax(self.decoder_logit_infer, -1)

    def _init_optimizer(self):

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(\
                    labels = tf.one_hot(self.decoder_targets,depth=self.vocab_size, dtype=tf.float32),
                    logits = self.decoder_logit \
        )
        self.loss = tf.reduce_mean(stepwise_cross_entropy)
        opt=tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)
        #self.train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        # add
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
        self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

class My():
    def __init__(self, threshold=0.5, batch_size=128):
        #self.test = test
        self.threshold= threshold
        tf.reset_default_graph()
        self.MODEL = models.Word2Vec.load("data/seg_data.model.bin")
        self.model_path=('./model_2/')
        self.EPOCHS = 1000
        #import pickle
        #with open('tokenizer.pkl', 'rb') as f :
        #    self.tokenizer=pickle.load(f)

        #self.reverse = {v:i for i,v in self.tokenizer.word_index.items()}
        self.model = gc2s(batch_size=batch_size)

        self.gen_batch = helper.generate(self.model.batch_size,
                                         'without_idx')

        self.TIME = next(self.gen_batch)

    def next_feed(self):
        inputs_, batch_data = next(self.gen_batch)
        decoder_targets_, seq_lengths = helper.to_batch(batch_data, None)
        decay = (1.0/self.threshold) ** (1 / (self.EPOCHS *0.6))
        thr = self.threshold * pow(decay, self.time)
        return {
            self.model.inputs:inputs_,
            self.model.decoder_targets:decoder_targets_,
            self.model.decoder_lengths:seq_lengths,
            self.model.threshold: thr}

    def infer_feed(self):
        inputs_, batch_data = next(self.gen_batch)
        '''
        inputs_ = []
        tmp = []
        s= ['hotel ave_score','usr comment score','staff quality score','cleaness score','location score',
            'wifi satisfy score','total equipments score','comfort score', 'cp score']
        for score in s:
            print('give %s' %score)
            a =input('>>')
            tmp.append(int(a))
        inputs_.append(tmp)
        inputs_.append(tmp)
        inputs_ = np.asarray(inputs_)
        '''
        decoder_targets_ = np.ones(shape=[0, 2], dtype=np.int32)
        seq_lengths = np.ones(shape=[2]) * 10

        return {
            self.model.inputs:inputs_,
            self.model.decoder_targets:decoder_targets_,
            self.model.decoder_lengths:seq_lengths,
            self.model.threshold: 1}

    def test(self) :

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.visible_device_list = "2"

        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception ('No model for testing')
            for epoch in range(1) :
                print('Epoch {}'.format(epoch))
                for b in range(self.TIME) :
                    fd = self.infer_feed()
                    #embed()
                    predict_ = sess.run(self.model.decoder_prediction_infer, fd)

                    for i in range(2):
                        print(fd[self.model.inputs][i])
                        print(self.model.to_zh(predict_.T[i]))
    def train(self) :

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.visible_device_list = "0"

        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
                print('No model create new')

            loss_track = []
            for epoch in range(self.EPOCHS) :
                self.time = epoch
                print('Epoch {}'.format(epoch))
                for b in range(self.TIME) :
                    fd = self.next_feed()
                    #embed()
                    _, loss_ = sess.run([self.model.train_op, self.model.loss], fd)

                    if b == 0  or b % 20 == 0:
                        print('     batch {}'.format(b))
                        print('     minibatch loss :{}'.format(loss_))
                        predict_ = sess.run(self.model.decoder_prediction, fd)

                        #print('     forcing {}'.format(fd[self.model.forcing]))
                        for i, (inp, pred, tar) in \
                                enumerate(zip(fd[self.model.inputs],
                                              predict_.T,
                                              fd[self.model.decoder_targets].T)):
                            print(' sample {}'.format(i))
                            #print('     forcing {}'.format(fd[self.model.threshold]))
                            print('     inputs: {}'.format(inp))
                            print('     prediction:{}'.format(self.model.to_zh(pred)))
                            print('     targets :{}'.format(self.model.to_zh(tar)))
                            if i > 2 :
                                break
            checkpoint_path = self.model_path+"gc2s.ckpt"
            self.model.saver.save(sess, checkpoint_path)
            print('save model %s'%self.model_path)
if __name__ == '__main__' :
    import sys
    if sys.argv[1] == 'train' :
        th = float(sys.argv[2])
        gc2s = My(threshold=th, batch_size=128)
        gc2s.train()
    elif sys.argv[1] == 'test' :
        gc2s = My(threshold=1, batch_size=2)
        gc2s.test()
