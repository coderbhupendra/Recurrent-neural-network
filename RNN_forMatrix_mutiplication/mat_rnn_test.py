import theano
import theano.tensor as T
import numpy as np
import os.path
import cPickle,pickle
import random
import matplotlib.pyplot as plt
 
class RNN(object):
 
	def __init__(self, nin, n_hidden, nout):
		


		lr = T.scalar()
		w = T.matrix()
		t = T.vector()
 		self.activ = T.nnet.sigmoid
		if os.path.exists('W_uh.pkl'):
			with open('W_uh.pkl') as f:
				print"yes weights found"
				w1 = theano.shared(cPickle.load(f), name='W_uh')
				W_uh = w1
			with open('W_hh.pkl') as f:
				w2 = theano.shared(cPickle.load(f), name='W_hh')
				W_hh = w2
			with open('W_hy.pkl') as f:
				w3 = theano.shared(cPickle.load(f), name='W_hy')
				W_hy= w3
			with open('b_hh.pkl') as f:
				w4 = theano.shared(cPickle.load(f), name='b_hh')
				b_hh = w4    
			with open('b_hy.pkl') as f:
				w5 = theano.shared(cPickle.load(f), name='b_hy')
				b_hy = w5
		else:    print"no weights found"


		h0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
 
		h, _ = theano.scan(self.recurrent_fn, sequences = w,
					   outputs_info = [h0_tm1],
					   non_sequences = [W_hh, W_uh, W_hy, b_hh])
 
		y = T.dot(h[-1], W_hy) + b_hy

		cost = y
 
		self.train_step = theano.function([w], cost,
							on_unused_input='warn',
							
							allow_input_downcast=True)
		
	def recurrent_fn(self, u_t, h_tm1, W_hh, W_uh, W_hy, b_hh):
		h_t = self.activ(T.dot(h_tm1, W_hh) + T.dot(u_t, W_uh) + b_hh)
		
		return h_t

	def get_param(self):
		return self.param

if __name__ == '__main__':
	rnn = RNN(6, 20, 9)
	lr = 0.01
	e = 1
	vals = []
	for i in xrange(int(5)):
		u=np.random.rand(3,3)
		v=np.random.rand(3,3)
		#u=np.array([[.1,.1,.1],[.2,.2,.2],[.3,.3,.3]])
		#v=np.array([[.1,.1,.1],[.2,.2,.2],[.3,.3,.3]])
		w=np.concatenate((u,v),axis=1)

		t = (np.dot(u,v)).flatten()

		c = rnn.train_step(w)
		print "{0} \n: {1} ".format(t.reshape(3,3),c.reshape(3,3))
		#print "{0}\n".format((t-c))
		


