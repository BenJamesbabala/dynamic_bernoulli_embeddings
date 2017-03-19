from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import pickle
import tensorflow as tf
import time

from data import *
from models import *
from args import *

args = parse_args()
print(args)

# DATA
d = bern_emb_data(args.cs, args.ns, args.fpath, args.dynamic, args.n_epochs)

dir_name = 'fits/'+d.name+'/EF_EMB_' +  time.strftime("%y_%m_%d_%H_%M_%S")
while os.path.isdir(dir_name):
    time.sleep(np.random.randint(10))
    dir_name = 'fits/'+d.name+'/EF_EMB_' +  time.strftime("%y_%m_%d_%H_%M_%S")
os.makedirs(dir_name)

# MODEL
if args.dynamic:
    if args.init:
        if 'alpha_constant' in args.init:
            fit = pickle.load(open(os.path.join('fits', d.name, args.init.replace('/alpha_constant','') ,'variational.dat')))
            m = dynamic_bern_emb_model(args.K, d, d.n_train, args.cs, args.ns, args.sig, args.lam, fit['alpha'])
        else:
            fit = pickle.load(open(os.path.join('fits', d.name, args.init ,'variational.dat')))
            m = dynamic_bern_emb_model(args.K, d, d.n_train, args.cs, args.ns, args.sig, args.lam, fit['alpha'], fit['rho'])
    else:
        m = dynamic_bern_emb_model(args.K, d, d.n_train, args.cs, args.ns, args.sig, args.lam)
else:
    if args.init:
        fit = pickle.load(open(os.path.join('fits', d.name, args.init ,'variational.dat')))
        m = bern_emb_model(args.K, d, d.n_train, args.cs, args.ns, args.lam, fit['alpha'], fit['rho'])
    else:
         m = bern_emb_model(args.K, d, d.n_train, args.cs, args.ns, args.lam)

# MAP INFERENCE
inference = ed.MAP({}, data = m.data)

sess = ed.get_session()
inference.initialize(optimizer=tf.train.AdagradOptimizer(learning_rate=args.eta))
tf.initialize_all_variables().run()

print('\n \n training is starting\n')
# TRAINING
train_loss = np.zeros(args.n_iter)
with open(dir_name+"/log_file.txt", "w") as text_file:
    text_file.write(str(args)+'\n')
    for i in range(args.n_iter):
        for ii in range(args.n_epochs - 1):
            print(str(ii)+'/'+str(args.n_epochs)+'   iter'+str(i))
            sess.run([inference.train], feed_dict=d.train_feed(m.placeholders))
        _, train_loss[i] = sess.run([inference.train, inference.loss], feed_dict=d.train_feed(m.placeholders))
        m.dump(dir_name+"/variational"+str(i)+".dat")
        text_file.write("iteration {:d}/{:d}, train loss: {:0.3f}\n".format(i, args.n_iter, train_loss[i])) 

    # MODEL EVALUATION
    if args.dynamic:
        m_test = dynamic_bern_emb_model(args.K, d, d.n_test, args.cs, 0, args.sig, args.lam, m.alpha.eval(), m.rho.eval())
        m_valid = dynamic_bern_emb_model(args.K, d, d.n_valid, args.cs, 0, args.sig, args.lam, m.alpha.eval(), m.rho.eval())
    else:
        m_test = bern_emb_model(args.K, d, d.n_test, args.cs, 0, args.lam, m.alpha.eval(), m.rho.eval())
        m_valid = bern_emb_model(args.K, d, d.n_valid, args.cs, 0, args.lam, m.alpha.eval(), m.rho.eval())

    tf.initialize_variables([m_test.alpha, m_valid.alpha, m_test.rho, m_valid.rho]).run()
    test_loss = m_test.eval_log_like(d.test_feed(m_test.placeholders), sess)
    valid_loss = m_valid.eval_log_like(d.valid_feed(m_valid.placeholders), sess)

    text_file.write("valid loss: {:0.5f} +- {:0.5f}\n".format(np.mean(valid_loss),np.std(valid_loss)/np.sqrt(len(valid_loss)))) 
    text_file.write("test loss: {:0.5f} +- {:0.5f}\n".format(np.mean(test_loss),np.std(test_loss)/np.sqrt(len(test_loss)))) 

# SAVE RESULTS
with open("fits/"+d.name+"/log_file.txt", "a") as text_file:
    text_file.write('\n'+dir_name+'\t')
    text_file.write("valid loss: {:0.5f} +- {:0.5f}\t".format(np.mean(valid_loss),np.std(valid_loss)/np.sqrt(len(valid_loss)))) 
    text_file.write("test loss: {:0.5f} +- {:0.5f}\t".format(np.mean(test_loss),np.std(test_loss)/np.sqrt(len(test_loss)))) 
    text_file.write(str(args)+'\n')

m.dump(dir_name+"/variational.dat")
m.print_word_similarities(d.query_words, d, 10, dir_name, sess)
if args.dynamic:
    words = m.detect_drift(d, dir_name, sess)
    m.print_word_similarities(words[:50], d, 10, dir_name, sess)
m.plot_params(dir_name, d, 500)
