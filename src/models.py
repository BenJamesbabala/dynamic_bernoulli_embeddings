import edward as ed
import numpy as np
import os
import pickle
import tensorflow as tf

from edward.models import Normal, Bernoulli
from sklearn.manifold import TSNE
from utils import plot_with_labels


class emb_model(object):
    def __init__(self):
        raise NotImplementedError()

    def dump(self, fname):
        raise NotImplementedError()

    def detect_drift(self, d, dir_name, sess):
        raise NotImplementedError()

    def eval_log_like(self, feed_dict, sess):
        return sess.run(tf.log(self.y_pos.mean()+0.000001), feed_dict = feed_dict)

    def plot_params(self, dir_name, d, plot_only=500):

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_alpha2 = tsne.fit_transform(self.alpha.eval()[:plot_only])
        plot_with_labels(low_dim_embs_alpha2[:plot_only], d.labels[:plot_only], dir_name + '/alpha.eps')

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_rho2 = tsne.fit_transform(self.rho.eval()[:plot_only])
        plot_with_labels(low_dim_embs_rho2[:plot_only], d.labels[:plot_only], dir_name + '/rho.eps')

    def print_word_similarities(self, words, d, num, dir_name, sess):
        """
        prints pairs which large inner products alpha`rho and rho`alpha 
        """
        query_word = ed.placeholder(dtype=tf.int32)

        unigram = tf.tile(tf.expand_dims(tf.constant(d.unigram.astype('float32')), [1]), [1, self.K])
        query_rho = tf.gather(self.rho, query_word)
        val_rho, idx_rho = tf.nn.top_k(tf.matmul(query_rho, tf.multiply(self.alpha, unigram), transpose_b=True), num)

        query_alpha = tf.gather(self.alpha, query_word)
        val_alpha, idx_alpha = tf.nn.top_k(tf.matmul(query_alpha, tf.multiply(self.rho, unigram), transpose_b=True), num)

        w_idx = np.array([d.dictionary[x] for x in words])

        vr, ir, va, ia = sess.run([val_rho, idx_rho, val_alpha, idx_alpha], {query_word: w_idx})

        with open(os.path.join(dir_name,'rho_queries.txt'), "w+") as text_file:
            for i in xrange(len(words)):
                text_file.write("\n\n=====================================\n%s\n=====================================" % (words[i]))
                for (nbr, dist) in zip(ir[i, :num], vr[i, :num]):
                    text_file.write("\n%-20s %6.4f" % (d.labels[nbr], dist))

        with open(os.path.join(dir_name,'alpha_queries.txt'), "w+") as text_file:
            for i in xrange(len(words)):
                text_file.write("\n\n=====================================\n%s\n=====================================" % (words[i]))
                for (nbr, dist) in zip(ia[i, :num], va[i, :num]):
                    text_file.write("\n%-20s %6.4f" % (d.labels[nbr], dist))

    def print_topics(self, d, num, dir_name, sess):
        _, idx_rho = tf.nn.top_k(tf.transpose(self.rho[:500]), num)
        _, idx_alpha = tf.nn.top_k(tf.transpose(self.alpha[:500]), num)
        _, neg_idx_rho = tf.nn.top_k(tf.transpose(-self.rho[:500]), num)
        _, neg_idx_alpha = tf.nn.top_k(tf.transpose(-self.alpha[:500]), num)

        ir, ia, nir, nia = sess.run([idx_rho, idx_alpha, neg_idx_rho, neg_idx_alpha])

        with open(os.path.join(dir_name,'rho_topics.txt'), "w+") as text_file:
            for i in xrange(self.K):
                text_file.write("\n\n===================\n topic %d\n===================" % (i))
                for w_idx in ir[i, :num]:
                    text_file.write("\n%-20s" % (d.labels[w_idx]))
                text_file.write("\n...")
                for w_idx in reversed(nir[i, :num]):
                    text_file.write("\n%-20s" % (d.labels[w_idx]))

        with open(os.path.join(dir_name,'alpha_topics.txt'), "w+") as text_file:
            for i in xrange(self.K):
                text_file.write("\n\n===================\n topic %d\n===================" % (i))
                for w_idx in ia[i, :num]:
                    text_file.write("\n%-20s" % (d.labels[w_idx]))
                text_file.write("\n...")
                for w_idx in reversed(nia[i, :num]):
                    text_file.write("\n%-20s" % (d.labels[w_idx]))


class bern_emb_model(emb_model):
    def __init__(self, K, d, n_minibatch, cs, ns, lam=10000.0, alpha_init = False, rho_init = False):
        self.K = K
        self.n_minibatch = n_minibatch
        self.cs = cs
        self.ns = ns
        self.lam = lam
        

        # Data Placeholder
        self.words = ed.placeholder(tf.int32, shape = (self.n_minibatch + self.cs))
        self.placeholders = self.words
        

        # Index Masks
        self.p_mask = tf.range(self.cs/2, self.n_minibatch + self.cs/2)
        rows = tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch, 1])
        columns = tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch), [1]), [1, self.cs/2])
        self.ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

        # Embedding vectors
        if rho_init is not False:
            self.rho = tf.Variable(rho_init)
        else:
            self.rho = tf.Variable(0.1*tf.random_normal([d.L, self.K])/self.K)

        # Context vectors
        if alpha_init is not False:
            self.alpha = tf.Variable(alpha_init)
        else:
            self.alpha = tf.Variable(0.1*tf.random_normal([d.L, self.K])/self.K)


        self.prior_rho = Normal(mu = self.rho,
                                sigma = (np.sqrt(1.0*d.N/self.n_minibatch) * self.lam).astype('float32') 
                                         * tf.ones([d.L, self.K]))

        self.prior_alpha = Normal(mu = self.alpha,
                                  sigma = (np.sqrt(1.0*d.N/self.n_minibatch) * self.lam).astype('float32') 
                                           * tf.ones([d.L, self.K]))

        self.data = {self.prior_rho: tf.zeros((d.L, self.K)),
                     self.prior_alpha: tf.zeros((d.L, self.K))}

        # Taget and Context Indices
        self.p_idx = tf.gather(self.words, self.p_mask)
        self.ctx_idx = tf.squeeze(tf.gather(self.words, self.ctx_mask))
        
        # Negative samples
        unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(d.unigram)), [0]), [self.n_minibatch, 1])
        self.n_idx = tf.multinomial(unigram_logits, self.ns)

        self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)

        self.p_rho = tf.squeeze(tf.gather(self.rho, self.p_idx))
        self.n_rho = tf.gather(self.rho, self.n_idx)

        # Natural parameter
        if self.n_minibatch == 1:
            ctx_sum = tf.reduce_sum(self.ctx_alphas,[0])
            p_eta = tf.matmul(tf.expand_dims(self.p_rho,0), tf.expand_dims(ctx_sum,1))
            n_eta = p_eta
        else:
            ctx_sum = tf.reduce_sum(self.ctx_alphas,[1])
            p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(self.p_rho, ctx_sum),-1),1)
            n_eta = tf.reduce_sum(tf.multiply(self.n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
        
        # Conditional likelihood
        self.y_pos = Bernoulli(logits = p_eta)
        self.y_neg = Bernoulli(logits = n_eta)

        self.data = {self.y_pos: tf.ones((self.n_minibatch, 1)), self.y_neg: tf.zeros((self.n_minibatch, self.ns))}


    def dump(self, fname):
            dat = {'rho':  self.rho.eval(),
                   'alpha':  self.alpha.eval()}
            pickle.dump( dat, open( fname, "a+" ) )


class dynamic_bern_emb_model(emb_model):
    def __init__(self, K, d, n_minibatch, cs, ns, sig, lam, alpha_init=False, rho_init=False):
        self.K = K
        self.n_minibatch = n_minibatch
        self.cs = cs
        self.ns = ns
        self.T = len(self.n_minibatch)
        self.sig = sig
        self.lam = lam
        self.alpha_fixed = False

        # Embedding vectors
        if rho_init is not False:
            if len(rho_init.shape) > 2:
                self.rho = tf.Variable(tf.squeeze(rho_init))
            else:
                self.rho = tf.Variable(tf.concat([tf.zeros([1, d.L, self.K]),
                                              tf.tile(tf.expand_dims(rho_init,[0]),[self.T, 1, 1])], 0))
            if alpha_init is not False:
                self.alpha = tf.Variable(alpha_init)
                print('rho and alpha initialized from fit and trainable')
            else:
                self.alpha = tf.Variable(tf.random_normal([d.L, self.K])/self.K)
                print('rho initialized from fit and trainable, alpha initialized at random, trainable')
        else:
            self.rho = tf.Variable(tf.random_normal([self.T+1, d.L, self.K])/self.K)
            if alpha_init is not False:
                print('rho initialized at random and trainable, alpha initialized from fit, constant')
                self.alpha = tf.Variable(alpha_init, trainable=False)
                self.alpha_fixed = True
            else:
                self.alpha = tf.Variable(tf.random_normal([d.L, self.K])/self.K)
                print('rho and alpha initialized at random and trainable')

        # Prior on temporal dynamics
        variance = np.sqrt(1.0*d.N/self.n_minibatch) * 2.0 * self.sig
        variance[0] = 1000*variance[0]
        sigma = tf.tile(tf.expand_dims(tf.expand_dims(variance.astype('float32'), [1]), [1]),[1, d.L, self.K]) * tf.ones([self.T, d.L, self.K])
        self.prior_rho = Normal(mu = tf.slice(self.rho, [0, 0, 0], [self.T, d.L, self.K]) 
                                  - tf.slice(self.rho, [1, 0, 0], [self.T, d.L, self.K]),
                               sigma = sigma)

        if self.alpha_fixed:
            self.data = {self.prior_rho: tf.zeros((self.T, d.L, self.K))}

        else:
            self.prior_alpha = Normal(mu = self.alpha,
                                  sigma = (np.sqrt(1.0*d.N/self.n_minibatch.sum()) * self.lam).astype('float32') * tf.ones([d.L, self.K]))
            self.data = {self.prior_rho: tf.zeros((self.T, d.L, self.K)),
                     self.prior_alpha: tf.zeros((d.L, self.K))}

        self.placeholders = ['']*self.T
        self.y_pos = ['']*self.T
        self.y_neg = ['']*self.T

        for t in range(self.T):
            # Index Masks
            p_mask = tf.range(self.cs/2,self.n_minibatch[t] + self.cs/2)
            rows = tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch[t], 1])
            columns = tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch[t]), [1]), [1, self.cs/2])
            
            ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

            # Data Placeholder
            self.placeholders[t] = ed.placeholder(tf.int32, shape = (self.n_minibatch[t] + self.cs))

            # Taget and Context Indices
            self.p_idx = tf.gather(self.placeholders[t], p_mask)
            self.ctx_idx = tf.squeeze(tf.gather(self.placeholders[t], ctx_mask))
            
            # Negative samples
            unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(d.unigram)), [0]), [self.n_minibatch[t], 1])
            self.n_idx = tf.multinomial(unigram_logits, self.ns)

            # Context vectors
            self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)

            self.rho_t = tf.squeeze(tf.slice(self.rho, [t+1, 0, 0], [1, d.L, self.K]))
            
            self.p_rho_t = tf.squeeze(tf.gather(self.rho_t, self.p_idx))
            self.n_rho_t = tf.gather(self.rho_t, self.n_idx)

            # Natural parameter
            ctx_sum = tf.reduce_sum(self.ctx_alphas,[1])
            p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(self.p_rho_t, ctx_sum),-1),1)
            n_eta = tf.reduce_sum(tf.multiply(self.n_rho_t, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
            
            # Conditional likelihood
            self.y_pos[t] = Bernoulli(logits = p_eta)
            self.y_neg[t] = Bernoulli(logits = n_eta)

            self.data[self.y_pos[t]] = tf.ones((self.n_minibatch[t], 1))
            self.data[self.y_neg[t]] = tf.zeros((self.n_minibatch[t], self.ns))

    def eval_log_like(self, feed_dict, sess):
        log_p = np.zeros((0,1))
        for t in range(self.T):
            log_p_t = sess.run(tf.log(self.y_pos[t].mean()+0.000001), feed_dict = feed_dict)
            log_p = np.vstack((log_p, log_p_t))
        return log_p

    def dump(self, fname):
            dat = {'rho':  self.rho.eval(),
                   'alpha':  self.alpha.eval()}
            pickle.dump( dat, open( fname, "a+" ) )

    def plot_params(self, dir_name, d, plot_only=500):
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_alpha = tsne.fit_transform(self.alpha.eval()[:plot_only])
        plot_with_labels(low_dim_embs_alpha[:plot_only], d.labels[:plot_only], dir_name + 'alpha.eps')
        
        np_rho = self.rho.eval()

        for t in [0, int(self.T/2), self.T-1]:
            w_idx_t = np.argsort(d.unigram_t[t,:])[::-1][:plot_only]
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_rho = tsne.fit_transform(np_rho[t,w_idx_t,:])
            print(dir_name + 'freq_rho_' + str(t) + '.eps')
            plot_with_labels(low_dim_embs_rho, d.labels[w_idx_t], dir_name + 'freq_rho_' + str(t) + '.eps')

        for t in [0, int(self.T/2), self.T-1]:
            w_idx_t = range(plot_only)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_rho = tsne.fit_transform(np_rho[t,w_idx_t,:])
            plot_with_labels(low_dim_embs_rho, d.labels[w_idx_t], dir_name + 'rho_' + str(t) + '.eps')

    def detect_drift(self, d, dir_name, sess, metric='total_dist'):
        if metric == 'total_dist':
            rho_1 = tf.squeeze(tf.slice(self.rho, [1, 0, 0], [1, d.L, self.K]))
            rho_T = tf.squeeze(tf.slice(self.rho, [self.T, 0, 0], [1, d.L, self.K]))
            tf_dist, tf_w_idx = tf.nn.top_k(tf.reduce_sum(tf.square(rho_T-rho_1),1), 500)
        elif metric == 'variance':
            rho_ctr = self.rho - tf.tile(tf.expand_dims(tf.reduce_mean(self.rho, [0]) ,[0]), [self.T+1, 1, 1])
            rho_var = tf.reduce_mean(tf.reduce_sum(tf.square(rho_ctr), [-1]), [0])
            tf_dist, tf_w_idx = tf.nn.top_k(rho_var, 500)
        else:
            print('unknown metric')
            return
        dist, w_idx = sess.run([tf_dist, tf_w_idx])
        words = d.labels[w_idx]
        f_name = os.path.join(dir_name, metric+'_top_drifting_words.txt')
        with open(f_name, "w+") as text_file:
           for (w, drift) in zip(w_idx,dist):
               text_file.write("\n%-20s %6.4f" % (d.labels[w], drift))
        return words

    def print_word_similarities(self, words, d, num, dir_name, sess):
        t = ed.placeholder(dtype=tf.int32)
        query_word = ed.placeholder(dtype=tf.int32)
        query_rho_t = tf.expand_dims(tf.squeeze(tf.slice(self.rho, [t, query_word, 0], [1, 1, self.K])), [0])
        rho_t = tf.squeeze(tf.slice(self.rho, [t, 0, 0], [1, d.L, self.K]))
         
        val_rho, idx_rho = tf.nn.top_k(tf.matmul(tf.nn.l2_normalize(query_rho_t, dim=0), tf.nn.l2_normalize(rho_t, dim=1), transpose_b=True), num)

        for x in words:
            f_name = os.path.join(dir_name, '%s_queries_rho_cos.txt' % (x))
            with open(f_name, "w+") as text_file:
                for t_idx in xrange(self.T):
                    vr, ir = sess.run([val_rho, idx_rho], {query_word: d.dictionary[x], t: t_idx})
                    text_file.write("\n\n=====================================\n%s, t = %d\n=====================================" % (x,t_idx))
                    for ii in range(num):
                        text_file.write("\n%-20s %6.4f" % (d.labels[ir[0,ii]], vr[0,ii]))



    def print_topics(self, d, num, dir_name, sess):
        pass


