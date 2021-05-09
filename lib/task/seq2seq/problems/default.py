from ..summary import *
from lib.layers.basic import *
from lib.train.problem import Problem


def word_dropout(inp, inp_len, dropout, method, voc):
    inp_shape = tf.shape(inp)

    border = tf.fill([inp_shape[0], 1], False)

    mask = tf.sequence_mask(inp_len - 2, inp_shape[1] - 2)
    mask = tf.concat((border, mask, border), axis=1)
    mask = tf.logical_and(mask, tf.random_uniform(inp_shape) < dropout)

    if method == 'unk':
        replacement = tf.fill(inp_shape, tf.cast(voc._unk, inp.dtype))
    elif method == 'random_word':
        replacement = tf.random_uniform(inp_shape, minval=max(voc.bos, voc.eos, voc._unk)+1, maxval=voc.size(), dtype=inp.dtype)
    else:
        raise ValueError("Unknown word dropout method: %r" % method)

    return tf.where(mask, replacement, inp)


class DefaultProblem(Problem):

    def __init__(self, models, sum_loss=False, use_small_batch_multiplier=False,
        inp_word_dropout=0, out_word_dropout=0, word_dropout_method='unk',raml=False,switchout=False,raml_temp=1.0, switchout_temp=1.0
    ):
        assert len(models) == 1

        self.models = models
        self.model = list(self.models.values())[0]

        self.inp_voc = self.model.inp_voc
        self.out_voc = self.model.out_voc

        self.sum_loss = sum_loss
        self.use_small_batch_multiplier = use_small_batch_multiplier

        self.inp_word_dropout = inp_word_dropout
        self.out_word_dropout = out_word_dropout
        self.word_dropout_method = word_dropout_method

        self.raml=raml
        self.raml_temp=raml_temp
        self.switchout=switchout
        self.switchout_temp=switchout_temp

        if self.use_small_batch_multiplier:
            self.max_batch_size_var = tf.get_variable("max_batch_size", shape=[], initializer=tf.ones_initializer(), trainable=False)

    def hamming_distance_sample(self,sents, tau, bos_id, eos_id, pad_id, vocab_size):
        # mask
        mask = [
        tf.equal(sents, bos_id),
        tf.equal(sents, eos_id),
        tf.equal(sents, pad_id),
        ]
        mask = tf.stack(mask, axis=0)
        mask = tf.reduce_any(mask, axis=0)

        # first, sample the number of words to corrupt for each sentence
        batch_size, n_steps = tf.unstack(tf.shape(sents))
        logits = -tf.range(tf.to_float(n_steps), dtype=tf.float32) * tau
        logits = tf.expand_dims(logits, axis=0)
        logits = tf.tile(logits, [batch_size, 1])
        logits = tf.where(mask,
        x=tf.fill([batch_size, n_steps], -float("inf")), y=logits)

        # sample the number of words to corrupt at each sentence
        num_words = tf.multinomial(logits, num_samples=1)
        num_words = tf.reshape(num_words, [batch_size])
        num_words = tf.to_float(num_words)

        # <bos> and <eos> should never be replaced!
        lengths = tf.reduce_sum(1.0 - tf.to_float(mask), axis=1)

        # sample corrupted positions
        probs = num_words / lengths
        probs = tf.expand_dims(probs, axis=1)
        probs = tf.tile(probs, [1, n_steps])
        probs = tf.where(mask, x=tf.zeros_like(probs), y=probs)
        bernoulli = tf.distributions.Bernoulli(probs=probs, dtype=tf.int32)

        pos = bernoulli.sample()
        pos = tf.cast(pos, tf.bool)

        # sample the corrupted values
        val = tf.random_uniform([batch_size, n_steps], minval=1, maxval=vocab_size, dtype=tf.int32)
        val = tf.where(pos, x=val, y=tf.zeros_like(val))
        sents = tf.mod(sents + val, vocab_size)
        return sents

    def _make_encdec_batch(self, batch, is_train):
        encdec_batch = copy(batch)

        if is_train and self.inp_word_dropout > 0:
            encdec_batch['inp'] = word_dropout(encdec_batch['inp'], encdec_batch['inp_len'], self.inp_word_dropout, self.word_dropout_method, self.model.inp_voc)

        if is_train and self.out_word_dropout > 0:
            encdec_batch['out'] = word_dropout(encdec_batch['out'], encdec_batch['out_len'], self.out_word_dropout, self.word_dropout_method, self.model.out_voc)

        return encdec_batch

    def _make_da_batch(self,batch,is_train):
        da_batch = copy(batch)

        if is_train and self.raml:
            da_batch['out']=self.hamming_distance_sample(da_batch['out'], self.raml_temp,self.out_voc.bos,self.out_voc.eos, self.out_voc.eos, self.out_voc.size())

        if is_train and self.switchout:
            da_batch['inp']=self.hamming_distance_sample(da_batch['inp'], self.switchout_temp,self.inp_voc.bos,self.inp_voc.eos, self.inp_voc.eos, self.inp_voc.size())

        return da_batch

    def batch_counters(self, pre_batch, is_train):
        if hasattr(self.model, 'batch_counters'):
            return self.model.batch_counters(pre_batch, is_train)

        batch=self._make_da_batch(pre_batch,is_train)

        rdo = self.model.encode_decode(self._make_encdec_batch(batch, is_train), is_train)

        with dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(rdo, batch['out'], batch['out_len'])  # [batch_size * nout * ovoc_size]
            loss_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])

        counters = dict(
            loss=tf.reduce_sum(loss_values),
            out_len=tf.to_float(tf.reduce_sum(batch['out_len'])),
        )
        append_counters_common_metrics(counters, logits, batch['out'], batch['out_len'], is_train)
        append_counters_xent(counters, loss_values, batch['out_len'])
        append_counters_io(counters, batch['inp'], batch['out'], batch['inp_len'], batch['out_len'])
        return counters

    def get_xent(self, pre_batch, is_train):
        if hasattr(self.model, 'batch_counters'):
            return self.model.batch_counters(pre_batch, is_train)

        batch=self._make_da_batch(pre_batch,is_train)

        rdo = self.model.encode_decode(self._make_encdec_batch(batch, is_train), is_train)

        with dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(rdo, batch['out'],
                                                   batch['out_len'])  # [batch_size * nout * ovoc_size]
            loss_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])

        return loss_values

    def loss_multibatch(self, counters, is_train):
        if self.sum_loss:
            value = tf.reduce_sum(counters['loss'])
        else:
            value = tf.reduce_sum(counters['loss']) / tf.reduce_sum(counters['out_len'])

        if self.use_small_batch_multiplier and is_train:
            batch_size = tf.reduce_sum(counters['out_len'])
            max_batch_size = tf.maximum(self.max_batch_size_var, batch_size)
            with tf.control_dependencies([tf.assign(self.max_batch_size_var, max_batch_size)]):
                small_batch_multiplier = batch_size / max_batch_size
                value = value * small_batch_multiplier

        return value

    def summary_multibatch(self, counters, prefix, is_train):
        res = []
        res += summarize_common_metrics(counters, prefix)
        res += summarize_xent(counters, prefix)
        res += summarize_io(counters, prefix)
        return res

    def params_summary(self):
        if hasattr(self.model, 'params_summary'):
            return self.model.params_summary()

        return []

    def make_feed_dict(self, batch, **kwargs):
        return self.model.make_feed_dict(batch, **kwargs)
