# pylint: disable=C0111, C0103
import tensorflow as tf
import numpy as np

def s2s_lstm_fixed(x, xf, hidden_dim, n_layers, reuse=None, use_bn=False, is_training=None, keep_prob=1.0, project=False):
    with tf.variable_scope('seq2seq', reuse=reuse):
        with tf.variable_scope('encode'):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            outputs1, states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
        es = [states]
        for i in range(1, n_layers):
            with tf.variable_scope('encode%d' % i, reuse=False):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                if use_bn:
                    outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
                outputs1, states = tf.nn.dynamic_rnn(lstm, tf.nn.dropout(outputs1, keep_prob), dtype=tf.float32)
            es.append(states)
            
        if project:
            old_es = es
            es = []
            for state in old_es:
                new_c = tf.layers.dense(state.c, hidden_dim, use_bias=False)
                new_h = tf.layers.dense(state.h, hidden_dim, use_bias=False)
                es.append(tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h))
            print('hi')
        with tf.variable_scope('decode', reuse=False):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            outputs, states = tf.nn.dynamic_rnn(lstm, xf, initial_state=es[0])
        for i in range(1, n_layers):
            with tf.variable_scope('decode%d' % i, reuse=False):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                if use_bn:
                    outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
                outputs, states = tf.nn.dynamic_rnn(lstm, tf.nn.dropout(outputs, keep_prob), initial_state=es[i])
        if use_bn:
            outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
    return tf.nn.dropout(outputs, keep_prob), tf.nn.dropout(outputs1[:,-1], keep_prob)

def s2s_lstm_multiresolution(x, xf, x_daily, hidden_dim=64, n_layers=2, daily_dim=64, reuse=None, keep_prob=1.0, use_bn=True, is_training=False, use_sigmoid=True):
    with tf.variable_scope('seq2seq', reuse=reuse):
        with tf.variable_scope('hourly_encode'):
            with tf.variable_scope('encode'):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                outputs1, states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
                if use_bn:
                        outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
            es = [states]
            for i in range(1, n_layers):
                with tf.variable_scope('encode%d' % i, reuse=False):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                    outputs1, states = tf.nn.dynamic_rnn(
                        lstm, tf.nn.dropout(outputs1, keep_prob), dtype=tf.float32)
                    if use_bn:
                        outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
                es.append(states)
        with tf.variable_scope('daily_encode'):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(daily_dim)
            outs, daily_states_tuple = tf.nn.dynamic_rnn(lstm, x_daily, dtype=tf.float32)
            
            out = tf.contrib.layers.batch_norm(outs[:,-1], is_training=is_training, updates_collections=None, scale=True, decay=0.95)

            dd_weight = tf.Variable(tf.random_normal([hidden_dim,1]) / np.sqrt(hidden_dim), name='daily_daily_weight')
            hd_weight = tf.Variable(tf.random_normal([hidden_dim,1]) / np.sqrt(hidden_dim), name='hourly_daily_weight')
            daily_bias = tf.Variable(tf.zeros(1) / np.sqrt(daily_dim), name='daily_bias')

            dh_weight = tf.Variable(tf.random_normal([hidden_dim,1]) / np.sqrt(hidden_dim), name='daily_hourly_weight')
            hh_weight = tf.Variable(tf.random_normal([hidden_dim,1]) / np.sqrt(hidden_dim), name='hourly_hourly_weight')
            hourly_bias = tf.Variable(tf.zeros(1) / np.sqrt(hidden_dim), name='hourly_bias')
            
            wd = out @ dd_weight + outputs1[:,-1,:] @ hd_weight + daily_bias
            wh = out @ dd_weight + outputs1[:,-1,:] @ hh_weight + hourly_bias

            if use_sigmoid:
                wd = tf.nn.sigmoid(wd)
                wh = tf.nn.sigmoid(wh)

            es[0] = tf.nn.rnn_cell.LSTMStateTuple(daily_states_tuple.c * wd + es[0].c * wh, daily_states_tuple.h * wd + es[0].h * wh)
        with tf.variable_scope('decode', reuse=False):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            outputs, states = tf.nn.dynamic_rnn(lstm, xf, initial_state=es[0])
            if use_bn:
                outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
        for i in range(1, n_layers):
            with tf.variable_scope('decode%d' % i, reuse=False):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                outputs, states = tf.nn.dynamic_rnn(
                    lstm, tf.nn.dropout(outputs, keep_prob), initial_state=es[i])
            if use_bn:
                outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)

    return tf.nn.dropout(outputs, keep_prob)


def other_s2s_lstm_multiresolution(x, xf, x_daily, hidden_dim=64, n_layers=2, daily_dim=64, reuse=None, keep_prob=1.0, use_bn=True, is_training=False, use_sigmoid=True):
    with tf.variable_scope('seq2seq', reuse=reuse):
        with tf.variable_scope('hourly_encode'):
            with tf.variable_scope('encode'):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                outputs1, states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
                if use_bn:
                        outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
            es = [states]
            for i in range(1, n_layers):
                with tf.variable_scope('encode%d' % i, reuse=False):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                    outputs1, states = tf.nn.dynamic_rnn(
                        lstm, tf.nn.dropout(outputs1, keep_prob), dtype=tf.float32)
                    if use_bn:
                        outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
                es.append(states)
        
        old_es = es
        es = []
        for state in old_es:
            new_c = tf.layers.dense(state.c, hidden_dim, use_bias=False)
            new_h = tf.layers.dense(state.h, hidden_dim, use_bias=False)
            es.append(tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h))
            
        with tf.variable_scope('daily_encode'):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(daily_dim)
            outs, daily_states_tuple = tf.nn.dynamic_rnn(lstm, x_daily, dtype=tf.float32)
            
            new_c = tf.layers.dense(daily_states_tuple.c, hidden_dim)
            new_h = tf.layers.dense(daily_states_tuple.h, hidden_dim)
            
            es[0] = tf.nn.rnn_cell.LSTMStateTuple(new_c + es[0].c, new_h + es[0].h)
        with tf.variable_scope('decode', reuse=False):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            outputs, states = tf.nn.dynamic_rnn(lstm, xf, initial_state=es[0])
            if use_bn:
                outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
        for i in range(1, n_layers):
            with tf.variable_scope('decode%d' % i, reuse=False):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                outputs, states = tf.nn.dynamic_rnn(
                    lstm, tf.nn.dropout(outputs, keep_prob), initial_state=es[i])
            if use_bn:
                outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)

    return tf.nn.dropout(outputs, keep_prob)

def s2s_lstm_multiresolution_project_daily( x, xf, x_daily, hidden_dim=64, n_layers=2, 
                                            daily_dim=32, reuse=None, keep_prob=1.0, use_bn=True, 
                                            is_training=False, recurrent_keep_prob=None):
    recurrent_keep_prob = recurrent_keep_prob or keep_prob
    with tf.variable_scope('seq2seq', reuse=reuse):
        with tf.variable_scope('hourly_encode'):
            with tf.variable_scope('encode'):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                outputs1, states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
                if use_bn:
                        outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
            es = [states]
            for i in range(1, n_layers):
                with tf.variable_scope('encode%d' % i, reuse=False):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                    outputs1 = tf.nn.dropout(outputs1, keep_prob)
                    outputs1, states = tf.nn.dynamic_rnn(
                        lstm, outputs1, dtype=tf.float32)
                    if use_bn:
                        outputs1 = tf.contrib.layers.batch_norm(outputs1, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
                es.append(states)
        with tf.variable_scope('daily_encode'):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(daily_dim)
            outs, daily_states_tuple = tf.nn.dynamic_rnn(lstm, x_daily, dtype=tf.float32)
            
            project_matrix = tf.Variable(tf.orthogonal_initializer()((daily_dim, hidden_dim)))
            es[0] = tf.nn.rnn_cell.LSTMStateTuple(daily_states_tuple.c @ project_matrix + es[0].c, daily_states_tuple.h @ project_matrix + es[0].h)
        with tf.variable_scope('decode', reuse=False):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            outputs, states = tf.nn.dynamic_rnn(lstm, xf, initial_state=es[0])
            if use_bn:
                outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)
        for i in range(1, n_layers):
            with tf.variable_scope('decode%d' % i, reuse=False):
                lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                outputs, states = tf.nn.dynamic_rnn(
                    lstm, tf.nn.dropout(outputs, keep_prob), initial_state=es[i])
            if use_bn:
                outputs = tf.contrib.layers.batch_norm(outputs, is_training=is_training, updates_collections=None, scale=True, decay=0.95)

    return tf.nn.dropout(outputs, keep_prob), tf.nn.dropout(outputs1[:,-1], keep_prob)