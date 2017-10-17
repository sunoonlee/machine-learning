import numpy as np
import tensorflow as tf  # 1.0.1

from data_helpers import (buckets, en_tokenizer, add_go_and_eos_tokens, bucket_and_pad,
                          _PAD, _GO, _EOS, _UNK, _PAD_ID, _GO_ID, _EOS_ID, _UNK_ID)


class NmtModel:
    """主要用于 restore 模型, 进行 decode.
    训练代码尚未放进来."""

    def __init__(self, cell_size, num_layers, embedding_size, num_sampled,
                 vocab_enc, vocab_dec, batch_size, learning_rate, do_decode):

        self.cell_size = cell_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.vocab_enc = vocab_enc
        self.vocab_dec = vocab_dec
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.do_decode = do_decode

        self.num_encoder_symbols = len(vocab_enc)
        self.num_decoder_symbols = len(vocab_dec)
        self.word2id_enc = {w: i for i, w in enumerate(vocab_enc)}
        self.word2id_dec = {w: i for i, w in enumerate(vocab_dec)}

        self.buckets = buckets

        def rnn_cell(num_layers):
            def single_cell():
                return tf.contrib.rnn.GRUCell(cell_size)
            cell = single_cell()
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
            return cell

        def seq2seq_f(encoder_placeholders, decoder_placeholders, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_placeholders,
                decoder_placeholders,
                rnn_cell(num_layers),
                num_encoder_symbols=self.num_encoder_symbols,
                num_decoder_symbols=self.num_decoder_symbols,
                embedding_size=embedding_size,
                output_projection=output_projection,
                feed_previous=do_decode)

        def sampled_loss(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                weights=softmax_w_t,
                biases=softmax_b,
                labels=labels,
                inputs=logits,
                num_sampled=num_sampled,
                num_classes=self.num_decoder_symbols)

        # create placeholders
        max_encoder_length, max_decoder_length = buckets[-1]
        self.encoder_placeholders = [
            tf.placeholder(tf.int32, shape=[None], name="encoder_%d" % i)
            for i in range(max_encoder_length)]
        self.decoder_placeholders = [
            tf.placeholder(tf.int32, shape=[None], name="decoder_%d" % i)
            for i in range(max_decoder_length)]
        self.target_placeholders = [
            tf.placeholder(tf.int32, shape=[None], name="target_%d" % i)
            for i in range(max_decoder_length)]
        self.target_weights_placeholders = [
            tf.placeholder(tf.float32, shape=[None], name="decoder_weight_%d" % i)
            for i in range(max_decoder_length)]

        # output_projection
        softmax_w_t = tf.get_variable("proj_w", [self.num_decoder_symbols, cell_size], dtype=tf.float32)
        softmax_w = tf.transpose(softmax_w_t)
        softmax_b = tf.get_variable("proj_b", [self.num_decoder_symbols], dtype=tf.float32)
        output_projection = (softmax_w, softmax_b)

        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_placeholders, self.decoder_placeholders, self.target_placeholders,
            self.target_weights_placeholders, buckets, lambda x, y: seq2seq_f(x, y, do_decode),
            softmax_loss_function=sampled_loss)

        if do_decode and output_projection:
            for bucket_id in range(len(buckets)):
                self.outputs[bucket_id] = [
                    tf.matmul(output, output_projection[0]) + output_projection[1]
                    for output in self.outputs[bucket_id]]


    def load_variables(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)


    def generate_feed(self, encoder_data, decoder_data, bucket_id, use_whole_bucket=False, batch_start=0):
        """对 inputs 做转置, 并喂给 placeholder 列表, 得到 feed_dict"""
        def left_shift(decoder_inputs):
            """generate targets grom decoder_inputs"""
            return [list(input_[1:]) + [_PAD_ID] for input_ in decoder_inputs]

        if use_whole_bucket:
            encoder_inputs = encoder_data[bucket_id]
            decoder_inputs = decoder_data[bucket_id]            
        else:
            encoder_inputs = encoder_data[bucket_id][batch_start : batch_start+self.batch_size]
            decoder_inputs = decoder_data[bucket_id][batch_start : batch_start+self.batch_size]

        encoder_inputs = list(zip(*encoder_inputs))
        target_inputs = list(zip(*left_shift(decoder_inputs)))
        decoder_inputs = list(zip(*decoder_inputs)) 

        # Prepare input data
        feed_dict = dict()
        encoder_size, decoder_size = buckets[bucket_id]
        for i in range(encoder_size):
            feed_dict[self.encoder_placeholders[i].name] = np.asarray(encoder_inputs[i], dtype=int)
        for i in range(decoder_size):
            feed_dict[self.decoder_placeholders[i].name] = np.asarray(decoder_inputs[i], dtype=int)
            feed_dict[self.target_placeholders[i].name] = np.asarray(target_inputs[i], dtype=int)        
            # 这里使用 weights 把 <PAD> 的损失屏蔽了
            feed_dict[self.target_weights_placeholders[i].name] = np.asarray(
                [float(idx != _PAD_ID) for idx in target_inputs[i]], dtype=float)
        return feed_dict


    def decode(self, sess, sentences):
        def cut_at_eos(sentence):
            if _EOS in sentence:        
                return sentence[:sentence.index(_EOS)]
            else:
                return sentence

        def no_prepending_pad(sentence):
            for i in range(len(sentence)):
                if sentence[i] != _PAD:
                    return sentence[i:]

        encoder_sentences = [en_tokenizer(s) for s in sentences]
        decoder_sentences = add_tokens(encoder_sentences)  # decode 时, decoder_sentences 不影响结果
        encoder_data, decoder_data = bucket_and_pad(
            encoder_sentences, decoder_sentences, buckets, self.word2id_enc, self.word2id_dec)
        data_sizes = [len(encoder_data[i]) for i in range(len(buckets))]

        for bucket_id in range(len(buckets)):
            cur_data_size = data_sizes[bucket_id]
            if cur_data_size == 0:
                continue  # 某个 bucket 为空的特殊情形

            encoder_size, decoder_size = buckets[bucket_id]
            bucket_feed = self.generate_feed(encoder_data, decoder_data, bucket_id, use_whole_bucket=True)
            output_bucket = np.zeros((cur_data_size, decoder_size), dtype=int)  
            # output_bucket 用于记录当前bucket输出值, 形状是 outputs 的"转置"
            
            for i in range(decoder_size):
                prob = self.outputs[bucket_id][i]  # 第i个词的概率输出
                output_bucket[:, i] = np.argmax(sess.run(prob, feed_dict=bucket_feed), axis=1)
            
            for j in range(cur_data_size):
                sen = [self.vocab_dec[output_bucket[j, k]] for k in range(decoder_size)]
                input_ = [self.vocab_enc[i] for i in encoder_data[bucket_id][j]]
                # target_ = [vocab_dec[i] for i in decoder_data_test[bucket_id][j][1:]]
                input_ = no_prepending_pad(input_)
                sen = cut_at_eos(sen)
                # target_ = cut_at_eos(target_)
                print(' input: ', ' '.join(input_))
                print('output: ', ' '.join(sen))
                # print('target: ', ' '.join(target_))
        return ''.join(sen)
