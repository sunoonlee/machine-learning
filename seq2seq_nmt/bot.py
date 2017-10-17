import pickle
import requests
from wxpy import *
import tensorflow as tf
from nmt_model import NmtModel


def prepare_model(sess, vocab_path, model_path):
    with open(vocab_path, 'rb') as f:
        vocab_enc, vocab_dec = pickle.load(f)

    model = NmtModel(100, 2, 100, 500, vocab_enc, vocab_dec, 50, 0.1, True)
    model.load_variables(sess, model_path)
    return model


bot = Bot()
sess = tf.Session()
model = prepare_model(sess, '^data/vocab_13w.pkl', '^tmp/model_v04.ckpt')


@bot.register()
def reply(msg):
    print(msg)
    return model.decode(sess, [msg.text])

bot.join()