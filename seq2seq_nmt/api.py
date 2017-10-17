from flask import Flask, request
from flask_restful import Resource, Api
import tensorflow as tf
from nmt_model import NmtModel


def prepare_model(sess, vocab_path, model_path):
    with open(vocab_path, 'rb') as f:
        vocab_enc, vocab_dec = pickle.load(f)

    model = NmtModel(100, 2, 100, 500, vocab_enc, vocab_dec, 50, 0.1, True)
    model.load_variables(sess, model_path)
    return model


app = Flask(__name__)
api = Api(app)
sess = tf.Session()
model = prepare_model(sess, '^data/vocab_7w.pkl', '^model_v03/model_v03.ckpt')

class Seq2seqAPI(Resource):
    def get(self, source):
        return {source: model.decode(source)}

api.add_resource(Seq2seqAPI, '/<string:source>')


if __name__ == '__main__':
    app.run(debug=True)