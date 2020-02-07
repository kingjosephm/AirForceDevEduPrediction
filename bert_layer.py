import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow_hub as hub
from utils import tokenization # official BERT tokenization script from Google, available at https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
from utils.utils import bert_encode

class BertLayer:

    def __init__(self):
        self.bert_layer = self.get_bert_model()
        self.tokenizer = self.build_tokenizer(self.bert_layer)

    def get_bert_model(self):
        '''
        :return: BERT layer from Tensorflow Hub, using locally-saved copy
        '''
        return hub.KerasLayer('../data/BERT_Model', trainable=True)

    def build_tokenizer(self, bert_layer):
        '''
        Encodees text into tokens, masks, and segment flags
        :return:
        '''
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        return tokenization.FullTokenizer(vocab_file, do_lower_case)

    def create_layer(self, max_len=160):

        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        return input_word_ids, input_mask, segment_ids, clf_output

    def encode_text(self, string_feature, max_len=160):
        return bert_encode(string_feature.values, self.tokenizer, max_len=max_len)