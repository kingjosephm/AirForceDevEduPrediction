import tensorflow as tf
import json
import pandas as pd
from utils.utils import factorize_columns, r_square
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
from utils import tokenization # official BERT tokenization script from Google, available at https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
from utils.utils import bert_encode
from keras.backend import clear_session

def read_data():
    '''
    Loads appended CSV dataset and json dictionary of factorized variable categorical codes, creates X dataframe
    that excludes high-dimensional and other columns.
    Returns:
        f - dataframe of factorized categorical features (excluding string features) and numeric features
        categorical_mappings - dictionary of  mappings for each categorical feature in dataframe
    '''
    # Load unfactorized data
    df = pd.read_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data_unfactorized.csv', low_memory=False)
    df.drop(columns=['Final rank', 'Ballot rank'], inplace=True) # drop cols that are product of outcome variable of interest

    # Convert strings to factors and get mappings
    categorical_mappings, df_fac = factorize_columns(df)

    # Get config
    with open('config.json') as j:
        config = json.load(j)

    return df_fac, categorical_mappings, config

def get_bert_layer():
    '''
    :return: BERT layer from Tensorflow Hub, using locally-saved copy
    '''
#   Typical procedure to run (requires internet access):
#   module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
#   bert_layer = hub.KerasLayer(module_url, trainable=True)
    path = r'\\pii_zippy\d\USAF PME Board Evaluations\BERT'
    return hub.KerasLayer(path, trainable=True)

def build_tokenizer(bert_layer):
    '''
    Encodees text into tokens, masks, and segment flags
    :return:
    '''
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    return tokenization.FullTokenizer(vocab_file, do_lower_case)

def build_bert_model(bert_layer, max_len=512, model_summary=False):

    clear_session()
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='relu')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-5), loss='mse', metrics=['mse', r_square])

    if model_summary:
        model.summary()

    return model

def train_model(input_feature, outcome_feature, max_epochs=3, max_len=512, model_summary=False):

    bert_layer = get_bert_layer()
    tokenizer = build_tokenizer(bert_layer)
    model = build_bert_model(bert_layer, max_len=max_len, model_summary=model_summary)

    input_text = bert_encode(input_feature.values, tokenizer, max_len=max_len)
    return model.fit(input_text, outcome_feature.values, validation_split=0.2, epochs=max_epochs, batch_size=16)



if __name__ == '__main__':

    df, categorical_mappings, config = read_data()
    # One string or multiple string columns as predictors?

    history = train_model(input_feature=df[config['string_feature']],
                          outcome_feature=df[config['outcome_feature']],
                          max_len=config['max_string_len'],
                          max_epochs=config['max_epochs'])
