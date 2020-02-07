import pandas as pd
import numpy as np
import json, keras, os, shap, itertools
import matplotlib.pyplot as plt
from utils.utils import high_dimension
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.backend import clear_session, zeros
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.utils import r_square, factorize_columns
from bert_layer import BertLayer
np.random.seed(0)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 550)


def read_data():
    '''
    Loads appended CSV dataset and json dictionary of factorized variable categorical codes, creates X dataframe
    that excludes high-dimensional and other columns.
    Returns:
        df_fac - dataframe of factorized categorical features (excluding string features) and numeric features
        categorical_mappings - dictionary of  mappings for each categorical feature in dataframe
        config file
    '''
    # Load unfactorized data
    df = pd.read_csv('../data/combined_data_unfactorized.csv', low_memory=False)

    # Get config
    with open('config.json') as j:
        config = json.load(j)

    # Convert strings to factors and get mappings
    if config['string_features']:
        categorical_mappings, df_fac = factorize_columns(df[[i for i in df.columns if i not in config['string_features']]])
    else:
        categorical_mappings, df_fac = factorize_columns(df)

    return df_fac, categorical_mappings, config


def save_model(model, path, file_name):

    if isinstance(model, keras.engine.training.Model):
        model.save(os.path.join(path, file_name + '.h5'))
    else:
        print('Model type unrecognized recognized and not saved.')

class RegressionPrediction:

    def __init__(self, config=None, data=None, categorical_features=None, autorun=True):

        if autorun:
            self.data, self.categorical_mappings, self.config = read_data()

            high_dim = high_dimension(self.categorical_mappings, nr=self.config['max_feature_categories']) # exclude categorical features with > 'max_feature_categories' unique categories

            self.string_features = [i for i in self.config['string_features']]
            self.categorical_features = [i for i in list(self.categorical_mappings.keys())
                                         if i not in high_dim+self.config['excluded_features']+[self.config['identifier_feature']]+[self.config['outcome_feature']]+self.config['string_features']]

            self.numeric_features = [i for i in self.data.columns if i not in self.categorical_features
                                     and i not in self.config['excluded_features']+[self.config['identifier_feature']]+[self.config['outcome_feature']]+self.config['string_features']]

            self.model = self.construct_model()
            self.model = self.train_model() # trains 3 epochs for BERT layer(s)
            if self.string_features:
                self.model = self.freeze_bert_layers(self.model)
            self.model = self.train_model(max_epochs=self.config['max_epochs'])

            #self.shap_values = self.shap_values(n=self.config['num_shap_observations'])

        else:
            if config is None:
                self.config = {
                    "seed": 12345,
                    "nodes_per_dense_layer": {"one": 256, "two": 512},
                    "dropout_share_per_layer": {"zero": 0.1, "one": 0.2, "two": 0.5},
                    "l2_regularization": 0.05,
                    "max_epochs": 30,
                    "batch_size": 128,
                    "test_share": 0.25,
                    "optimizer": "adam",
                    "patience": 5,
                    "loss": "mse",
                    "outcome_feature": None,
                    "excluded_features": [],
                    "identifier_feature": None,
                    "string_features": [],
                    "max_str_len": 160,
                    "max_feature_categories": 600,
                    "num_shap_observations": 2}

            if data is None:
                raise ValueError("\nInput data required, model not run.")
            else:
                self.data = data
            if categorical_features is None: # assume all features numeric
                self.categorical_features = []
                self.string_features = []
                self.numeric_features = [col for col in self.data.columns if col != self.config['outcome_feature']]

    def train_test_split(self):
        '''
        :return: Training, validation, and test sets based on desired test share
        '''
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(self.data.loc[:, self.data.columns!=self.config['outcome_feature']],
                                                            self.data[self.config['outcome_feature']],
                                                            test_size=self.config['test_share'], random_state=self.config['seed'])
        return X_train, X_test, y_train, y_test


    def normalize_numeric_features(self, df):
        '''
        Min-max normalizes all features in Pandas dataframe (apply this numeric features only).
        :df: Pandas dataframe
        Returns: normalized dataframe
        '''
        num = df[self.numeric_features].copy()
        scaler = MinMaxScaler()
        scaled_vals = scaler.fit_transform(num)
        df.loc[:, :] = scaled_vals
        return df

    def construct_model(self):
        '''
        :return: keras.engine.training.Model object
        '''

        # Verify number of layers for nodes equals number of layers for dropout
        if len(self.config['nodes_per_dense_layer']) + 1 != len(self.config['dropout_share_per_layer']):
            raise ValueError(
                "\nNumber of layers for dropout share and nodes per dense layer different! These must be the same.")

        clear_session()
        if self.categorical_features:
            categorical_input_layers = [Input(shape=(1,), dtype='int32') for _ in self.categorical_features]
            embedded_layers = [Embedding(input_dim=self.data[col].nunique(), output_dim=self.data[col].nunique(),
                                         embeddings_regularizer=regularizers.l2(self.config['l2_regularization']))(lyr)
                               for (col, lyr) in zip(self.categorical_features, categorical_input_layers)]
            flatten_layers = [Flatten()(lyr) for lyr in embedded_layers]
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            if self.string_features:
                input_word_ids, input_mask, segment_ids, clf_output = BertLayer().create_layer(
                    max_len=self.config['max_str_len'])
                bert_output = Dense(1, activation='relu')(clf_output)
                concat_layer = Concatenate(axis=1)([bert_output] + flatten_layers + [numeric_input_layer])
            else:
                input_word_ids = []
                input_mask = []
                segment_ids = []
                concat_layer = Concatenate(axis=1)(flatten_layers + [numeric_input_layer])
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(concat_layer)
        else:  # assume no categorical features, treat all as numeric
            input_word_ids = []
            input_mask = []
            segment_ids = []
            categorical_input_layers = []
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(numeric_input_layer)
        for _ in range(len(self.config['nodes_per_dense_layer'])):
            dense_layer = Dense(list(self.config['nodes_per_dense_layer'].values())[_], activation='relu',
                                kernel_regularizer=regularizers.l2(self.config['l2_regularization']))(dropout_layer)
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[_ + 1])(dense_layer)
        output_layer = Dense(1, activation='linear', name='output')(dropout_layer)
        model = Model(inputs=[input_word_ids, input_mask, segment_ids] + categorical_input_layers + [numeric_input_layer],
            outputs=output_layer)
        return model

    def freeze_bert_layers(self, model):
        new_model = Model(model.inputs, model.outputs)
        for layer in new_model.layers:
            if type(layer).__name__ == "KerasLayer":
                layer.trainable = False
        return new_model

    def process_data(self):
        '''
        :return: normalized train, test, valid sets as arrays
        '''
        X_train, X_test, y_train, y_test = self.train_test_split()

        # Normalize numeric features
        X_train[self.numeric_features] = self.normalize_numeric_features(X_train[self.numeric_features])
        X_test[self.numeric_features] = self.normalize_numeric_features(X_test[self.numeric_features])

        # Encode text fields
        if self.string_features:
            train_text = [BertLayer().encode_text(X_train[i], max_len=self.config['max_str_len']) for i in
                          self.string_features]
            train_text = list(itertools.chain(*train_text))  # flattens list
            test_text = [BertLayer().encode_text(X_test[i], max_len=self.config['max_str_len']) for i in self.string_features]
            test_text = list(itertools.chain(*test_text))

            # Convert to arrays
            X_train = self.convert_arrays(X_train[[i for i in X_train.columns if i not in self.string_features]],
                                     self.categorical_features, self.numeric_features)
            X_test = self.convert_arrays(X_test[[i for i in X_test.columns if i not in self.string_features]],
                                    self.categorical_features, self.numeric_features)
            X_train = train_text + X_train
            X_test = test_text + X_test

        else:
            # Convert to arrays
            X_train = self.convert_arrays(X_train, self.categorical_features, self.numeric_features)
            X_test = self.convert_arrays(X_test, self.categorical_features, self.numeric_features)

        return X_train, X_test, y_train, y_test

    def train_model(self, max_epochs=3):
        '''
        :return: keras model wrapper
        '''
        X_train, X_test, y_train, y_test = self.process_data()
        model = self.construct_model()
        model.compile(loss=self.config['loss'], optimizer=self.config['optimizer'], metrics=['mse', r_square])
        model.fit(X_train, y_train, epochs=max_epochs, batch_size=self.config['batch_size'], verbose=1,
                  validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                  patience=self.config['patience'], restore_best_weights=True)])
        return model

    def convert_arrays(self, df, categorical_features, numeric_features):
        '''
        Function creates separate np.array for each categorical feature and appends with one np.array containing all numeric features
        :param df:
        :param categorical_features:
        :param numeric_features:
        :return: list of np.arrays of length number of categorical_features + 1
        '''
        return [df[col] for col in categorical_features] + [df[numeric_features]]

    def expand_to_2d(self, arr):
        if len(arr.shape) >= 2:
            return arr.values
        return np.expand_dims(arr, axis=1)

    def generate_prediction(self):
        '''
        :return: Pands dataframe of predicted values for training and validation sets with associated identifiers
        '''
        X_train, X_test, _, _ = self.train_test_split() # recover original identifier
        X_train_norm, X_test_norm, _, _ = self.process_data() # normalized data in array form for prediction
        train_prediction = pd.DataFrame(self.model.predict(X_train_norm), index=X_train[self.config['identifier_feature']]).reset_index().rename(columns={0: 'prediction'})
        val_prediction = pd.DataFrame(self.model.predict(X_test_norm), index=X_test[self.config['identifier_feature']]).reset_index().rename(columns={0: 'prediction'})
        return train_prediction, val_prediction

    def shap_values(self, n=128):
        subset = self.data.sample(n=n)
        shap_subset = self.convert_arrays(subset, self.categorical_features, self.numeric_features)
        #shap_subset = [self.expand_to_2d(i) for i in shap_subset]
        shap_values = shap.DeepExplainer(Model(self.model.inputs, self.model.output), shap_subset).shap_values(shap_subset)
        if self.categorical_features:
            shap_values = [np.hstack(arr_list) for arr_list in shap_values]
        return shap_values

if __name__ == '__main__':

    # Allow categorical and numeric features
    model = RegressionPrediction()
    model.train_model()
    #save_model(model.model, r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\results', 'initial_model')

    plt.close()
    plt.plot(model.model.history.history['mse'][5:])
    plt.plot(model.model.history.history['val_mse'][5:])
    plt.title('Mean Squared Error')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    plt.savefig(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\results\figures\mse_performance.png', dpi=350)

    plt.close()
    plt.plot(model.model.history.history['val_r_square'][5:])
    plt.title('Coefficient of Determination (R2)')
    plt.ylabel('Share of variance explained')
    plt.xlabel('Epoch')
    plt.legend(['validation'])
    plt.show()
    plt.savefig(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\results\figures\r2_performance.png', dpi=350)