import pandas as pd
import numpy as np
import json, keras, os
import matplotlib.pyplot as plt
from utils.utils import high_dimension
from keras import regularizers
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Concatenate
from keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.utils import r_square
np.random.seed(0)
pd.options.mode.chained_assignment = None


def read_data():
    '''
    Loads appended CSV dataset and json dictionary of factorized variable categorical codes, creates X dataframe
    that excludes high-dimensional and other columns.
    Returns:
        X - dataframe of features
        y - series of outcome feature
        categorical_mappings - dictionary of  mappings for each categorical feature in dataframe
    '''
    # Load data
    df = pd.read_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data.csv')
    with open(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\cat_codes.json') as j:
        categorical_mappings = json.load(j)

    # Get config
    with open('config.json') as j:
        config = json.load(j)
    return df, categorical_mappings, config


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
            self.categorical_features = [i for i in list(self.categorical_mappings.keys()) if i not in [self.config['excluded_features']] and i not in high_dim]
            self.numeric_features = [col for col in self.data.columns if col not in self.categorical_features
                                     and col not in [self.config['excluded_features']] and self.config['outcome_feature'] not in col
                                     and col not in high_dim]
            self.model = self.construct_model()
            self.model = self.train_model()

        else:
            if config is None:
                raise ValueError("\nconfig file required, model not run.")
            else:
                self.config = config
            if data is None:
                raise ValueError("\nInput data required, model not run.")
            else:
                self.data = data
            if categorical_features is None: # assume all features numeric
                self.categorical_features = []
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

        # Add check if deep layer number equal dropout number

        clear_session()
        if self.categorical_features:
            categorical_input_layers = [Input(shape=(1,), dtype='int32') for _ in self.categorical_features]
            embedded_layers = [Embedding(input_dim=self.data[col].nunique(), output_dim=self.data[col].nunique(),
                                         embeddings_regularizer=regularizers.l2(self.config['l2_regularization']))(lyr) for (col, lyr) in zip(self.categorical_features, categorical_input_layers)]
            flatten_layers = [Flatten()(lyr) for lyr in embedded_layers]
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            concat_layer = Concatenate(axis=1)(flatten_layers + [numeric_input_layer])
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(concat_layer)
        else:
            categorical_input_layers = []
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(numeric_input_layer)
        for _ in range(len(self.config['nodes_per_dense_layer'])):
            dense_layer = Dense(list(self.config['nodes_per_dense_layer'].values())[_], activation='relu', kernel_regularizer=regularizers.l2(self.config['l2_regularization']))(dropout_layer)
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[_+1])(dense_layer)
        output_layer = Dense(1, activation='linear', name='output')(dropout_layer)
        model = Model(inputs=categorical_input_layers+[numeric_input_layer], outputs=output_layer)
        return model

    def process_data(self):
        '''
        :return: normalized train, test, valid sets as arrays
        '''
        X_train, X_test, y_train, y_test = self.train_test_split()
        # Normalize numeric features
        X_train[self.numeric_features] = self.normalize_numeric_features(X_train[self.numeric_features])
        X_test[self.numeric_features] = self.normalize_numeric_features(X_test[self.numeric_features])
        # Convert to arrays
        X_train = self.convert_arrays(X_train, self.categorical_features, self.numeric_features)
        X_test = self.convert_arrays(X_test, self.categorical_features, self.numeric_features)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        '''
        :return: keras model wrapper
        '''
        X_train, X_test, y_train, y_test = self.process_data()
        model = self.construct_model()
        model.compile(loss=self.config['loss'], optimizer=self.config['optimizer'], metrics=['mse', r_square])
        model.fit(X_train, y_train, epochs=self.config['max_epochs'], batch_size=self.config['batch_size'], verbose=1, validation_data=(X_test, y_test))
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

    def generate_prediction(self):
        '''
        :return: Pands dataframe of predicted values for training and validation sets with associated identifiers
        '''
        X_train, X_test, _, _ = self.train_test_split() # recover original identifier
        X_train_norm, X_test_norm, _, _ = self.process_data() # normalized data in array form for prediction
        train_prediction = pd.DataFrame(self.model.predict(X_train_norm), index=X_train[self.config['identifier_feature']]).reset_index().rename(columns={0: 'prediction'})
        val_prediction = pd.DataFrame(self.model.predict(X_test_norm), index=X_test[self.config['identifier_feature']]).reset_index().rename(columns={0: 'prediction'})
        return train_prediction, val_prediction


if __name__ == '__main__':

    # Allow categorical and numeric features
    model = RegressionPrediction()
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