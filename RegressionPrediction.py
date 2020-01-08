import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from utils.utils import high_dimension
from keras import regularizers
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Concatenate
from  keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
np.random.seed(0)


def read_data():
    '''
    Loads appended CSV dataset and json dictionary of factorized variable categorical codes, creates X dataframe
    that excludes high-dimensional and other columns.
    Returns:
        X - dataframe of features
        y - series of outcome feature
        cat_features - list of categorical features in X
    '''
    # Load data
    df = pd.read_csv(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\combined_data.csv')
    with open(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\cat_codes.json') as j:
        dictionary = json.load(j)

    # Get config
    with open('config.json') as j:
        config = json.load(j)

    # Identify high dimensional categorical columns to exclude
    high_dim = high_dimension(dictionary, nr=500)

    # Create datasets
    # Note - excluding high-dimensional features for now
    df = df.loc[:, ~df.columns.isin(['SSN', 'Final rank', 'Ballot rank', 'Mbr Cmts', 'SR Cmts'] + high_dim)]

    # Identify categorical (factorized) vs numeric columns
    categorical_features = [i for i in df.columns if i in dictionary.keys()]
    return df, categorical_features, config


class RegressionPrediction:

    def __init__(self, config, data, categorical_features=None):
        self.config = config
        self.data = data.replace(-1, 0)
        if categorical_features is not None:
            self.categorical_features = [col for col in categorical_features]
            self.numeric_features = [col for col in self.data.columns if
                                     col not in self.categorical_features and col != self.config['outcome_feature']]
        else:
            self.categorical_features = None
            self.numeric_features = [col for col in self.data.columns if col != self.config['outcome_feature']]

    def train_test_valid_splits(self, X, y):
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_share'], random_state=self.config['seed'])
        # Train validation split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                        test_size=self.config['test_share'], random_state=self.config['seed'])
        return X_train, X_test, X_val, y_train, y_test, y_val


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


    def embedded_model(self):

        clear_session()
        if self.categorical_features != None:
            categorical_input_layers = [Input(shape=(1,), dtype='int32') for _ in self.categorical_features]
            embedded_layers = [Embedding(input_dim=self.data[col].nunique(), output_dim=self.data[col].nunique(),
                                         embeddings_regularizer=regularizers.l2(0.02))(lyr) for (col, lyr) in zip(self.categorical_features, categorical_input_layers)]
            flatten_layers = [Flatten()(lyr) for lyr in embedded_layers]
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            concat_layer = Concatenate(axis=1)(flatten_layers + [numeric_input_layer])
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(concat_layer)
        else:
            categorical_input_layers = []
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(numeric_input_layer)
        for _ in range(len(self.config['nodes_per_dense_layer'])):
            dense_layer = Dense(list(self.config['nodes_per_dense_layer'].values())[_], activation='relu')(dropout_layer)
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[_+1])(dense_layer)
        output_layer = Dense(1, activation='linear', name='output')(dropout_layer)
        model = Model(inputs=categorical_input_layers+[numeric_input_layer], outputs=output_layer)
        return model

    def train_data(self):
        X_train, X_test, X_val, y_train, y_test, y_val = self.train_test_valid_splits(self.data.loc[:, self.data.columns!=self.config['outcome_feature']], self.data[self.config['outcome_feature']])
        X_train[self.numeric_features] = self.normalize_numeric_features(X_train[self.numeric_features])
        X_val[self.numeric_features] = self.normalize_numeric_features(X_val[self.numeric_features])
        X_train = self.convert_arrays(X_train, self.categorical_features, self.numeric_features)
        X_val = self.convert_arrays(X_val, self.categorical_features, self.numeric_features)
        model = self.embedded_model()
        model.compile(loss='mse', optimizer=self.config['optimizer'], metrics=self.config['performance_metrics'])
        if categorical_features is not None:
            model.fit(X_train, y_train, epochs=self.config['max_epochs'], batch_size=self.config['batch_size'], verbose=1, validation_data=(X_val, y_val))
        else:
            model.fit(X_train[self.numeric_features], y_train, epochs=self.config['max_epochs'], batch_size=self.config['batch_size'],
                      verbose=1, validation_data=(X_val[self.numeric_features], y_val))
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


if __name__ == '__main__':
    # Get all the things
    df, categorical_features, config = read_data()

    # Allow categorical and numeric features
    history = RegressionPrediction(config, df, categorical_features).train_data()

    plt.close()
    plt.plot(history.history.history['loss'][10:])
    plt.plot(history.history.history['val_loss'][10:])
    plt.title('Mean Squared Error')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    plt.savefig(r'\\pii_zippy\d\USAF PME Board Evaluations\Processed data\results\figures\cat_numeric.png', dpi=250)

