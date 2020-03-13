import pandas as pd
import numpy as np
import shap, itertools, math
import statsmodels.api as sm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.utils import r_square, read_data
from bert_layer import BertLayer
pd.options.mode.chained_assignment = None


class RegressionPrediction:

    def __init__(self, embed_categoricals=True, train_network=True):

        self.data, self.categorical_mappings, self.config = read_data()

        # Build feature lists
        self.string_features = [i for i in self.config['string_features']]

        if embed_categoricals:
            self.categorical_features = [i for i in list(self.categorical_mappings.keys())
                                         if i not in self.config['identifier_features']+self.config['outcome_feature']+self.config['string_features']]
        else:
            self.categorical_features = [] # treat all as numeric

        self.numeric_features = [i for i in self.data.columns if i not in self.categorical_features
                                 and i not in self.config['identifier_features']+self.config['outcome_feature']+self.config['string_features']]

        if train_network:
            self.bert_model, \
            self.main_model, \
            self.bert_pred_train, \
            self.bert_pred_val, \
            self.X_train, \
            self.X_val = self.train_network()


    def train_test_split(self):
        '''
        :return: Training, validation, and test sets based on desired test share
        '''
        # Train test split
        X_train, X_val, y_train, y_val = train_test_split(self.data[[i for i in self.data.columns if i not in self.config['outcome_feature']+self.config['identifier_features']]],
                                                            self.data[self.config['outcome_feature']],
                                                            test_size=self.config['test_share'], random_state=self.config['seed'])

        # Convert outcome to pd.Series, since fed as list in config
        y_train = y_train.iloc[:, 0]
        y_val = y_val.iloc[:, 0]
        return X_train, X_val, y_train, y_val

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

    def construct_network(self):
        '''
        :return: keras.engine.training.Model object
        '''

        # Verify number of layers for nodes equals number of layers for dropout in main network
        if len(self.config['nodes_per_dense_layer']) + 1 != len(self.config['dropout_share_per_layer']):
            raise ValueError(
                "\nNumber of layers for dropout share and nodes per dense layer different! These must be the same.")

        clear_session()

        # Build bert neural network
        if self.string_features:
            input_word_ids, input_mask, segment_ids, clf_output = BertLayer().create_layer(
                max_len=self.config['max_str_len'])
            dense_layer = Dense(64, activation='relu')(clf_output)
            dense_layer = Dense(128, activation='relu')(dense_layer)
            dense_layer = Dense(256, activation='relu')(dense_layer)
            dense_layer = Dense(64, activation='relu')(dense_layer)
            bert_output_layer = Dense(1, activation='linear', name='bert_output_layer')(dense_layer)
            bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=bert_output_layer)
        else:
            bert_model = None

        # Build main neural network
        if self.categorical_features:
            categorical_input_layers = [Input(shape=(1,), dtype='int32') for _ in self.categorical_features]
            embedded_layers = [Embedding(input_dim=self.data[col].nunique()+1, output_dim=round(math.sqrt(self.data[col].nunique())),
                                         embeddings_regularizer=regularizers.l1_l2(self.config['l1_regularization'],
                                         self.config['l2_regularization']))(lyr) for (col, lyr) in zip(self.categorical_features, categorical_input_layers)]
            flatten_layers = [Flatten()(lyr) for lyr in embedded_layers]
            if self.string_features:
                numeric_input_layer = Input(shape=(len(self.numeric_features) + 1,)) # add extra tensor for BERT model prediction
            else:
                numeric_input_layer = Input(shape=(len(self.numeric_features),))
            concat_layer = Concatenate(axis=1)(flatten_layers + [numeric_input_layer])
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(concat_layer)
        else:  # assume no categorical features, treat all as numeric
            categorical_input_layers = []
            if self.string_features:
                numeric_input_layer = Input(shape=(len(self.numeric_features) + 1,))
            else:
                numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[0])(numeric_input_layer)
        for _ in range(len(self.config['nodes_per_dense_layer'])):
            dense_layer = Dense(list(self.config['nodes_per_dense_layer'].values())[_], activation='relu',
                                kernel_regularizer=regularizers.l1_l2(self.config['l1_regularization'],
                                                                      self.config['l2_regularization']))(dropout_layer)
            dropout_layer = Dropout(list(self.config['dropout_share_per_layer'].values())[_ + 1])(dense_layer)
        output_layer = Dense(1, activation='linear', name='output')(dropout_layer)
        main_model = Model(inputs=categorical_input_layers + [numeric_input_layer], outputs=output_layer)
        return bert_model, main_model

    def process_data(self):
        '''
        :return: normalized train, test, valid sets as arrays
        '''
        X_train, X_val, y_train, y_val = self.train_test_split()
        # Normalize numeric features
        X_train[self.numeric_features] = self.normalize_numeric_features(X_train[self.numeric_features])
        X_val[self.numeric_features] = self.normalize_numeric_features(X_val[self.numeric_features])

        # Encode text fields
        if self.string_features:
            train_text = [BertLayer().encode_text(X_train[i], max_len=self.config['max_str_len']) for i in
                          self.string_features]
            train_text = list(itertools.chain(*train_text))  # flattens list
            val_text = [BertLayer().encode_text(X_val[i], max_len=self.config['max_str_len']) for i in self.string_features]
            val_text = list(itertools.chain(*val_text))

            # Convert to arrays
            X_train = self.convert_arrays(X_train[[i for i in X_train.columns if i not in self.string_features]],
                                     self.categorical_features, self.numeric_features)
            X_val = self.convert_arrays(X_val[[i for i in X_val.columns if i not in self.string_features]],
                                    self.categorical_features, self.numeric_features)

        else:
            # Convert to arrays
            train_text = []
            val_text = []
            X_train = self.convert_arrays(X_train, self.categorical_features, self.numeric_features)
            X_val = self.convert_arrays(X_val, self.categorical_features, self.numeric_features)

        return train_text, val_text, X_train, X_val, y_train, y_val

    def train_network(self):
        '''
        :return: keras model wrapper for BERT and main neural networks, as well as arrayed-form of X_train and X_val
        '''
        train_text, val_text, X_train, X_val, y_train, y_val = self.process_data()

        bert_model, main_model = self.construct_network()

        # Train bert network
        if self.string_features:
            bert_model.compile(loss=MSE, optimizer=Adam(amsgrad=True), metrics=[MSE, r_square])
            bert_model.fit(train_text, y_train.values, epochs=self.config['max_bert_epochs'], batch_size=32,
                           verbose=1, validation_data=(val_text, y_val.values))

            # Concatenate predictions from BERT layer to main neural network numeric input array
            X_train[-1]['bert_pred'] = bert_model.predict(train_text)
            X_val[-1]['bert_pred'] = bert_model.predict(val_text)

            # Create df of BERT predictions and identifier
            bert_pred_train = pd.concat([self.data[self.config['identifier_features']], X_train[-1]['bert_pred']], axis=1, join='inner')
            bert_pred_val = pd.concat([self.data[self.config['identifier_features']], X_val[-1]['bert_pred']], axis=1, join='inner')
        else:
            bert_pred_train = None
            bert_pred_val = None

        # Train main network, with bert output as input
        main_model.compile(loss=MSE, optimizer=Adam(amsgrad=True), metrics=[MSE, r_square])

        if self.config['early_stoppage']:
            main_model.fit(X_train, y_train, epochs=self.config['max_main_epochs'], batch_size=self.config['batch_size'], verbose=2,
                      validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=2,
                      patience=self.config['patience'], restore_best_weights=True)])
        else:
            main_model.fit(X_train, y_train, epochs=self.config['max_main_epochs'], batch_size=self.config['batch_size'], verbose=2,
                      validation_data=(X_val, y_val))

        return bert_model, main_model, bert_pred_train, bert_pred_val, X_train, X_val

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
            return arr
        return np.expand_dims(arr, axis=1)

    def generate_prediction(self):
        '''
        Generates predicted values for train and validation sets. Note - this will only work if model is train_model=True in init statement.
        :return: Pands dataframe of predicted values for training and validation sets with associated identifiers
        '''
        X_train, X_val, _, _ = self.train_test_split()  # recover index for indentifying unique observations

        train_prediction = pd.DataFrame(self.main_model.predict(self.X_train), index=X_train.index).rename(columns={0: 'prediction'})
        val_prediction = pd.DataFrame(self.main_model.predict(self.X_val), index=X_val.index).rename(columns={0: 'prediction'})

        # Merge with full train and val dataframes
        train_set = pd.concat([self.data, train_prediction], axis=1, join='inner')
        val_set = pd.concat([self.data, val_prediction], axis=1, join='inner')
        train_set['error'] = train_set[''.join(self.config['outcome_feature'])] - train_set['prediction']
        val_set['error'] = val_set[''.join(self.config['outcome_feature'])] - val_set['prediction']

        def rearrange_cols(df):
            '''
            Rearranges columns to place identifier, outcome, and prediction as first three, followed by rest
            :param df: Pandas dataframe
            :return: rearranged df
            '''
            cols = self.config['identifier_features']+self.config['outcome_feature']+['prediction', 'error']
            cols = cols + [i for i in df.columns if i not in cols]
            return df[cols]

        train_set = rearrange_cols(train_set)
        val_set = rearrange_cols(val_set)

        # Convert categorical features back to original values
        for _ in train_set, val_set:
            # Convert categorical features back to original values
            for key, values in self.categorical_mappings.items():
                _[key] = _[key].replace(to_replace=values, value=None)
                _[key] = _[key].replace([0, 0.0, '0', '00', '000', '0000'], np.NaN)  # missing coded as zero
            # Convert dates back to original pd.datetime format (YYYY-MM-DD)
            dates = [i for i in _.columns if "DATE" in i or i in ['BDAY', 'DEROS', 'GRADE CURR DOR', 'TAFCSD', 'TAFMSD',
                    'TFCSD', 'ETO', 'ODSD']]
            for col in dates:
                _[col] = _[col].replace([0, 0.0, '0', '00', '000', '0000'], np.NaN)
                _[col] = pd.to_datetime('1900-01-01', format='%Y-%m-%d') + pd.to_timedelta(_[col], unit='D')
        return train_set, val_set

    def shap_values(self, n=128):
        subset = self.data.sample(n=n)
        shap_subset = self.convert_arrays(subset, self.categorical_features, self.numeric_features)
        shap_subset = [self.expand_to_2d(i.values) for i in shap_subset]
        shap_values = shap.DeepExplainer(Model(self.main_model.inputs, self.main_model.output), shap_subset).shap_values(shap_subset)
        if self.categorical_features:
            shap_values = [np.hstack(arr_list) for arr_list in shap_values]
        return shap_values

    def ols_regression(self):
        '''
        OLS regression of whole dataset treating all features as numeric
        :return: OLS results printout
        '''
        X_train, X_val, y_train, y_val = self.train_test_split()

        if self.string_features: # remove any string features
            X_train.drop(columns=self.string_features, inplace=True)
            X_val.drop(columns=self.string_features, inplace=True)
        X_train = sm.add_constant(X_train)
        X_val = sm.add_constant(X_val)
        ols_obj = sm.OLS(y_train, X_train).fit()

        # Calculate MSE and R2 out-of-sample performance
        coef = np.array(ols_obj.params)
        yhat = np.dot(X_val, coef)
        mse = np.sum((y_val - yhat) ** 2) / (len(yhat) - len(coef) + 1)
        r2 = 1 - (np.sum((y_val - yhat) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
        return (ols_obj, mse, r2)