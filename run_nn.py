import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from RegressionPrediction import RegressionPrediction
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 550)
from datetime import date
import os, pickle

def plot_diagnostic(diagnostic='r_square', constrain_N=0, plt_path=None, bert=False):

    val_temp = model_obj.main_model.history.history['val_' + diagnostic].copy()
    train_temp = model_obj.train_hist.history['val_' + diagnostic].copy()[:len(val_temp)] # if early_stoppage=False might exceed length of validation

    if bert:
        train_temp = model_obj.bert_model.history.history['val_' + diagnostic].copy()
        val_temp = model_obj.bert_model.history.history['val_' + diagnostic].copy()

    for i in range(min(len(train_temp), constrain_N)):
        train_temp[i] = np.NaN
        val_temp[i] = np.NaN

    plt.clf()
    plt.close()
    plt.plot(train_temp)
    plt.plot(val_temp)
    if diagnostic == 'r_square':
        plt.title('Coefficient of Determination (R2)')
        plt.ylabel('Share of variance explained')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
    elif diagnostic == 'mean_squared_error':
        plt.title('Mean Squared Error')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
    else:
        raise ValueError("\nModel diagnostic not understood, must be 'r_square' or 'mean_squared_error'")

    if plt_path is not None:
        path, filename = os.path.split(plt_path)
        os.makedirs(path, exist_ok=True)
        plt.savefig(plt_path, dpi=512)

def save_results():
    '''
    Saves class attributes to pickle and h5 files. Note - with custom BERT layer this produced an error saving the
    Keras model wrappers themselves, so instead serialize model metadata and weights.
    :return: None
    '''
    path = '../results/' + str(date.today()) + '/model/'
    os.makedirs(path, exist_ok=True)

    # Create OLS object to save
    ols_obj = model_obj.ols_regression()

    # BERT predictions (if model run with string feature(s))
    try:
        bert_pred_train = model_obj.bert_pred_train
        bert_pred_val = model_obj.bert_pred_val
    except AttributeError:
        bert_pred_train = pd.DataFrame()
        bert_pred_val = pd.DataFrame()

    # Train, val dataframes transformed back to human-readable form with prediction
    train, val = model_obj.generate_prediction()

    # Save model weights
    try:
        model_obj.bert_model.save_weights(path + 'bert_weights.h5')
    except AttributeError:
        pass
    model_obj.main_model.save_weights(path + 'main_weights.h5')

    # Save model description as serialized JSON files
    try:
        bert_model_json = model_obj.bert_model.to_json()
    except AttributeError:
        bert_model_json = {}
    main_model_json = model_obj.main_model.to_json()

    # Save model history as dictionary
    try:
        bert_model_hist = model_obj.bert_model.history.history
    except AttributeError:
        bert_model_hist = {}
    main_model_hist = model_obj.main_model.history.history

    with open(path + 'model.pkl', 'wb') as p:
        pickle.dump(model_obj.config, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_obj.data, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_obj.X_train, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_obj.X_val, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_obj.categorical_mappings, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_obj.numeric_features, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_obj.string_features, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(bert_model_hist, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(main_model_hist, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(bert_model_json, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(main_model_json, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ols_obj, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(bert_pred_train, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(bert_pred_val, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train, p, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val, p, pickle.HIGHEST_PROTOCOL)

    # Save results of loss over epochs
    plot_diagnostic(diagnostic='r_square', constrain_N=5, plt_path='../results/' + str(date.today()) + '/figures/main_rsquare.png')
    plot_diagnostic(diagnostic='mean_squared_error', constrain_N=5, plt_path='../results/' + str(date.today()) + '/figures/main_mse.png')
    if model_obj.bert_model is not None:
        plot_diagnostic(diagnostic='r_square', constrain_N=1, plt_path='../results/'+str(date.today())+'/figures/bert_rsquare.png', bert=True)
        plot_diagnostic(diagnostic='mean_squared_error', constrain_N=1, plt_path='../results/' + str(date.today()) + '/figures/bert_mse.png', bert=True)
    plt.close()
    print("\nResults saved to disk.")


if __name__ == '__main__':

    # Allow categorical and numeric features
    model_obj = RegressionPrediction(train_network=True, embed_categoricals=True)
    save_results() # save


