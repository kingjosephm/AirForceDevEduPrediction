import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from RegressionPrediction import RegressionPrediction
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 550)

def show_diagnostic(diagnostic='r_square', constrain_N=20):

    if diagnostic == 'r_square':
        train_temp = model_obj.history.history[diagnostic]
        val_temp = model_obj.history.history['val_'+diagnostic]

        for i in range(constrain_N):
            train_temp[i] = np.NaN
            val_temp[i] = np.NaN

        plt.close()
        plt.plot(train_temp)
        plt.plot(val_temp)
        #plt.xlim(0, len(train_temp))
        plt.title('Coefficient of Determination (R2)')
        plt.ylabel('Share of variance explained')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
        plt.show()

    if diagnostic == 'mean_squared_error':
        train_temp = model_obj.history.history[diagnostic]
        val_temp = model_obj.history.history['val_'+diagnostic]

        for i in range(constrain_N):
            train_temp[i] = np.NaN
            val_temp[i] = np.NaN

        plt.close()
        plt.plot(train_temp)
        plt.plot(val_temp)
        #plt.xlim(0, len(train_temp))
        plt.title('Mean Squared Error')
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'])
        plt.show()


if __name__ == '__main__':


    # Allow categorical and numeric features
    model_obj = RegressionPrediction(embed_categoricals=False)
    train, val = model_obj.generate_prediction()

    #shap = model.shap_values(n=2)


    show_diagnostic(diagnostic='r_square', constrain_N=30)
    show_diagnostic(diagnostic='mean_squared_error', constrain_N=30)

    plt.savefig('../results/r2_performance.png', dpi=350)
    plt.savefig('../results/mse_performance.png', dpi=350)