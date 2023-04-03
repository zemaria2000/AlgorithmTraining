# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, d2_pinball_score
import math
from settings import TRAIN_SPLIT, PREVIOUS_STEPS, VARIABLES, DATA_DIR
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# 1. CHOOSING A VARIABLE TO PREDICT 
var_to_predict = input(f'Choose a model from the following list\n {VARIABLES}: ')
if var_to_predict not in VARIABLES:
    print("Model spelt incorretly / model doesn't exist in the list given")
    var_to_predict = input(f"Please select a model from the list \n {VARIABLES}: ")


# -----------------------------------------------------------------------------
# 2. LOADING THE MODEL TRAINED
model = load_model(f'models/{var_to_predict}.h5')


# -----------------------------------------------------------------------------
# 3. LOADING THE DATASET TO SPLIT INTO TESTING AND TRAINING
input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
# retrieving the variables in which we are interested
df = input_df[['Date', f'{var_to_predict}']]
# Converting the date
# df['Date'] = pd.to_datetime(df['Date'])

#----------------------------------------------------------------------------------------------------------------------------
# NORMALIZING THE DATA
scaler = MinMaxScaler()
df[f'{var_to_predict}'] = scaler.fit_transform(np.array(df[f'{var_to_predict}']).reshape(-1, 1))


# -----------------------------------------------------------------------------
# 4. CREATING THE TEST SPLIT

# splitting data between training and testing
train_data_size = int(TRAIN_SPLIT * len(df)) 
train_data = df[:train_data_size]
test_data = df[train_data_size:len(df)]

# Function that divides our dataset according to the previous_steps number
# watch this - https://www.youtube.com/watch?v=6S2v7G-OupA&t=888s&ab_channel=DigitalSreeni
# this function will take our data and for each 'previous_steps' timestamps, the next one is saved in the y_values
def divide_time_series(x, y, prev_steps):
    x_values = []
    y_values = []

    for i in range(len(x)-prev_steps):
        x_values.append(x.iloc[i:(i+prev_steps)].values)
        y_values.append(y.iloc[i+prev_steps])

    return np.array(x_values), np.array(y_values)

# Training and Testing dataset 
train_X, train_y = divide_time_series(x = train_data[f'{var_to_predict}'],
                                y = train_data[f'{var_to_predict}'],
                                prev_steps = PREVIOUS_STEPS)
test_X, test_y = divide_time_series(x = test_data[f'{var_to_predict}'],
                                y = test_data[f'{var_to_predict}'],
                                prev_steps = PREVIOUS_STEPS)


# -----------------------------------------------------------------------------
# 5. PREDICTIONG ON OUR TEST DATA
test_predict = model.predict(test_X)
# Generating a vector with the y predictions
test_predict_y = []
for i in range(len(test_predict)):
    test_predict_y.append(test_predict[i][PREVIOUS_STEPS-1])


# -----------------------------------------------------------------------------
# 6. ANALYSING OUR DATA
# Some important indicators
mse = mean_squared_error(test_y, test_predict_y)
rmse = math.sqrt(mse)
r2 = r2_score(test_y, test_predict_y)
pinball = d2_pinball_score(test_y, test_predict_y)
list = {'model': f'{model}', 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2], 'Pinball': [pinball]}
print(list)

# to remove after....

# manipulate a dataset



df1 = pd.read_csv(f"datasets/ReacEc_L1.csv")
df2 = pd.read_csv(f"datasets/ReacEc_L3.csv")
df3 = pd.read_csv(f"datasets/RealE_SUM.csv")
df4 = pd.read_csv(f"datasets/H_TDH_I_L3_N.csv")
plt.style.use('ggplot')
plt.subplot(1,2,1)
plt.plot(df1['ReacEc_L1'], '-r', linewidth = 0.7)
plt.plot(df2['ReacEc_L3'], '-b', linewidth = 0.7)
plt.plot(df3['RealE_SUM'], '-g', linewidth = 0.7)
plt.legend(['ReacEc_L1', 'ReacEc_L3', 'RealE_SUM'])
plt.subplot(1,2,2)
plt.plot(df4['H_TDH_I_L3_N'], '--k', linewidth = 0.3)
plt.legend(['U_L1_N'])

plt.show()



# # PLOTTING A GRAPH
# plt.plot(test_y, linewidth = 0.5)
# plt.plot(test_predict_y, linewidth = 0.5)
# plt.title('Comparison between real and predicted data')
# plt.legend(['Real Values', 'Predicted Values'])
# # plt.savefig(f"{FIGURES_DIR}{model}_RealVSPrediction.png")
# plt.show()


