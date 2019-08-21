import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import math
from scipy.stats import norm
from statsmodels.tsa import stattools
import keras
from keras import backend as K
from keras import objectives
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape, Conv2DTranspose, LSTM
from keras.models import Model, Sequential
from PIL import Image
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import os
from contextlib import redirect_stdout
from sklearn.cluster import KMeans

# load images
direc = "pictures/"
data = []
for i in range(300):
    if i < 10:
        im = Image.open(direc + "frame00"+str(i)+".png")
    elif i >= 10 and i < 100:
        im = Image.open(direc + "frame0"+str(i)+".png")
    elif i >= 100 and i < 110:
        im = Image.open(direc + "frame10"+str(i-100)+".png")
    elif i >= 110 and i < 200:
        im = Image.open(direc + "frame1"+str(i-100)+".png")
    elif i >= 200 and i < 210:
        im = Image.open(direc + "frame20"+str(i-200)+".png")
    else:
        im = Image.open(direc + "frame2"+str(i-200)+".png")
    data.append(np.array(im))
frames = np.array(data)

# split dataset into train and test
x_train_1 = frames[0:290]
x_test_1 = frames[290:297]

x_train_1 = x_train_1.astype('float32') / 255.
x_test_1 = x_test_1.astype('float32') / 255.

#hyperparameters
batch_size = 50
image_size = 64
original_dim = image_size * image_size
latent_dim = 8
intermediate_dim = 256
nb_epoch = 200
epsilon_std = 1.0
kernel_size = 2
filters = 1024
input_shape = (image_size, image_size, 3)
conv_layers = 6
dense_layers = 1
digit_size = 64

# create encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(conv_layers):
    
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    filters //= 2
shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(latent_dim * 2, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# sample the latent variable

def new_sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    alpha = 0.001 # limit the stochasticity to reduce bluriness
    return z_mean + K.exp(0.5 * z_log_var) * epsilon * alpha
z = Lambda(new_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# create the decoder

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(latent_dim * 2, activation='relu')(latent_inputs)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)
filters *= 4
for i in range(conv_layers):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters *= 2
outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

output = decoder(encoder(inputs)[2])
cvae = Model(inputs, output, name='cvae')
reconstruction_loss = objectives.binary_crossentropy(K.flatten(inputs),K.flatten(output))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
cvae_loss = K.mean(reconstruction_loss + 3 * kl_loss) # use disentangled VAE for better results (beta = 3)
cvae.add_loss(cvae_loss)
cvae.compile(optimizer='adam')

#save summaries of parts of the VAE
directory_models = "models"
if not os.path.exists(directory_models):
    os.makedirs(directory_models)
    
with open(directory_models + '/encoder_summary.txt', 'w') as f:
    with redirect_stdout(f):
        encoder.summary()

with open(directory_models + '/decoder_summary.txt', 'w') as f:
    with redirect_stdout(f):
        decoder.summary()

with open(directory_models + '/cvae_summary.txt', 'w') as f:
    with redirect_stdout(f):
        cvae.summary()

logger = keras.callbacks.TensorBoard(log_dir = "log", write_graph = True)

cvae_history = cvae.fit(x = x_train_1,
                        shuffle = True,
                        epochs=150,
                        callbacks = [logger],
                        batch_size=50)

directory_cvae_training = "cvae_training"
if not os.path.exists(directory_cvae_training):
    os.makedirs(directory_cvae_training)

# plot history
plt.figure(figsize = (10,latent_dim))
plt.title("Training history of VAE")
plt.plot(cvae_history.history['loss'], label='train')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss function")
plt.savefig(directory_cvae_training + "/graph.png")
#plt.show()
plt.clf()

# make directory to store reconstructed images from VAE model

directory_recon = "reconstruction"
if not os.path.exists(directory_recon):
    os.makedirs(directory_recon)

# present reconstructed image from training set	
	 
fig = plt.figure("Reconstruct_train")
tr = 99
original_train = (x_train_1[tr] * 255).astype(np.uint8)
created_train = (cvae.predict(x_train_1[tr].reshape(1, image_size,image_size,3)).reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
images_train = ("Original", original_train), ("Reconstruction", created_train)
plt.title("Frame " + str(tr+1)+ "\n PSNR: " + str(psnr(original_train, created_train))+ ", SSIM: " + str(ssim(original_train, created_train, multichannel=True)))
plt.axis("off")
for (i, (name, image)) in enumerate(images_train):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(image)
    plt.axis("off")

# save and show the figure
plt.savefig(directory_recon + '/recon_train.png')
plt.clf()

# present reconstructed image from test set

fig2 = plt.figure("Reconstruct_test")
te = 0
original_test = (x_test_1[te] * 255).astype(np.uint8)
created_test = (cvae.predict(x_test_1[te].reshape(1, image_size,image_size,3)).reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
images_test = ("Original", original_test), ("Reconstruction", created_test)
plt.title("Frame " + str(x_train_1.shape[0]+1+te) + "\n PSNR: " + str(psnr(original_test, created_test))+ ", SSIM: " + str(ssim(original_test, created_test, multichannel=True)))
plt.axis("off")
for (i, (name, image)) in enumerate(images_test):
    ax = fig2.add_subplot(1, 2, i + 1)
    ax.set_title(name)
    plt.imshow(image)
    plt.axis("off")

plt.savefig(directory_recon + '/recon_test.png')
plt.clf()

# store mu, sigma and sample latent variable for each frame on the training set

m_t = []
log_var_t = []
z_t = []

for i in range(x_train_1.shape[0]):
    temp = encoder.predict(x_train_1[i].reshape(1, image_size, image_size, 3))
    m_t.append(temp[0][0])
    log_var_t.append(temp[1][0])
    z_t.append(temp[2][0])

# create data frame of the latent variables
	
names = ['feature'+str(i+1) for i in range(z_t[0].shape[0])]
df = pd.DataFrame(data = z_t, columns = names)
groups = [g for g in range(latent_dim)]

# plot each feature

directory_features = "features"
if not os.path.exists(directory_features):
    os.makedirs(directory_features)
values = df.values
# plot each column

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    plt.plot(values[:, group])
    plt.title("Feature " + (str(group + 1)) )
    plt.xlabel("Frames")
    plt.ylabel("Values")
    plt.savefig(directory_features + "/feature"+str(group + 1)+".png")
    plt.clf()
    
# create data frame with latent variables shifted for one step

df_new = df.shift()

# group the features into clusters

df_T = df.T

wcss = []

directory_kmeans_graph = "kmeans_graph"
if not os.path.exists(directory_kmeans_graph):
    os.makedirs(directory_kmeans_graph)

for i in range(1, latent_dim):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_T)
    wcss.append(kmeans.inertia_)
plt.bar(range(1, latent_dim), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig(directory_kmeans_graph + "/elbow_bar.png")
plt.clf()

plt.plot(range(1, latent_dim), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig(directory_kmeans_graph + "/elbow_line.png")
plt.clf()

plt.plot(range(1, latent_dim), wcss, "r")
plt.bar(range(1, latent_dim), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig(directory_kmeans_graph + "/elbow.png")
plt.clf()

#example of furter reduction of the features
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
clusters = kmeans.fit_predict(df_T)


directory_kmeans_features = "kmeans_features"
if not os.path.exists(directory_kmeans_features):
    os.makedirs(directory_kmeans_features)

count = 0
for c in range(latent_dim):
    if c > max(clusters):
        break
    ft = []
    fig = plt.figure(figsize=(10,latent_dim))
    ax = fig.add_subplot(111)
    for i in range(len(clusters)):
        if c == clusters[i]:
            ax.plot(df.values[:,i], label = "feature " + str(i+1))
            ft.append(i)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
    svfg = ""
    tlt = ""
    for f in ft:
        svfg += str(f+1)
        tlt += str(f+1) + ","
    if len(ft) == 1:
        plt.title("Feature: "+tlt[:-1])
        plt.ylabel('Values of feature')
    else:
        plt.title("Features: "+tlt[:-1])
        plt.ylabel('Values of features')        
    plt.xlabel('Frames')
    fig.savefig(directory_kmeans_features + "/features" + svfg + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.clf()
    
# create acf graphs for each feature

directory_acf = "acf/simple"
if not os.path.exists(directory_acf):
    os.makedirs(directory_acf)
	
i = 1
# plot each column
acf_res = []

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    temp = stattools.acf(df[names[group]])
    plt.bar(range(len(temp)), temp, width = 0.1)
    plt.plot(temp, "go")
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.title("ACF for feature " + str(group + 1))
    plt.axhline(y = 0, linestyle = "--")
    plt.axhline(y = -1.96/np.sqrt(len(df[names[group]])), linestyle = "--")
    plt.axhline(y = 1.96/np.sqrt(len(df[names[group]])), linestyle = "--")
    acf_res.append(temp)
    plt.savefig(directory_acf + "/feature" + str(i) + "_acf.png")	
    i += 1
    plt.clf()
    
# create pacf graphs for each feature

directory_pacf = "pacf/simple"
if not os.path.exists(directory_pacf):
    os.makedirs(directory_pacf)

i = 1
pacf_res = []

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    temp = stattools.pacf(df[names[group]])
    plt.bar(range(len(temp)), temp, width = 0.1)
    plt.plot(temp, "ro")
    plt.xlabel("Lags")
    plt.ylabel("PACF")
    plt.title("PACF for feature " + str(group + 1))
    plt.axhline(y = 0, linestyle = "--")
    plt.axhline(y = -1.96/np.sqrt(len(df[names[0]])), linestyle = "--")
    plt.axhline(y = 1.96/np.sqrt(len(df[names[0]])), linestyle = "--")
    pacf_res.append(temp)
    plt.savefig(directory_pacf + "/feature" + str(i) + "_pacf.png")
    i += 1
    plt.clf()
    
directory_hist = "histograms/simple"
if not os.path.exists(directory_hist):
    os.makedirs(directory_hist)

for g in groups:
    plt.figure(figsize = (10,latent_dim))
    df[names[g]].hist()
    plt.title("Feature " + str(g + 1))
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    plt.savefig(directory_hist + "/hist_feature_" + str(g+1) + ".png")
    plt.clf()
    
# create function to calculate the predictions

def predict(coef, vec):
    pred = 0.0
    for i in range(1, len(coef)+1):
        pred += coef[i-1] * vec[-i]
    return pred

# make prediction using same arima (2,1,0) order for all features

end = x_test_1.shape[0]
new_z_df = pd.DataFrame(np.zeros((end,latent_dim)), columns = names)
params = []
for g in range(latent_dim):
    feature = list(df[names[g]].values)
    model = ARIMA(feature, order = (2,1,0))
    model_fit = model.fit()
    ar_coef = model_fit.arparams
    params.append(ar_coef)
    outcome = []
    for i in range(end):
        yhat = predict(ar_coef, feature)
        feature.append(yhat)
        outcome.append(yhat)
    new_z_df[names[g]] = pd.DataFrame(outcome)

new_z = []
for i in range(new_z_df.shape[0]):
    new_z.append(new_z_df.iloc[i].values)	
	
# display the predictions for the naive time series
    
directory_naive_time_series_original = "naive_time_series/original"
if not os.path.exists(directory_naive_time_series_original):
    os.makedirs(directory_naive_time_series_original)

directory_naive_time_series_predictions = "naive_time_series/predictions"
if not os.path.exists(directory_naive_time_series_predictions):
    os.makedirs(directory_naive_time_series_predictions)	
	
	
psnr_tm = []
ssim_tm = []
for i in range(len(new_z)):
    pred_z = decoder.predict(new_z[i].reshape(1,latent_dim))
    digit = (pred_z[0].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    original = (x_test_1[i].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    orig = Image.fromarray(original)
    orig.save(directory_naive_time_series_original + "/frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
    pred = Image.fromarray(digit)
    pred.save(directory_naive_time_series_predictions + "/prediction_for_frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
    psnr_tm.append(psnr(original, digit))
    ssim_tm.append(ssim(original,digit, multichannel=True))
 
tm1 = pd.DataFrame()
tm1["PSNR"], tm1["SSIM"] = psnr_tm, ssim_tm
directory_naive_time_series_values = "naive_time_series/psnr_ssim_values"
if not os.path.exists(directory_naive_time_series_values):
    os.makedirs(directory_naive_time_series_values)
tm1.to_csv(directory_naive_time_series_values + "/values.csv",
			encoding='utf-8')
 
directory_naive_time_series_graph = "naive_time_series/graph"
if not os.path.exists(directory_naive_time_series_graph):
    os.makedirs(directory_naive_time_series_graph)
    
frm = [x_train_1.shape[0]+1+i for i in range(x_test_1.shape[0])]
plt.figure(figsize = (10,latent_dim))
plt.plot(frm, psnr_tm, "-*")
plt.xlabel("Frames")
plt.ylabel("PSNR")
plt.title("PSNR Scores for Naive Time Series")
plt.savefig(directory_naive_time_series_graph + "/psnr_naive_time_series.png")
plt.clf()
plt.figure(figsize = (10,latent_dim))
plt.plot(frm, ssim_tm, "-*", color = "red")
plt.xlabel("Frames")
plt.title("SSIM Scores for Naive Time Series")
plt.savefig(directory_naive_time_series_graph + "/ssim_naive_time_series.png")
plt.clf()

# make predictions with optimal time series

new_z_df3 = pd.DataFrame(np.zeros((end,latent_dim)), columns = names)
store_order = pd.DataFrame(np.zeros((latent_dim,3)), index = names, columns = ["p", "d", "q"])
for name in names:
    feature = list(df[name].values)
    stepwise_model = auto_arima(feature, start_p=1, start_d=0,
                               max_p=4, max_d=2,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
    order = [int(i) for i in stepwise_model.order]
    store_order.loc[name, :] = order
    outcome = stepwise_model.fit_predict(df[name], n_periods= x_test_1.shape[0])
    new_z_df3[name] = pd.DataFrame(outcome)

store_order = store_order.astype(int)
directory_orders = "orders"
if not os.path.exists(directory_orders):
    os.makedirs(directory_orders)
store_order.to_csv(directory_orders + "/orders.csv",
                  encoding='utf-8')

# display the predictions for the optimal time series

directory_optimal_time_series_original = "optimal_time_series/original"
if not os.path.exists(directory_optimal_time_series_original):
    os.makedirs(directory_optimal_time_series_original)

directory_optimal_time_series_predictions = "optimal_time_series/predictions"
if not os.path.exists(directory_optimal_time_series_predictions):
    os.makedirs(directory_optimal_time_series_predictions)	

new_z3 = []
for i in range(new_z_df3.shape[0]):
    new_z3.append(new_z_df3.iloc[i].values)
psnr_tm3 = []
ssim_tm3 = []
for i in range(len(new_z3)):
    pred_z = decoder.predict(new_z3[i].reshape(1,latent_dim))
    digit = (pred_z[0].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    original = (x_test_1[i].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    orig = Image.fromarray(original)
    orig.save(directory_optimal_time_series_original + "/frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
    pred = Image.fromarray(digit)
    pred.save(directory_optimal_time_series_predictions + "/prediction_for_frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
    psnr_tm3.append(psnr(original, digit))
    ssim_tm3.append(ssim(original,digit, multichannel=True))

tm3 = pd.DataFrame()
tm3["PSNR"], tm3["SSIM"] = psnr_tm3, ssim_tm3
directory_optimal_time_series_values = "optimal_time_series/psnr_ssim_values"
if not os.path.exists(directory_optimal_time_series_values):
    os.makedirs(directory_optimal_time_series_values)
tm3.to_csv(directory_optimal_time_series_values + "/values.csv",
           encoding='utf-8')

directory_optimal_time_series_graph = "optimal_time_series/graph"
if not os.path.exists(directory_optimal_time_series_graph):
    os.makedirs(directory_optimal_time_series_graph)
plt.figure(figsize = (10,latent_dim))
plt.plot(frm, psnr_tm3, "-*")
plt.xlabel("Frames")
plt.ylabel("PSNR")
plt.title("PSNR Scores for Naive Time Series")
plt.savefig(directory_optimal_time_series_graph + "/psnr_naive_time_series.png")
plt.clf()
plt.figure(figsize = (10,latent_dim))
plt.plot(frm, ssim_tm3, "-*", color = "red")
plt.xlabel("Frames")
plt.title("SSIM Scores for Naive Time Series")
plt.savefig(directory_optimal_time_series_graph + "/ssim_naive_time_series.png")
plt.clf()

# time series analysis

# check if the features are stationary

directory_stationary0_test = "stationary_analysis/stationary0_test"
if not os.path.exists(directory_stationary0_test):
    os.makedirs(directory_stationary0_test)

# ADF test

stationary0_test = {}
for name in names:
    usefull_values_raw = adfuller(df[name],
                                  autolag = "AIC",
                                  regression = "c")[:5]
    usefull_values = [val for val in usefull_values_raw[:4]]
    usefull_values.extend([usefull_values_raw[4]["1%"],
                           usefull_values_raw[4]["5%"],
                           usefull_values_raw[4]["10%"],])
    stationary0_test[name] = pd.DataFrame({"Label": ["Test Statistic",
                                                      "p-value",
                                                      "# Lags Used",
                                                      "# of Observations Used",
                                                      "Critical value for 1%",
                                                      "Critical value for 5%",
                                                      "Critical value for 10%"],
                                            "Value": usefull_values})
    stationary0_test[name].to_csv(directory_stationary0_test + "/" + name + ".csv",
                                  sep='\t',
                                  encoding='utf-8')

# check if the first difference of the features are stationary

directory_stationary1_test = "stationary_analysis/stationary1_test"
if not os.path.exists(directory_stationary1_test):
    os.makedirs(directory_stationary1_test)

stationary1_test = {}
for name in names:
    feat_diff = df[name] - df[name].shift()
    feat_diff.dropna(inplace = True)
    usefull_values_raw = adfuller(feat_diff,
                                  autolag = "AIC",
                                  regression = "c")[:5]
    usefull_values = [val for val in usefull_values_raw[:4]]
    usefull_values.extend([usefull_values_raw[4]["1%"],
                           usefull_values_raw[4]["5%"],
                           usefull_values_raw[4]["10%"],])
    stationary1_test[name] = pd.DataFrame({"Label": ["Test Statistic",
                                                      "p-value",
                                                      "# Lags Used",
                                                      "# of Observations Used",
                                                      "Critical value for 1%",
                                                      "Critical value for 5%",
                                                      "Critical value for 10%"],
                                            "Value": usefull_values})
    stationary1_test[name].to_csv(directory_stationary1_test + "/diff_" + name+".csv",
                                  sep='\t',
                                  encoding='utf-8')
    
# check if the second difference of the features are stationary

directory_stationary2_test = "stationary_analysis/stationary2_test"
if not os.path.exists(directory_stationary2_test):
    os.makedirs(directory_stationary2_test)

stationary2_test = {}
for name in names:
    feat_diff = df[name] - df[name].shift(2)
    feat_diff.dropna(inplace = True)
    usefull_values_raw = adfuller(feat_diff,
                                  autolag = "AIC",
                                  regression = "c")[:5]
    usefull_values = [val for val in usefull_values_raw[:4]]
    usefull_values.extend([usefull_values_raw[4]["1%"],
                           usefull_values_raw[4]["5%"],
                           usefull_values_raw[4]["10%"],])
    stationary2_test[name] = pd.DataFrame({"Label": ["Test Statistic",
                                                      "p-value",
                                                      "# Lags Used",
                                                      "# of Observations Used",
                                                      "Critical value for 1%",
                                                      "Critical value for 5%",
                                                      "Critical value for 10%"],
                                            "Value": usefull_values})
    stationary2_test[name].to_csv(directory_stationary2_test + "/diff2_" + name+".csv",
                                  sep='\t',
                                  encoding='utf-8')
		
# create acf graphs for each feature with 1 difference

directory_acf_diff_1 = "acf/diff_1"
if not os.path.exists(directory_acf_diff_1):
    os.makedirs(directory_acf_diff_1)
	
i = 1
# plot each column
acf1_res = []

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    target = df[names[group]] - df[names[group]].shift()
    target.dropna(inplace = True)
    temp = stattools.acf(target)
    plt.bar(range(len(temp)),
            temp,
            width = 0.1)
    plt.plot(temp, "go")
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.title("ACF for first difference of feature " + str(group + 1))
    plt.axhline(y = 0, linestyle = "--")
    plt.axhline(y = -1.96/np.sqrt(len(target)), linestyle = "--")
    plt.axhline(y = 1.96/np.sqrt(len(target)), linestyle = "--")
    acf1_res.append(temp)
    plt.savefig(directory_acf_diff_1 + "/diff_1_feature" + str(i) + "_acf.png")	
    i += 1
    plt.clf()
    
# create pacf graphs for each feature with 1 difference

directory_pacf_diff_1 = "pacf/diff_1"
if not os.path.exists(directory_pacf_diff_1):
    os.makedirs(directory_pacf_diff_1)

i = 1
pacf1_res = []

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    target = df[names[group]] - df[names[group]].shift()
    target.dropna(inplace = True)
    temp = stattools.pacf(target)
    plt.bar(range(len(temp)),
            temp,
            width = 0.1)
    plt.plot(temp, "ro")
    plt.xlabel("Lags")
    plt.ylabel("PACF")
    plt.title("PACF for first difference of feature " + str(group + 1))
    plt.axhline(y = 0, linestyle = "--")
    plt.axhline(y = -1.96/np.sqrt(len(target)), linestyle = "--")
    plt.axhline(y = 1.96/np.sqrt(len(target)), linestyle = "--")
    pacf_res.append(temp)
    plt.savefig(directory_pacf_diff_1 + "/diff_1_feature" + str(i) + "_pacf.png")
    i += 1
    plt.clf()
    
directory_hist_diff_1 = "histograms/diff_1"
if not os.path.exists(directory_hist_diff_1):
    os.makedirs(directory_hist_diff_1)

for g in groups:
    plt.figure(figsize = (10,latent_dim))
    temp = df[names[group]] - df[names[g]].shift()
    temp.dropna(inplace = True)
    temp.hist()
    plt.title("First difference of feature " + str(g + 1))
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    plt.savefig(directory_hist_diff_1 + "/hist_diff_1_feature_" + str(g+1) + ".png")
    plt.clf()
    
# create acf graphs for each feature with 2 difference

directory_acf_diff_2 = "acf/diff_2"
if not os.path.exists(directory_acf_diff_2):
    os.makedirs(directory_acf_diff_2)
	
i = 1

# plot each column
acf_res2 = []

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    target = df[names[group]] - df[names[group]].shift(2)
    target.dropna(inplace = True)
    temp = stattools.acf(target)
    plt.bar(range(len(temp)), temp, width = 0.1)
    plt.plot(temp, "go")
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.title("ACF for second difference of feature " + str(group + 1))
    plt.axhline(y = 0, linestyle = "--")
    plt.axhline(y = -1.96/np.sqrt(len(target)), linestyle = "--")
    plt.axhline(y = 1.96/np.sqrt(len(target)), linestyle = "--")
    acf_res.append(temp)
    plt.savefig(directory_acf_diff_2 + "/diff_2_feature" + str(i) + "_acf.png")	
    i += 1
    plt.clf()
    
# create pacf graphs for each feature with 1 difference

directory_pacf_diff_2 = "pacf/diff_2"
if not os.path.exists(directory_pacf_diff_2):
    os.makedirs(directory_pacf_diff_2)

i = 1
pacf_res = []

for group in groups:
    plt.figure(figsize=(10,latent_dim))
    target = df[names[group]] - df[names[group]].shift(2)
    target.dropna(inplace = True)
    temp = stattools.pacf(target)
    plt.bar(range(len(temp)), temp, width = 0.1)
    plt.plot(temp, "ro")
    plt.xlabel("Lags")
    plt.ylabel("PACF")
    plt.title("PACF for second difference of feature " + str(group + 1))
    plt.axhline(y = 0, linestyle = "--")
    plt.axhline(y = -1.96/np.sqrt(len(target)), linestyle = "--")
    plt.axhline(y = 1.96/np.sqrt(len(target)), linestyle = "--")
    pacf_res.append(temp)
    plt.savefig(directory_pacf_diff_2 + "/diff_2_feature" + str(i) + "_pacf.png")
    i += 1
    plt.clf()
    
directory_hist_diff_2 = "histograms/diff_2"
if not os.path.exists(directory_hist_diff_2):
    os.makedirs(directory_hist_diff_2)

for g in groups:
    plt.figure(figsize = (10,latent_dim))
    temp = df[names[group]] - df[names[g]].shift(2)
    temp.dropna(inplace = True)
    temp.hist()
    plt.title("First difference of feature " + str(g + 1))
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    plt.savefig(directory_hist_diff_2 + "/hist_diff_1_feature_" + str(g+1) + ".png")
    plt.clf()

	  
# LSTM approach

# create dataset for the neural network model

def create_dataset(data, n_inputs=1, n_outputs=1, dropnan=True):
    n_features = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    # input sequence for time steps (t-n, ... t-1)
    for i in range(n_inputs, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_features)]
    # predicted sequence for time steps (t, t+1, ... t+n)
    for i in range(0, n_outputs):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('feature%d(t)' % (j+1)) for j in range(n_features)]
        else:
            names += [('feature%d(t+%d)' % (j+1, i)) for j in range(n_features)]
    result = pd.concat(cols, axis=1)
    result.columns = names
    if dropnan:
        result.dropna(inplace=True)
    return result



dataset = df
values = dataset.values
n_steps = 3
lstm_dataset = create_dataset(values, n_steps, 1)

# split dataset into train and test sets

values = lstm_dataset.values
n_train_time_per_frame = 280
train = values[:n_train_time_per_frame, :]
test = values[n_train_time_per_frame:, :]
n_features = latent_dim
n_obs = n_steps * n_features
train_X, train_y = train[:, :n_obs], train[:, n_obs:]
test_X, test_y = test[:, :n_obs], test[:, n_obs:]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

logger_lstm1 = keras.callbacks.TensorBoard(log_dir = "log_lstm1",
                                           write_graph = True,
                                           histogram_freq = 40)

# design model 1
model = Sequential()
model.add(LSTM(latent_dim * 8, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(latent_dim * 4, activation='relu'))
model.add(Dense(latent_dim * 4, activation='relu'))
model.add(Dense(latent_dim * 2, activation='relu'))
model.add(Dense(latent_dim * 2, activation='relu'))
model.add(Dense(latent_dim))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, 
                    train_y, 
                    epochs=200, 
                    batch_size=50, 
                    validation_data=(test_X, test_y), 
                    verbose=2, 
                    callbacks = [logger_lstm1],
                    shuffle=False)

with open(directory_models + '/lstm1_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# display loss function values for training and test set

directory_lstm1_history = "lstm1/history"
if not os.path.exists(directory_lstm1_history):
    os.makedirs(directory_lstm1_history)

# plot history
plt.figure(figsize = (10,latent_dim))
plt.title("Training history of LSTM model 2")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss function")
plt.savefig(directory_lstm1_history + "/graph.png")
plt.clf()

logger_lstm2 = keras.callbacks.TensorBoard(log_dir = "log_lstm2",
                                           write_graph = True)

# design model 2
model2 = Sequential()
model2.add(LSTM(latent_dim * 8,
                return_sequences = True,
                input_shape=(train_X.shape[1], train_X.shape[2])))
model2.add(LSTM(latent_dim * 4))
model2.add(Dense(latent_dim * 4, activation='relu'))
model2.add(Dense(latent_dim * 2, activation='relu'))
model2.add(Dense(latent_dim * 2, activation='relu'))
model2.add(Dense(latent_dim))
model2.compile(loss='mae',
               optimizer='adam')

# fit second model
	
history2 = model2.fit(train_X, 
                      train_y, 
                      epochs=200, 
                      batch_size=50, 
                      validation_data=(test_X, test_y),
                      callbacks = [logger_lstm2],
                      verbose=2)

with open(directory_models + '/lstm2_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model2.summary()

# plot history

directory_lstm2_history = "lstm2/history"
if not os.path.exists(directory_lstm2_history):
    os.makedirs(directory_lstm2_history)

plt.figure(figsize = (10,latent_dim))
plt.title("Training history of LSTM model 2")
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss function")
plt.savefig(directory_lstm2_history + "/graph.png")
plt.clf()

# Evaluate lstm model 1

test_X, test_y = test[:, :n_obs], test[:, n_obs:]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = inv_yhat[:,:n_features]
test_y = test_y.reshape((len(test_y), n_features))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = inv_y[:,:n_features]
rmse = math.sqrt(((inv_y - inv_yhat) ** 2).mean(axis=None))
print('RMSE: %.3f' % rmse)

# display predictions for lstm model 1

lstm_predictions = []
end = frames.shape[0] - x_train_1.shape[0]
temp_test = test[-1, :n_obs]

for i in range(x_test_1.shape[0]):
    store = temp_test[latent_dim:]
    temp = temp_test.reshape((1, 1, temp_test.shape[0]))
    lstm_pred = model.predict(temp[-1].reshape(1,1,3*latent_dim))[0]
    lstm_predictions.append(lstm_pred)
    temp_test = np.concatenate([store, lstm_pred])

directory_lstm1_original = "lstm1/original"
if not os.path.exists(directory_lstm1_original):
    os.makedirs(directory_lstm1_original)

directory_lstm1_predictions = "lstm1/predictions"
if not os.path.exists(directory_lstm1_predictions):
    os.makedirs(directory_lstm1_predictions)

directory_lstm1_result = "lstm1/result"
if not os.path.exists(directory_lstm1_result):
    os.makedirs(directory_lstm1_result)

psnr_lstm = []
ssim_lstm = []
for i in range(len(lstm_predictions)):
    img_prize = decoder.predict(lstm_predictions[i].reshape(1,latent_dim))
    digit_size = 64
    digit = (img_prize[0].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    fig = plt.figure("Images")
    original = (x_test_1[i] * 255).astype(np.uint8)
    created = (img_prize[0].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    images = ("Original", original), ("Prediction", created)
    plt.title("PSNR: " + str(psnr(original, created)) + ", SSIM: " + str(ssim(original,created, multichannel=True)))
    psnr_lstm.append(psnr(original, created))
    ssim_lstm.append(ssim(original,created, multichannel=True))
    plt.axis("off")
    for (j, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, j + 1)
        ax.set_title(name)
        image = Image.fromarray(image, 'RGB')
        if(j==0):
            image.save(directory_lstm1_original + "/frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
        else:
            image.save(directory_lstm1_predictions + "/prediction_for_frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
        plt.axis("off")
    plt.clf()
    
lstm1 = pd.DataFrame()
lstm1["PSNR"], lstm1["SSIM"] = psnr_lstm, ssim_lstm
directory_lstm1_values = "lstm1/psnr_ssim_values"
if not os.path.exists(directory_lstm1_values):
    os.makedirs(directory_lstm1_values)
lstm1.to_csv(directory_lstm1_values + "/values.csv",
             encoding='utf-8')

directory_lstm1_graph = "lstm1/graph"
if not os.path.exists(directory_lstm1_graph):
    os.makedirs(directory_lstm1_graph)
plt.figure(figsize = (10,latent_dim))
plt.plot(frm, psnr_lstm, "-*")
plt.xlabel("Frames")
plt.ylabel("PSNR")
plt.title("PSNR Scores for model 1 LSTM")
plt.savefig(directory_lstm1_graph + "/psnr_lstm1.png")
plt.clf()

plt.figure(figsize = (10,latent_dim))
plt.plot(frm, ssim_lstm, "-*", color = "red")
plt.xlabel("Frames")
plt.ylabel("SSIM")
plt.title("SSIM Scores for model 1 LSTM")
plt.savefig(directory_lstm1_graph + "/ssim_lstm1.png")
plt.clf()
# evaluate lstm model 2

test_X, test_y = test[:, :n_obs], test[:, n_obs:]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
yhat = model2.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = inv_yhat[:,:n_features]
test_y = test_y.reshape((len(test_y), n_features))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = inv_y[:,:n_features]
rmse = math.sqrt(((inv_y - inv_yhat) ** 2).mean(axis=None))
print('RMSE: %.3f' % rmse)

# display predictions for lstm model 2

lstm_predictions2 = []
end = frames.shape[0] - x_train_1.shape[0]
temp_test = test[-1, :n_obs]
for i in range(x_test_1.shape[0]):
    store = temp_test[latent_dim:]
    temp = temp_test.reshape((1, 1, temp_test.shape[0]))
    lstm_pred = model2.predict(temp[-1].reshape(1,1,3*latent_dim))[0]
    lstm_predictions2.append(lstm_pred)
    temp_test = np.concatenate([store, lstm_pred])

directory_lstm2_original = "lstm2/original"
if not os.path.exists(directory_lstm2_original):
    os.makedirs(directory_lstm2_original)

directory_lstm2_predictions = "lstm2/predictions"
if not os.path.exists(directory_lstm2_predictions):
    os.makedirs(directory_lstm2_predictions)

psnr_lstm2 = []
ssim_lstm2 = []
for i in range(len(lstm_predictions2)):
    img_prize = decoder.predict(lstm_predictions2[i].reshape(1,latent_dim))
    digit = (img_prize[0].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    fig = plt.figure("Images")
    original = (x_test_1[i] * 255).astype(np.uint8)
    created = (img_prize[0].reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    images = ("Original", original), ("Prediction", created)
    plt.title("PSNR: " + str(psnr(original, created)) + ", SSIM: " + str(ssim(original,created, multichannel=True)))
    psnr_lstm2.append(psnr(original, created))
    ssim_lstm2.append(ssim(original,created, multichannel=True))
    plt.axis("off")
    for (j, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 2, j + 1)
        ax.set_title(name)
        image = Image.fromarray(image, 'RGB')
        if(j==0):
            image.save(directory_lstm2_original + "/frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
        else:
            image.save(directory_lstm2_predictions + "/prediction_for_frame" + str(x_train_1.shape[0] + 1 + i) + ".png")
        plt.axis("off")

    # show the figure
    #plt.show()
    plt.clf()
lstm2 = pd.DataFrame()
lstm2["PSNR"], lstm2["SSIM"] = psnr_lstm2, ssim_lstm2
directory_lstm2_values = "lstm2/psnr_ssim_values"
if not os.path.exists(directory_lstm2_values):
    os.makedirs(directory_lstm2_values)
lstm2.to_csv(directory_lstm2_values + "/values.csv",
             encoding='utf-8')	

directory_lstm2_graph = "lstm2/graph"
if not os.path.exists(directory_lstm2_graph):
    os.makedirs(directory_lstm2_graph)
plt.figure(figsize = (10,latent_dim))
plt.plot(frm, psnr_lstm2, "-*")
plt.xlabel("Frames")
plt.ylabel("PSNR")
plt.title("PSNR Scores for model 2 LSTM")
plt.savefig(directory_lstm2_graph + "/psnr_lstm2.png")
plt.clf()

plt.figure(figsize = (10,latent_dim))
plt.plot(frm, ssim_lstm, "-*", color = "red")
plt.xlabel("Frames")
plt.ylabel("SSIM")
plt.title("SSIM Scores for model 2 LSTM")
plt.savefig(directory_lstm2_graph + "/ssim_lstm2.png")
plt.clf()

# psnr and ssim graph of all models

directory_graph = "graph"
if not os.path.exists(directory_graph):
    os.makedirs(directory_graph)	
	
fig = plt.figure(figsize=(10,latent_dim))
ax = fig.add_subplot(111)
ax.plot(frm,
        psnr_lstm,
        "-*",
        color = "red",
        label = "lstm_1")
ax.plot(frm,
        psnr_lstm2,
        "-*",
        color = "blue",
        label = "lstm_2")
ax.plot(frm,
        psnr_tm, "-*",
        color = "green",
        label = "naive_time_series")
ax.plot(frm,
        psnr_tm3, "-*",
        color = "yellow",
        label = "optimal_times_series")
plt.xlabel("Frames")
plt.ylabel("PSNR")
plt.title("PSNR Scores for all models")
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, 
                labels,
                loc=2,
                bbox_to_anchor=(1.05, 1))
fig.savefig(directory_graph + "/psnr.png",
            bbox_extra_artists = (lgd,),
            bbox_inches='tight')
plt.clf()

fig = plt.figure(figsize=(10,latent_dim))
ax = fig.add_subplot(111)
ax.plot(frm,
        ssim_lstm,
        "-*",
        color = "red",
        label = "lstm_1")
ax.plot(frm,
        ssim_lstm2,
        "-*",
        color = "blue",
        label = "lstm_2")
ax.plot(frm,
        ssim_tm,
        "-*",
        color = "green",
        label = "naive_time_series")
ax.plot(frm,
        ssim_tm3,
        "-*", 
        color = "yellow", 
        label = "optimal_times_series")
plt.xlabel("Frames")
plt.ylabel("SSIM")
plt.title("SSIM Scores for all models")
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles,
                labels,
                loc=2,
                bbox_to_anchor=(1.05, 1))
plt.savefig(directory_graph + "/ssim.png",
            bbox_extra_artists = (lgd,),
            bbox_inches='tight')
plt.clf()