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
latent_dim = 16
intermediate_dim = 256
nb_epoch = 200
epsilon_std = 1.0
kernel_size = 2
filters = 1024
input_shape = (image_size, image_size, 3)
conv_layers = 6
dense_layers = 1
digit_size = 64

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
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# sample the latent variable

def new_sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    alpha = 0.001 # limit the stochasticity to reduce blurriness
    return z_mean + K.exp(0.5 * z_log_var) * epsilon * alpha
z = Lambda(new_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# create the decoder

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
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

cvae_history = cvae.fit(x = x_train_1,
                        shuffle = True,
                        epochs=150,
                        batch_size=50)

z_t = []

for i in range(x_train_1.shape[0]):
    temp = encoder.predict(x_train_1[i].reshape(1, image_size, image_size, 3))
    z_t.append(temp[2][0])

# create data frame of the latent variables
	
names = ['feature'+str(i+1) for i in range(z_t[0].shape[0])]
df = pd.DataFrame(data = z_t, columns = names)
groups = [g for g in range(latent_dim)]

# group the features into clusters

df_T = df.T

wcss = []

directory_kmeans_graph = "select_number_of_features"
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
