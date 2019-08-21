import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape, Conv2DTranspose, LSTM
from keras.layers import Conv3D, Conv3DTranspose, RNN, Embedding
import keras
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras import objectives
from PIL import Image, ImageFile
import warnings
warnings.filterwarnings("ignore")
import os

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

eps_vals = [(1/(10 ** i)) for i in range(5)]
eps_vals.append(0)
psnr_vals_train = []
ssim_vals_train = []
psnr_vals_test = []
ssim_vals_test = []
for eps in eps_vals:
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
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon * eps  # limit the stochasticity to reduce bluriness
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
    
    cvae.fit(x = x_train_1,
             shuffle = True,
             epochs=150,
             batch_size=50)
    
    directory_recon = "epsilon_grid_search"
    if not os.path.exists(directory_recon):
        os.makedirs(directory_recon)
    
    # present reconstructed image from training set	
    	 
    fig = plt.figure("Reconstruct_train")
    tr = 99
    original_train = (x_train_1[tr] * 255).astype(np.uint8)
    created_train = (cvae.predict(x_train_1[tr].reshape(1, image_size,image_size,3)).reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    images_train = ("Original", original_train), ("Reconstruction", created_train)
    tr_psnr = psnr(original_train, created_train)
    tr_ssim = ssim(original_train, created_train, multichannel=True)
    psnr_vals_train.append(tr_psnr)
    ssim_vals_train.append(tr_ssim)
    plt.axis("off")
    plt.imshow(created_train)
    plt.title("Reconstruction of frame " + str(tr + 1) + " for parameter = " + str(eps) + "\n PSNR: " + str(tr_psnr)+ ", SSIM: " + str(tr_ssim))
    plt.savefig(directory_recon + "/epsilon_" + str(eps) + '_recon_train.png')
    plt.clf()
    
    # present reconstructed image from test set
    
    fig2 = plt.figure("Reconstruct_test")
    te = 0
    original_test = (x_test_1[te] * 255).astype(np.uint8)
    created_test = (cvae.predict(x_test_1[te].reshape(1, image_size,image_size,3)).reshape(digit_size, digit_size,3) * 255).astype(np.uint8)
    images_test = ("Original", original_test), ("Reconstruction", created_test)
    tst_psnr = psnr(original_test, created_test)
    tst_ssim = ssim(original_test, created_test, multichannel=True)
    psnr_vals_test.append(tst_psnr)
    ssim_vals_test.append(tst_ssim)
    plt.axis("off")
    plt.imshow(created_test)
    plt.title("Reconstruction of frame " + str(x_train_1.shape[0]+1+te) + " for parameter = " + str(eps) + "\n PSNR: " + str(tst_psnr)+ ", SSIM: " + str(tst_ssim))
    plt.savefig(directory_recon + "/epsilon_" + str(eps) + '_recon_test.png')
    plt.clf()
    
    
fig3 = plt.figure("Original_train")
plt.axis("off")
plt.title("Frame " + str(tr + 1))
plt.imshow(original_train)
plt.savefig(directory_recon + "/original_train.png")
plt.clf()

fig4 = plt.figure("Original_test")
plt.axis("off")
plt.title("Frame " + str(x_train_1.shape[0]+1+te))
plt.imshow(original_test)
plt.savefig(directory_recon + "/original_test.png")
plt.clf()

df_results = pd.DataFrame()
df_results["Training_PSNR"], df_results["Training_SSIM"], df_results["Test_PSNR"], df_results["Test_SSIM"] = psnr_vals_train, ssim_vals_train, psnr_vals_test, ssim_vals_test
df_results.index = eps_vals
directory_epsilon_results = "epsilon_grid_search"
if not os.path.exists(directory_epsilon_results):
    os.makedirs(directory_epsilon_results)
df_results.to_csv(directory_epsilon_results + "/psnr_ssim_results.csv",
                  encoding='utf-8')