#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov  7 14:46:02 2021

@author: johnklein
"""

#%%
import tensorflow as tf
# if the import fails, try to install tf : pip install --upgrade tensorflow
from sklearn.preprocessing import StandardScaler
import numpy.random as rnd
import numpy as np
import os
import matplotlib.pyplot as plt

rep = "."


#%%
####################
#Dataset generation#
####################

rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100]).astype('float32')
X_test = scaler.transform(data[100:]).astype('float32')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(X_train[:,0], X_train[:,1], X_train[:,2],'.', label='dataset')
ax.legend()
plt.show()

#%%
###############################################
#Dimensionality reduction with an auto-encoder#
###############################################

d_in = 3 #input dimensionality
d_hid = 2 #code dimensionality
d_out = d_in #output dimensionality
learning_rate = 0.1

activation = tf.nn.elu

class basic_AE(tf.Module):
    def __init__(self, unit_nbrs, name=None):
        super().__init__(name=name)
        self.w1 = tf.Variable(tf.random.normal([unit_nbrs[0],unit_nbrs[1]]), name='w')
        self.b1 = tf.Variable(tf.zeros([unit_nbrs[1]]), name='b1')
        self.b2 = tf.Variable(tf.zeros([unit_nbrs[0]]), name='b2')
        self.K = len(unit_nbrs)-1

    @tf.function
    def __call__(self, x):
        z1 = activation(tf.matmul(x,self.w1) + self.b1)
        x_tilde = activation(tf.matmul(z1,tf.transpose(self.w1)) + self.b2)
        return x_tilde

def loss(target,pred):
    return tf.math.reduce_mean(tf.math.squared_difference(target, pred))  

#%% Model creation

mini_ae = basic_AE([d_in, d_hid],name="first_ae")
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
#%% Training
n_epochs = 500
train_loss_history = []
test_loss_history = []

for epoch in range(n_epochs):
    # Computing the function meanwhile recording a gradient tape
    with tf.GradientTape() as tape:
        train_loss = loss(X_train,mini_ae(X_train))
        
    train_loss_history.append(train_loss)
    test_loss_history.append(loss(X_test,mini_ae(X_test)))
    grads = tape.gradient(train_loss,mini_ae.trainable_variables)
    optimizer.apply_gradients(zip(grads, mini_ae.trainable_variables))
    print("Epoch %d  - \tf=%s" % (epoch, train_loss.numpy()),end="")



#%% Plots

fig = plt.figure(figsize=(4,3))
plt.plot(train_loss_history,label='')
plt.plot(test_loss_history)
plt.show()

codings_val = activation(tf.matmul(X_train,mini_ae.w1) + mini_ae.b1)

fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=12)
plt.ylabel("$z_2$", fontsize=12, rotation=0)
plt.title('Dim. Red. with AE')
plt.show()


#%% PCA

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train)  # you may forget centering that is done by sklearn PCA

singval = pca.singular_values_   # eigenvalues
comp = pca.components_           # principal components
proj = pca.transform(X_train)

plt.plot(proj[:,0],proj[:,1],'.')
plt.title('Dim. Red with PCA')


#%%
print(' Les résultats sont relativement proches, mais en essayant plusieurs fois,on se rend compte que la PCA est un peu plus régulière ')

# %%

# %%
