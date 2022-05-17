import tensorflow as tf # my version is 2.5.0
import tensorflowjs as tfjs
import numpy as np

# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# https://github.com/tqkhai2705/edge-detection/blob/master/Canny-TensorFlow.py

"""
 NOTE: 	All variables are initialized first for reducing proccessing time.
"""

np_filter_0 = np.zeros((3,3,1,2))
np_filter_0[1,0,0,0], np_filter_0[1,2,0,1] = 1,1 ### Left & Right
np_filter_90 = np.zeros((3,3,1,2))
np_filter_90[0,1,0,0], np_filter_90[2,1,0,1] = 1,1 ### Top & Bottom
np_filter_45 = np.zeros((3,3,1,2))
np_filter_45[0,2,0,0], np_filter_45[2,0,0,1] = 1,1 ### Top-Right & Bottom-Left
np_filter_135 = np.zeros((3,3,1,2))
np_filter_135[0,0,0,0], np_filter_135[2,2,0,1] = 1,1 ### Top-Left & Bottom-Right

filter_0 = tf.expand_dims(tf.constant(np_filter_0, tf.float32), axis=0)
filter_90 = tf.expand_dims(tf.constant(np_filter_90, tf.float32), axis=0)
filter_45 = tf.expand_dims(tf.constant(np_filter_45, tf.float32), axis=0)
filter_135 = tf.expand_dims(tf.constant(np_filter_135, tf.float32), axis=0)
	
np_filter_sure = np.ones([3,3,1,1])
np_filter_sure[1,1,0,0] = 0
filter_sure = tf.expand_dims(tf.constant(np_filter_sure, tf.float32), axis=0)
border_paddings = tf.expand_dims(tf.constant([[0,0],[1,1],[1,1],[0,0]]), axis=0)

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_kernels():
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    return Kx, Ky

GAUS_KERNEL=3
GAUS_SIGMA=1.2
MIN_RATE=.1
MAX_RATE=.2
RESIZED_SHAPE = 200

def create_model(norm=True, MAX=1.):
    x = tf.keras.Input((None,None,3))
    y = tf.keras.layers.experimental.preprocessing.Resizing(RESIZED_SHAPE, RESIZED_SHAPE)(x)
    y = tf.image.rgb_to_grayscale(y)

    # noise reduction
    gauss_conv = tf.keras.layers.Conv2D(1, GAUS_KERNEL, 1, padding="same", use_bias=False, name="gaussian_blur")(y)
    
    # horizontal and vertical derivatives
    sobel_convX = tf.keras.layers.Conv2D(1, 3, 1, padding="same", use_bias=False, name="sobel_x")(gauss_conv)
    sobel_convY = tf.keras.layers.Conv2D(1, 3, 1, padding="same", use_bias=False, name="sobel_y")(gauss_conv)

    G = tf.math.sqrt(tf.math.square(sobel_convX) + tf.math.square(sobel_convY))
    
    if norm is True:
        MAX = 1.
        mx = tf.math.reduce_max(G, axis=2)
        mx = tf.math.reduce_max(mx, axis=1)    
        G = G / tf.reshape(mx, (-1,1,1,1)) * MAX
    else:
        MAX = 255.
    
    theta = tf.math.atan2(sobel_convY, sobel_convX)  
    theta = (theta*180/np.pi)%180  
    
    D_0 = tf.cast(tf.greater_equal(theta,157.5), tf.float32) + tf.cast(tf.less(theta,22.5), tf.float32)
    D_45 = tf.cast(tf.greater_equal(theta,22.5), tf.float32) + tf.cast(tf.less(theta,67.5), tf.float32)
    D_90 = tf.cast(tf.greater_equal(theta,67.5), tf.float32) + tf.cast(tf.less(theta,112.5), tf.float32)
    D_135 = tf.cast(tf.greater_equal(theta,112.5), tf.float32) + tf.cast(tf.less(theta,157.5), tf.float32)
 
    # non-maximum suppression
    targetPixels_0 = tf.keras.layers.Conv2D(2, 3, 1, padding="same", use_bias=False, name="target_pixels_0")(G)
    targetPixels_90 = tf.keras.layers.Conv2D(2, 3, 1, padding="same", use_bias=False, name="target_pixels_90")(G)
    targetPixels_45 = tf.keras.layers.Conv2D(2, 3, 1, padding="same", use_bias=False, name="target_pixels_45")(G)
    targetPixels_135 = tf.keras.layers.Conv2D(2, 3, 1, padding="same", use_bias=False, name="target_pixels_135")(G)
    
    isGreater_0 = tf.cast(tf.greater(G*D_0, targetPixels_0), tf.float32)
    isMax_0 = isGreater_0[:,:,:,0:1]*isGreater_0[:,:,:,1:2]
    
    isGreater_90 = tf.cast(tf.greater(G*D_90, targetPixels_90), tf.float32)
    isMax_90 = isGreater_90[:,:,:,0:1]*isGreater_90[:,:,:,1:2]
    
    isGreater_45 = tf.cast(tf.greater(G*D_45, targetPixels_45), tf.float32)
    isMax_45 = isGreater_45[:,:,:,0:1]*isGreater_45[:,:,:,1:2]
    
    isGreater_135 = tf.cast(tf.greater(G*D_135, targetPixels_135), tf.float32)
    isMax_135 = isGreater_135[:,:,:,0:1]*isGreater_135[:,:,:,1:2]
    
    edges_raw = G*(isMax_0 + isMax_90 + isMax_45 + isMax_135)
    edges_raw = tf.clip_by_value(edges_raw, 0., MAX)

    # hysteresis thresholding
    edges_sure = tf.cast(tf.greater_equal(edges_raw,  MAX_RATE ), tf.float32)
    edges_weak = tf.cast(tf.less(edges_raw, MAX_RATE), tf.float32)*tf.cast(tf.greater_equal(edges_raw, MIN_RATE), tf.float32)
    
    connect_conv = tf.keras.layers.Conv2D(1, 3, 1, padding="same", use_bias=False, name="edges_connected")
    edges_connected = connect_conv(edges_sure) * edges_weak
    
    for _ in range(10):
        edges_connected = connect_conv(edges_connected) * edges_weak
    
    edges_final = edges_sure + tf.clip_by_value(edges_connected, 0., MAX)

    model = tf.keras.models.Model(x, edges_final)
    return model

Kg = gaussian_kernel(GAUS_KERNEL, GAUS_SIGMA)
Kx, Ky = sobel_kernels()
weights = {
    "gaussian_blur": np.reshape(Kg, (1,GAUS_KERNEL,GAUS_KERNEL,1,1)),
    "sobel_x": np.reshape(Kx, (1,3,3,1,1)),
    "sobel_y": np.reshape(Ky, (1,3,3,1,1)),
    "target_pixels_0": filter_0,
    "target_pixels_90": filter_90,
    "target_pixels_45": filter_45,
    "target_pixels_135": filter_135,
    "edges_connected": filter_sure
}

model = create_model(norm=True, MAX=1.)

for layer in model.layers:
    if layer.name in weights:
        layer.set_weights(weights[layer.name])

model.trainable = False
model.compile(loss=None)


#######################################
# save to h5 (keras) 
# to load model : model = tf.keras.models.load_model(<path>)
model.save("./canny_model/tf_canny_edge.h5")

#######################################
# save to json format to use in web browser or phone app
tfjs.converters.save_keras_model(model, './canny_model/tf_canny_edge')


#######################################
# tf 2.5.0 does not support tf.atan2 for tflite models
# so visit https://www.tensorflow.org/lite/guide/ops_select if you have an ugraded version

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]

# tflite_model = converter.convert()
# with open("./canny_model/tf_canny_edge.tflite", "wb") as f:
#     f.write(tflite_model)