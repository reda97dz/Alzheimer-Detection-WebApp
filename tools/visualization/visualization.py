from tensorflow import keras
from PIL import Image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np

def get_img_array(img_path, size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    array = np.array(img)
    array = array*1./255
    plt.imshow(array, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    array = np.expand_dims(array, axis=0)
    print("returning image array")
    return array

# Feature maps
def show_featuremaps(img_path,conv_base,size,square,level,figsize=(20,20)):
	img = get_img_array(img_path, size)
	model = Model(inputs=conv_base.inputs, outputs=conv_base.layers[level].output)
	feature_maps = model.predict(img)
	print(feature_maps[0,:,:,0].shape)
	print(feature_maps.shape)
	ix = 1
	plt.figure(figsize=figsize)
	for _ in range(square):
		for _ in range(square):
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(feature_maps[0,:,:,ix-1], cmap='gray')
			if ix >= feature_maps.shape[3]:
				break
			ix = ix + 1 
	
	plt.subplots_adjust(wspace=0, hspace=0)
	
	plt.show()


# Heatmap
def make_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    conv_base = model.get_layer('inception_resnet_v2')
    
    last_conv_layer = conv_base.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(conv_base.input, last_conv_layer.output)
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
      x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Made gradcam
def gradcam(model,imgpath,last_conv_layer_name='conv_7b_ac'):
    classifier_layer_names = []
    for layer in model.layers[1:]:
        classifier_layer_names.append(layer.name)

    img = get_img_array(imgpath, (160,256))

    preds = model.predict(img)
    heatmap = make_heatmap(
        img, model, last_conv_layer_name, classifier_layer_names
    )
    img = keras.preprocessing.image.load_img(imgpath)
    img = keras.preprocessing.image.img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4  + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return preds,superimposed_img
