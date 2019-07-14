import cv2

import numpy as np
import sys
import tensorflow as tf
keras = tf.keras
Model = keras.Model
import keras_model as ke

im = cv2.imread('sample.png', cv2.IMREAD_UNCHANGED)
alpha = im[:, :, 3]
cdata = im[:, :, :3]
cdata = cv2.cvtColor(cdata, cv2.COLOR_BGR2GRAY)
margin_1 = np.sum(alpha, axis=0)
margin_2 = np.sum(alpha, axis=1)
top = np.nonzero(margin_2)[0][0]
bottom = np.nonzero(margin_2)[0][-1] +1
left = np.nonzero(margin_1)[0][0]
right = np.nonzero(margin_1)[0][-1] +1

width = right - left
height = bottom - top
alpha = alpha[top:bottom, left:right]
cdata = cdata[top:bottom, left:right]


target_window = 190
margin = 30
if height > width:
#	scaler = (target_window, int(float(width * height)/target_window))
        scaler = (int(float(width * target_window) / height), target_window)
        print scaler, height, width
else:
#	scaler = (int(float(width * height)/target_window), target_window)
        scaler = (target_window, int(float(height * target_window)/ width))

depth_im = cv2.resize(cdata, scaler)
print depth_im.shape
depth_mask = cv2.resize(alpha, scaler) == 255
average_depth = np.mean(depth_im[depth_mask])
depth_im[depth_mask] = np.uint8(np.int64(depth_im[depth_mask])) + np.int64(55 - average_depth)
depth_im[~depth_mask] = 255
depth_im = depth_im.astype(float)/255

height, width = depth_im.shape[0], depth_im.shape[1]


final_depth_im = np.ones((target_window + 2 * margin, target_window + 2 * margin), dtype=float)
final_depth_mask = np.zeros((target_window + 2 * margin, target_window + 2 * margin), dtype=bool)
loc_x = int(np.floor((target_window + 2 * margin - width)/2) + 1)
loc_y = int(np.floor((target_window + 2 * margin - height)/2) + 1)

final_depth_im[loc_y: (loc_y + height), loc_x: (loc_x + width)] = depth_im
final_depth_mask[loc_y: (loc_y + height) , loc_x: (loc_x + width)] = depth_mask * 255

final_depth_im = np.float32(final_depth_im)
model = ke.loadNetwork()
final_depth_im = np.reshape( np.float32(final_depth_im), [1, 250, 250,1])
out = model.predict(final_depth_im)
print out.shape
print(np.sum(out))

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('upsample').output)
conv = intermediate_layer_model.predict(final_depth_im)


#conv = model.layers[1].get_weights()[1]

np.set_printoptions(precision=8)
np.set_printoptions(threshold=sys.maxsize)
#print(conv)
print (conv.shape)
print(conv[0][13][0])
#print(np.unique(conv[np.nonzero(conv)]))


#print(np.sum(conv))
print(np.sum(conv))
#print(conv[0,0:1,:])
#net = cv2.dnn.readNetFromCaffe("deploy_example.prototxt", "hardpose_69k.caffemodel")
#blob = cv2.dnn.blobFromImage(final_depth_im, 1, (250, 250), (), True, False)
#print blob.shape
#net.setInput(blob)
#out = net.forward()


classes =  np.argmax(out, axis=-1)
classes = np.reshape(classes, (250, 250))
classes[~final_depth_mask] = 0
x,y = np.nonzero(classes)
classes = np.uint8(classes)
classes = classes * 5
imagesc = cv2.applyColorMap(classes, cv2.COLORMAP_HSV)
#cv2.imshow('sample', imagesc)
#cv2.waitKey()