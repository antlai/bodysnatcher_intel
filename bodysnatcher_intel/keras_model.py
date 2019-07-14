import tensorflow as tf
try:
    import tensorflow.contrib.tensorrt as trt
except:
    print ("No TensorRT")


WEIGHTS_FILE = './weights.h5'

keras = tf.keras
K = keras.backend
layers = keras.layers
Input = keras.Input
Model = keras.Model
plot_model = keras.utils.plot_model

def getModel():
    imageIn = Input(shape=(250, 250, 1), name="imageIn")
    pad = layers.ZeroPadding2D(padding=(2,2))(imageIn)
    conv1 = layers.Conv2D(64, 5, strides=2, padding='valid',
                          activation='relu', name='conv1')(pad)
    pool1 = layers.MaxPooling2D(3, strides=3, padding='same',
                                name='pool1')(conv1)
    conv2 = layers.Conv2D(128, 3, strides=1, padding='valid', activation='relu',
                          name='conv2')(pool1)
    pool2 = layers.MaxPooling2D(2, strides=2, padding='valid')(conv2)
    conv3 = layers.Conv2D(256, 5, strides=1, padding='valid', activation='relu',
                          name='conv3')(pool2)
    conv4 = layers.Conv2D(46, 3, strides=1, padding='valid',
                          name='conv4-class')(conv3)
    deconv1 = layers.Conv2DTranspose(46, 3, strides=1, padding='valid',
                                     use_bias=True, #output_padding=0,
                                     name='upscore')(conv4)
    scorePool2 = layers.Conv2D(46, 1, strides=1, padding='valid',
                               name='score-pool2')(pool2)
    scorePool2c = layers.Cropping2D(cropping=((0, 4), (0, 4)))(scorePool2)
    scoreFuse = layers.Add()([scorePool2c, deconv1])
    drop1 = layers.Dropout(0.5)(scoreFuse)
    deconv2 = layers.Conv2DTranspose(46, 4, strides=2, padding='valid',
                                     use_bias=True, #output_padding=0,
                                     name='upsample-fused-16')(drop1)
    scorePool1 = layers.Conv2D(46, 1, strides=1, padding='valid',
                               name='score-pool1')(pool1)
    scorePool1c = layers.Cropping2D(name='score-pool1c',
                                    cropping=((0, 8), (0, 8)))(scorePool1)
    scoreFuse1 = layers.Add(name='score-final')([scorePool1c,  deconv2])
    drop2 = layers.Dropout(0.5)(scoreFuse1)
    deconv3 = layers.Conv2DTranspose(46, 19, strides=7, use_bias=True,
                                     padding='valid', name='upsample')(drop2)
#    reshape = layers.Reshape((250*250, 46))(deconv3)

    argmax = layers.Lambda(lambda x: K.argmax(x, -1))(deconv3)
    #argmax = deconv3 #delete


#    softmax = layers.Activation('softmax')(reshape)

    model= Model(imageIn, argmax)
    return model

#model.summary()
#plot_model(model,show_shapes=True,to_file='/tmp/model.png')

def loadNetwork() :
#    K.set_floatx('float16') slower on TX2...
    K.clear_session()
    config = tf.ConfigProto()
    #avoid allocating all the memory
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    K.set_learning_phase(0)
    model = getModel()
    model.load_weights(WEIGHTS_FILE, by_name=True)
    return model

def getFrozenGraph() :
    K.clear_session()
    K.set_learning_phase(0)
    model = loadNetwork()
    session = K.get_session()
    graph = session.graph;
    output = [x.op.name for x in model.outputs]
    with graph.as_default():
        g = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        outG = tf.graph_util.convert_variables_to_constants(session, g, output)
        return (outG, model)

def getRTGraph() :
    # Unfortunately it needs a post feb/2019 version of TensorRT to optimize
    # Convolution2DTranspose, i.e., where we spend most of the time...

    g, model = getFrozenGraph()
    output = [x.op.name for x in model.outputs]
    newG = trt.create_inference_graph(input_graph_def=g, outputs=output,
                                      max_batch_size=1,
                                      max_workspace_size_bytes=400000000,
                                      precision_mode='FP16')
    return newG, model
