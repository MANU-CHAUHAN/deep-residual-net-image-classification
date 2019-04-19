import keras
from keras.models import Model
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.layers import BatchNormalization, Activation, Conv2D, Input, MaxPooling2D, GlobalAveragePooling2D
import keras.backend as K
from keras.layers.merge import add

if K.image_dim_ordering() == "tf":
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def get_bn_relu(input):
    """Utility function to help in creating BN -> Relu block"""
    x = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation('relu')(x)


def get_conv_bn_relu(**params):
    """Helper function to get_ResNet_model CONV -> BN -> Relu block"""

    filters= params["filters"]
    kernel_size = params["kernel_size"]
    strides = params.setdefault("strides", (1,1))
    padding = params.setdefault("padding", "same")
    kernel_initializer = params.setdefault("kernel_initializer", "he_normal")
    kernel_regulizer = params.setdefault("kernel_regulizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regulizer)(input)

        return get_bn_relu(conv)
    return f


def get_bn_relu_conv(**params):
    """Helper function to get_ResNet_model BN -> Relu -> Conv. An improved version over normal Conv -> BN -> Relu"""

    filters = params["filters"]
    kernel_size = params["kernel_size"]
    strides = params.setdefault("strides", (1, 1))
    padding = params.setdefault("padding", "same")
    kernel_initializer = params.setdefault("kernel_initializer", "he_normal")
    kernel_regulizer = params.setdefault("kernel_regulizer", l2(1.e-4))

    def f(input):
        x = get_bn_relu(input)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regulizer)(x)
        return conv
    return f


def handle_shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input

    if input_shape[1] is not None and input_shape[2] is not None:
        stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                       kernel_size=(1, 1),
                                       strides=(stride_width, stride_height),
                                       padding="valid",
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=l2(0.0001))(input)

    else:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                   kernel_size=(1, 1),
                                   strides=(1,1),
                                   padding="valid",
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def get_residual_block(filters, repetitions, function_to_use, is_first_layer=False):
    """Helps to get_ResNet_model a residual block based on the function passed (basic 3x3 -> 3x3
    or bottleneck_block having 1x1 -> 3x3 -> 1x1)"""

    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = function_to_use(filters=filters, strides=strides,
                                    is_first_block_and_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f


def basic_block(filters, strides=(1, 1), is_first_block_and_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(input):
        # to avoid doing BN -> Relu again after BN -> Relu -> Maxpool
        if is_first_block_and_first_layer:
            conv = Conv2D(filters=filters, strides=strides,
                          kernel_size=(3, 3),
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)
        else:
            conv = get_bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=strides)(input)

        residual = get_bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=strides)(conv)

        return handle_shortcut(input=input, residual=residual)
    return f


def bottleneck_block(filters, strides=(1, 1), is_first_block_and_first_layer=False):
    """bottleneck_block architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf"""

    def f(input):
        if is_first_block_and_first_layer:
            # avoid BN -> Relu -> Conv again after BN -> Relu -> Maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=strides,
                                  padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = get_bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                  strides=strides)(input)

        conv_3_3 = get_bn_relu_conv(filters=filters, strides=(1, 1), kernel_size=(3, 3))(conv_1_1)

        residual = get_bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)

        return handle_shortcut(input, residual)
    return f


class ResNet_get_ResNet_modeler:
    @staticmethod
    def get_ResNet_model(input_shape, num_outputs, block_function, repetitions):
        """
           input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
           num_outputs: The number of outputs at final softmax layer
           block_function: The block function to use. This is either `basic_block` or `bottleneck_block`.
    
           repetitions: Number of repetitions of various block units.
                    At each block unit, the number of filters are doubled and the input size is halved
    
           Returns:
                The keras `Model`
                """
        input = Input(shape=input_shape)
    
        conv_1 = get_conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        block = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv_1)
    
        filters = 64
        for i, r in enumerate(repetitions):
            block = get_residual_block(filters=filters,repetitions=r,function_to_use=block_function,is_first_layer=(i == 0))(block)
            filters *= 2
    
        # last bn -> relu
        block = get_bn_relu(block)
    
        block = Conv2D(filters=num_outputs, kernel_size=(1, 1))(block)
    
        block = GlobalAveragePooling2D()(block)
    
        output = Activation('softmax')(block)
    
        model = Model(inputs=input, outputs=output)
        
        return model

    @staticmethod
    def get_ResNet_model_resnet_18(input_shape, num_outputs):
        return ResNet_get_ResNet_modeler.get_ResNet_model(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def get_ResNet_model_resnet_34(input_shape, num_outputs):
        return ResNet_get_ResNet_modeler.get_ResNet_model(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def get_ResNet_model_resnet_50(input_shape, num_outputs):
        return ResNet_get_ResNet_modeler.get_ResNet_model(input_shape, num_outputs, bottleneck_block, [3, 4, 6, 3])

    @staticmethod
    def get_ResNet_model_resnet_101(input_shape, num_outputs):
        return ResNet_get_ResNet_modeler.get_ResNet_model(input_shape, num_outputs, bottleneck_block, [3, 4, 23, 3])

    @staticmethod
    def get_ResNet_model_resnet_152(input_shape, num_outputs):
        return ResNet_get_ResNet_modeler.get_ResNet_model(input_shape, num_outputs, bottleneck_block, [3, 8, 36, 3])


# get data
(x_train, y_train), (x_test, y_test) = get_data()

# get model
model = ResNet_get_ResNet_modeler().get_ResNet_model_resnet_18(input_shape=(32, 32, 3), num_outputs=10)

print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))



    
    
