import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


def unet(input_shape, out_channel=4, kernel=3, pool_size=(2, 2), feature_base=32):
    input_holder = layers.Input(shape=input_shape)

    n, h, w, c = input_holder.shape
    h_pad = 32 - h % 32 if not h % 32 == 0 else 0
    w_pad = 32 - w % 32 if not w % 32 == 0 else 0
    padded_image = tf.pad(input_holder, [[0, 0], [0, h_pad], [0, w_pad], [0, 0]], "reflect")

    conv1_1 = layers.Conv2D(feature_base, (kernel, kernel), padding="same")(padded_image)
    conv1_1 = layers.LeakyReLU(0.2)(conv1_1)
    conv1_2 = layers.Conv2D(feature_base, (kernel, kernel), padding="same")(conv1_1)
    conv1_2 = layers.LeakyReLU(0.2)(conv1_2)

    pool_1 = layers.MaxPool2D(pool_size)(conv1_2)

    conv2_1 = layers.Conv2D(feature_base * 2, (kernel, kernel), padding="same")(pool_1)
    conv2_1 = layers.LeakyReLU(0.2)(conv2_1)
    conv2_2 = layers.Conv2D(feature_base * 2, (kernel, kernel), padding="same")(conv2_1)
    conv2_2 = layers.LeakyReLU(0.2)(conv2_2)

    pool_2 = layers.MaxPool2D(pool_size)(conv2_2)

    conv3_1 = layers.Conv2D(feature_base * 4, (kernel, kernel), padding="same")(pool_2)
    conv3_1 = layers.LeakyReLU(0.2)(conv3_1)
    conv3_2 = layers.Conv2D(feature_base * 4, (kernel, kernel), padding="same")(conv3_1)
    conv3_2 = layers.LeakyReLU(0.2)(conv3_2)

    pool_3 = layers.MaxPool2D(pool_size)(conv3_2)

    conv4_1 = layers.Conv2D(feature_base * 8, (kernel, kernel), padding="same")(pool_3)
    conv4_1 = layers.LeakyReLU(0.2)(conv4_1)
    conv4_2 = layers.Conv2D(feature_base * 8, (kernel, kernel), padding="same")(conv4_1)
    conv4_2 = layers.LeakyReLU(0.2)(conv4_2)

    pool_4 = layers.MaxPool2D(pool_size)(conv4_2)

    conv5_1 = layers.Conv2D(feature_base * 16, (kernel, kernel), padding="same")(pool_4)
    conv5_1 = layers.LeakyReLU(0.2)(conv5_1)
    conv5_2 = layers.Conv2D(feature_base * 16, (kernel, kernel), padding="same")(conv5_1)
    conv5_2 = layers.LeakyReLU(0.2)(conv5_2)

    unpool1 = layers.Conv2DTranspose(feature_base * 8, pool_size, (2, 2), "same")(conv5_2)
    concat1 = layers.Concatenate()([unpool1, conv4_2])
    conv6_1 = layers.Conv2D(feature_base * 8, (kernel, kernel), padding="same")(concat1)
    conv6_1 = layers.LeakyReLU(0.2)(conv6_1)
    conv6_2 = layers.Conv2D(feature_base * 8, (kernel, kernel), padding="same")(conv6_1)
    conv6_2 = layers.LeakyReLU(0.2)(conv6_2)

    unpool2 = layers.Conv2DTranspose(feature_base * 4, pool_size, (2, 2), "same")(conv6_2)
    concat2 = layers.Concatenate()([unpool2, conv3_2])
    conv7_1 = layers.Conv2D(feature_base * 4, (kernel, kernel), padding="same")(concat2)
    conv7_1 = layers.LeakyReLU(0.2)(conv7_1)
    conv7_2 = layers.Conv2D(feature_base * 4, (kernel, kernel), padding="same")(conv7_1)
    conv7_2 = layers.LeakyReLU(0.2)(conv7_2)

    unpool3 = layers.Conv2DTranspose(feature_base * 2, pool_size, (2, 2), "same")(conv7_2)
    concat3 = layers.Concatenate()([unpool3, conv2_2])
    conv8_1 = layers.Conv2D(feature_base * 2, (kernel, kernel), padding="same")(concat3)
    conv8_1 = layers.LeakyReLU(0.2)(conv8_1)
    conv8_2 = layers.Conv2D(feature_base * 2, (kernel, kernel), padding="same")(conv8_1)
    conv8_2 = layers.LeakyReLU(0.2)(conv8_2)

    unpool4 = layers.Conv2DTranspose(feature_base, pool_size, (2, 2), "same")(conv8_2)
    concat4 = layers.Concatenate()([unpool4, conv1_2])
    conv9_1 = layers.Conv2D(feature_base, (kernel, kernel), padding="same")(concat4)
    conv9_1 = layers.LeakyReLU(0.2)(conv9_1)
    conv9_2 = layers.Conv2D(feature_base, (kernel, kernel), padding="same")(conv9_1)
    conv9_2 = layers.LeakyReLU(0.2)(conv9_2)

    out = layers.Conv2D(out_channel, (1, 1), padding="same")(conv9_2)
    out_holder = out[:, :h, :w, :]

    net_model = keras.Model(inputs=input_holder, outputs=out_holder)
    return net_model


if __name__ == "__main__":
    test_input = tf.convert_to_tensor(np.random.randn(1, 512, 512, 4))
    net = unet((512, 512, 4))
    net.summary()
    output = net(test_input, training=False)
    print("test over")
