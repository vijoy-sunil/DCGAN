import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model

class Discriminator:
    def __init__(self, latent_dim, image_shape, learning_rate):
        self.latent_dim = latent_dim
        # real/fake image shape
        self.image_shape = image_shape
        # get model, optimizers, etc.
        self.model = self.create_model()
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def create_model(self):
        input_0 = layers.Input(shape=self.image_shape)
        conv_0 = layers.Conv2D(self.image_shape[0],
                               kernel_size=4,
                               strides=2,
                               padding='same')(input_0)
        conv_0 = layers.LeakyReLU(alpha=0.2)(conv_0)

        conv_1 = layers.Conv2D(self.image_shape[0] * 2,
                               kernel_size=4,
                               strides=2,
                               padding='same')(conv_0)
        conv_1 = layers.LeakyReLU(alpha=0.2)(conv_1)

        conv_2 = layers.Conv2D(self.image_shape[0] * 4,
                               kernel_size=4,
                               strides=2,
                               padding='same')(conv_1)
        conv_2 = layers.LeakyReLU(alpha=0.2)(conv_2)

        hidden_0 = layers.Flatten()(conv_2)
        hidden_0 = layers.Dropout(0.2)(hidden_0)
        hidden_0 = layers.Dense(1, activation='sigmoid')(hidden_0)

        model = tf.keras.Model(input_0, hidden_0, name='discriminator')
        model.summary()
        return model

    def save_weights(self, save_dir, train_id, e):
        weights_file_name = 'discriminator_weights_' + str(train_id) + '_' + str(e) + '.h5'
        weights_file_path = save_dir + weights_file_name
        self.model.save(weights_file_path)
        print("weights saved {}".format(weights_file_path))

    # load model weights, NOTE: we are not saving model architecture here
    def load_weights(self, save_dir, train_id, e):
        weights_file_name = 'discriminator_weights_' + str(train_id) + '_' + str(e) + '.h5'
        weights_file_path = save_dir + weights_file_name
        self.model = load_model(weights_file_path)
