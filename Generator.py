import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model

class Generator:
    def __init__(self, latent_dim, learning_rate):
        self.latent_dim = latent_dim
        # get model, optimizers, etc.
        self.model = self.create_model()
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def create_model(self):
        input_0 = layers.Input(shape=self.latent_dim,)
        hidden_0 = layers.Dense(8 * 8 * self.latent_dim)(input_0)
        hidden_0 = layers.Reshape((8, 8, self.latent_dim))(hidden_0)

        conv_0 = layers.Conv2DTranspose(self.latent_dim * 2,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same')(hidden_0)
        conv_0 = layers.LeakyReLU(alpha=0.2)(conv_0)

        conv_1 = layers.Conv2DTranspose(self.latent_dim * 2,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same')(conv_0)
        conv_1 = layers.LeakyReLU(alpha=0.2)(conv_1)

        conv_2 = layers.Conv2D(3,
                               kernel_size=5,
                               padding='same')(conv_1)

        conv_2 = layers.Activation('sigmoid')(conv_2)
        model = tf.keras.Model(input_0, conv_2, name='generator')
        model.summary()
        return model

    def save_weights(self, save_dir, train_id, e):
        weights_file_name = 'generator_weights_' + str(train_id) + '_' + str(e) + '.h5'
        weights_file_path = save_dir + weights_file_name
        self.model.save(weights_file_path)
        print("weights saved {}".format(weights_file_path))

    # load model weights, NOTE: we are not saving model architecture here
    def load_weights(self, save_dir, train_id, e):
        weights_file_name = 'generator_weights_' + str(train_id) + '_' + str(e) + '.h5'
        weights_file_path = save_dir + weights_file_name
        self.model = load_model(weights_file_path)
