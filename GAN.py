import os
# silence tensorflow messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
import Utils
from zipfile import ZipFile

class GAN(keras.Model):
    def __init__(self, d_model, g_model, batch_size, latent_dim, clear_history=True):
        super(GAN, self).__init__()
        self.g_loss_metric = None
        self.d_loss_metric = None
        self.loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.dataset_batch_gen = None
        self.latent_dim = latent_dim

        # models
        self.d_model = d_model
        self.g_model = g_model
        self.loss = tf.keras.losses.BinaryCrossentropy()

        # params
        self.batch_size = batch_size

        # directories
        self.save_dir = './SavedFiles/'
        self.weights_dir = self.save_dir + 'Weights/'
        self.peep_dir = self.save_dir + 'Peep/'
        self.data_dir = '../anime/archive.zip'
        self.unzip_dir = '../anime/'
        self.dataset_dir = 'images/'

        # clear saved files
        if clear_history == True:
            Utils.clear_history(self.peep_dir)
            Utils.clear_history(self.weights_dir)

    # set optimizers, loss fn and metrics
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    # overwrite train_step in base class
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Decode them to fake images
        generated_images = self.g_model(random_latent_vectors)
        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.d_model(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.d_model.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.d_model.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.d_model(self.g_model(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.g_model.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.g_model.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def load_dataset(self, fast_load, dim):
        if fast_load == False:
            with ZipFile(self.data_dir, "r") as zipobj:
                zipobj.extractall(self.unzip_dir)
        # Create a dataset from our folder, and rescale the images
        # to the [0-1] range
        unzip_path = self.unzip_dir + self.dataset_dir
        dataset = keras.preprocessing.image_dataset_from_directory(
            unzip_path,
            label_mode=None,
            image_size=(dim, dim),
            batch_size=self.batch_size)
        dataset = dataset.map(lambda x: x / 255.0)
        self.dataset_batch_gen = dataset

# call back functions, called at the end of every epoch
# save gen and disc model weights
class SaveWeights(keras.callbacks.Callback):
    def __init__(self, d_class, g_class, save_dir, t_id):
        self.d_class = d_class
        self.g_class = g_class
        self.save_dir = save_dir
        self.t_id = t_id

    def on_epoch_end(self, epoch, logs=None):
        self.d_class.save_weights(self.save_dir, self.t_id, epoch)
        self.g_class.save_weights(self.save_dir, self.t_id, epoch)

# save generated images in a grid
class PeepGenerator(keras.callbacks.Callback):
    def __init__(self, dim, latent_dim, peep_dir, t_id):
        self.peep_dir = peep_dir
        self.dim = dim
        self.num_img = dim * dim
        self.latent_dim = latent_dim
        self.t_id = t_id

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.g_model(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        pil_images_batch = []
        for i in range(self.num_img):
            pil_img = keras.preprocessing.image.array_to_img(generated_images[i])
            pil_images_batch.append(pil_img)

        Utils.save_generated_images(pil_images_batch,
                                    self.peep_dir,
                                    self.t_id,
                                    epoch,
                                    self.dim)
