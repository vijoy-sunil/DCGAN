import GAN
import Discriminator
import Generator
import Utils

if __name__ == '__main__':
    # id your run
    train_id = 0

    # parameters
    latent_dim = 128
    image_shape = (32, 32, 3)
    gen_lr = 0.0001
    dis_lr = 0.0001
    batch_size = 32
    epochs = 10

    D = Discriminator.Discriminator(latent_dim, image_shape, dis_lr)
    G = Generator.Generator(latent_dim, gen_lr)

    Gan = GAN.GAN(D.model,  # discriminator model
                  G.model,  # generator model
                  batch_size,  # batch_size
                  latent_dim,  # noise dim
                  clear_history=True)
    # load dataset
    Gan.load_dataset(fast_load=True, dim=32)
    # compile model
    Gan.compile(D.optimizer,  # discriminator optimizer
                G.optimizer,  # generator optimizer
                Gan.loss)  # loss function

    # train
    history = Gan.fit(Gan.dataset_batch_gen,
                      epochs=epochs,
                      callbacks=[
                          GAN.PeepGenerator(grid_dim=5,
                                            latent_dim=latent_dim,
                                            peep_dir=Gan.peep_dir,
                                            t_id=train_id),

                          GAN.SaveWeights(D, G,
                                          save_dir=Gan.weights_dir,
                                          t_id=train_id)]
                      )
    Utils.plot_save_loss(history.history['d_loss'],
                         history.history['g_loss'],
                         Gan.save_dir,
                         t_id=train_id)
