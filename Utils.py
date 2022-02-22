import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

# save images in a nxn grid
def grid_save(images, save_dir, t_id, e, grid_dim, subtext='peep_'):
    rows = cols = grid_dim
    assert len(images) == rows * cols

    for i in range(cols * rows):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(images[i])

    # save fig
    fig_name = subtext + str(t_id) + '_' + str(e) + '.png'
    fig_path = save_dir + fig_name
    plt.savefig(fig_path)
    print("generated image saved {}".format(fig_path))
    # clear buffer
    plt.clf()

def plot_save_loss(dis_loss, gen_loss, save_dir, t_id):
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(dis_loss, color='r', label='d_loss')
    plt.plot(gen_loss, color='g', label='g_loss')
    plt.legend(loc="upper left")
    # save fig
    fig_name = str(t_id) + '.png'
    fig_path = save_dir + fig_name
    plt.savefig(fig_path)
    plt.show()

# clear previous saved outputs
def clear_history(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

