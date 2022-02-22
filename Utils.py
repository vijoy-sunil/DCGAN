import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

# save images in a nxn grid, imgs is a list of PIL images
# why not use matplotlib to save this? Matplotlib expects the data
# to be either in the range [0,1] if it's float or [0,255] if it's
# an integer. We would need to do clipping to deal with our generated
# batch of images if we were using matplotlib
def save_generated_images(images, save_dir, t_id, e, grid_size):
    rows = cols = grid_size
    assert len(images) == rows * cols

    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    fig_name = 'peep_' + str(t_id) + '_' + str(e) + '.png'
    fig_path = save_dir + fig_name
    grid.save(fig_path)
    print("generated image saved {}".format(fig_path))

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

