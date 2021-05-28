import os

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util


if __name__ == "__main__":
    if not os.path.exists('results'):
        os.mkdir('results')

    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        # Forward
        model.test()

        # Convert image to numpy array
        fake = util.tensor2im(model.fake_B.data)
        # Save image
        img_path = model.get_image_paths()
        img_name = img_path[0].split('/')[-1]
        util.save_image(fake, 'results/{}'.format(img_name))
