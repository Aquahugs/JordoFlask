from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from models.models import create_model
import util.util as util

import os
import time

from flask import Flask, jsonify, request, redirect, url_for, send_file, make_response
from flask_cors import CORS, cross_origin

import base64
import datetime
import cv2

## Model options
from options.base_options import BaseOptions
from util.image_processing import get_face_position, resize, process_edge_image

class GenOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.isTrain = False

# Parse argument options
opt = GenOptions().parse()
# Some defaults values for generating purpose
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.use_dropout = True
opt.align_data = True
opt.model = 'one_direction_test'
opt.which_model_netG = 'unet_256'
opt.which_direction = 'AtoB'

# Load model
model = create_model(opt)

# Config
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'upload')
app.config['GAN_FOLDER'] = os.path.join(os.getcwd(), 'cgan')
#app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])


# Helpers
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def error(msg):
    return jsonify({'error': msg})

# Routers
@app.route('/')
def pong():
    return 'Pong', {'Content-Type': 'text-plain; charset=utf-8'}

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    image = request.files['file']
    #if not allowed_file(image.filename):
        #return error('Only supported %s' % app.config['ALLOWED_EXTENSIONS']), 415

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    return send_file(image_path)

@app.route('/gen_base64', methods=['POST'])
def gen_base64():
    if 'image' not in request.json:
        return error('Stupid request'), 412

    if 'base64' in request.args:
	    use_base64 = True if request.args.get('base64') == 'true' else False
    else:
        use_base64 = False

    image_data = str.encode(request.json['image'])
    image_data = image_data[23:]

    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    timestamp = datetime.datetime.now().isoformat()
    image_name = ip + '_' + timestamp + '.png'

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    with open(image_path, "wb") as f:
        f.write(base64.decodebytes(image_data))

    image = cv2.imread(image_path)
    image = resize(image, 256, 256)
    cv2.imwrite(image_path, image)

    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    # if not use_base64:
    #    return send_file(output_path)

    # image = open(output_path, 'rb').read()
    # encoded = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8')

    # return encoded
    return '/cgan/' + image_name

@app.route('/gen_photo', methods=['POST'])
def gen_photo():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    if 'base64' in request.args:
        use_base64 = True if request.args.get('base64') == 'true' else False
    else:
        use_base64 = False

    image = request.files['file']

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    ## Crop here
    found_face, position = get_face_position(image_path)
    if not found_face:
        return 'No face found', 404

    top = position['top']
    bottom = position['bottom']
    left = position['left']
    right = position['right']

    img = cv2.imread(image_path)
    img = img[top:bottom, left:right]
    img = resize(img, 256, 256)

    cv2.imwrite(image_path, img)

    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    # if not use_base64:
    #    return send_file(output_path)

    # image = open(output_path, 'rb').read()
    # encoded = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8')

    return '/cgan/' + image_name

@app.route('/gen_photo_nhat', methods=['POST'])
def gen_photo_nhat():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    if 'base64' in request.args:
        use_base64 = True if request.args.get('base64') == 'true' else False
    else:
        use_base64 = False

    image = request.files['file']

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    ## Crop here
    found_face, position = get_face_position(image_path)
    if not found_face:
        return 'No face found', 404

    top = position['top']
    bottom = position['bottom']
    left = position['left']
    right = position['right']

    img = cv2.imread(image_path)
    img = img[top:bottom, left:right]
    img = resize(img, 256, 256)

    cv2.imwrite(image_path, img)

    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    if not use_base64:
        return send_file(output_path)

    image = open(output_path, 'rb').read()
    encoded = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8')

    return encoded



@app.route('/gen_nhat', methods=['POST'])
def gen_nhat():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    if 'base64' in request.args:
	    use_base64 = True if request.args.get('base64') == 'true' else False
    else:
        use_base64 = False

    image = request.files['file']

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    print('hello', image_name)
    ## Crop here

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # image = image[image[:, :, 3] == 255,:3]
    # cv2.imwrite(image_path, image)
    # image = resize(image, 256, 256)
    # cv2.imwrite(image_path, image)


    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    if not use_base64:
        return send_file(output_path)

    image = open(output_path, 'rb').read()
    encoded = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8')

    return encoded

@app.route('/gen', methods=['POST'])
def gen():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    if 'base64' in request.args:
	    use_base64 = True if request.args.get('base64') == 'true' else False
    else:
        use_base64 = False

    image = request.files['file']

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    print('hello', image_name)
    ## Crop here

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # image = image[image[:, :, 3] == 255,:3]
    # cv2.imwrite(image_path, image)
    # image = resize(image, 256, 256)
    # cv2.imwrite(image_path, image)


    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    # if not use_base64:
    #    return send_file(output_path)

    # image = open(output_path, 'rb').read()
    # encoded = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8')

    # return encoded
    print('hi', image_name)
    return '/cgan/' + image_name

@app.route('/cgan/<image>')
def cgan(image):
    image_path = os.path.join(app.config['GAN_FOLDER'], image)
    return send_file(image_path)

@app.route('/upload/<image>')
def u(image):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
    return send_file(image_path)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
