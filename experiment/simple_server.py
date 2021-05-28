import os
import time

from flask import Flask, jsonify, request, redirect, url_for, send_file, make_response
from flask_cors import CORS, cross_origin

# Config
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'upload')
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])


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
    if not allowed_file(image.filename):
        return error('Only supported %s' % app.config['ALLOWED_EXTENSIONS']), 415

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    return send_file(image_path)


if __name__ == '__main__':
    app.run(debug=True)
