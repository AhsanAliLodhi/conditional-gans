import requests
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from bunch import Bunch
from stargan.main import get_solver,config
from PIL import Image
import base64
import io
import numpy as np

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def imageToString(image):
    print(image.mode)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

app = Flask(__name__,static_folder='templates')
config = vars(config)

config["mode"] =  "test"
config["dataset"] =  "RaFD"
config["rafd_crop_size"] =  128
config["n_critic"] =  3
config["rafd_image_dir"] =  "D:/work/praktikum2/data/multicamera/train"
config["image_size"] =  128
config["c_dim"] =  6
config["sample_dir"] =  "D:/work/praktikum2/code/stargan/multicamera_rel_nocycle_calforboth/samples"
config["log_dir"] =  "D:/work/praktikum2/code/stargan/multicamera_rel_nocycle_calforboth/logs"
config["model_save_dir"] =  "D:/work/praktikum2/code/stargan/multicamera_rel_nocycle_calforboth/models"
config["result_dir"] =  "D:/work/praktikum2/code/stargan/multicamera_rel_nocycle_calforboth/results"
config = Bunch(config)
print(config.log_dir)

solver = get_solver(config)
solver.restore_model(160000)

@app.route('/', methods = ['GET', 'POST'])
def hello():
    print("inside hello")
    if request.method == 'GET':
        message = "Hello"
        return render_template('/index.html', message=message)
    if request.method == 'POST':

        return ""

@app.route('/shift', methods = ['GET', 'POST'])
def shift():
    print("inside shift")
    if request.method == 'GET':
        return "In shift"
    if request.method == 'POST':
        image = request.form.get('image')
        shift_by = request.form.get('shift_by')
        image = image.split(',')
        base = image[0]
        image = stringToImage(image[1])
        result = solver.pass_from_generator(image,float(shift_by))
        #result.show()
        img_str = base+','+imageToString(result).decode("utf-8")
        print(base)
        return img_str

@app.route('/getcameraimage', methods = ['GET', 'POST'])
def getcameraimage():
    print("inside getcameraimage")
    if request.method == 'POST':
        camera = request.form.get('camera')
        image = request.form.get('image')
        print(camera,image)
        try:
            path = config["rafd_image_dir"].replace("train","test")+"/"+camera+"/"+image
            print(path)
            im = Image.open(path)
        except Exception as e:
            return e
        result = imageToString(im).decode("utf-8")
        return "data:image/png;base64,"+result

@app.route('/files/<path:filename>')
def download_file(filename):
    if request.method == 'GET':
        return send_from_directory(app.static_folder,filename)


if __name__ == "__main__":
    app.run(debug=True, port=1234) #host="0.0.0.0" for docker