import os
from predict import TRANSF
import glob
import numpy as np
import torch
import json
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(full_path)

            return redirect(url_for('predict',path=filename))

    return '''
    <!doctype html>
    <head>
        <link rel="icon" type="image/png" href="static/favicon.png"/>
        <style>

            body {background: linear-gradient(to right, #29323c, #485563);} 
            h1 {text-align: center; font-family: 'Ubuntu', sans-serif; color: white;}
            h2 {text-align: center; font-family: 'Ubuntu', sans-serif; color: white;}
            form {text-align: center; color: white;}
            .rcorners1 {margin: auto; max-width: 40%; border: 3px solid green; padding: 10px; text-align: left; color: white; font-family: 'Roboto', sans-serif;}
            .flowerimage {display: block; margin-left: auto; margin-right: auto; max-width: 20%; height:auto; margin-bottom: 25px}

        </style>
        <title>Flower-Torch</title>
    </head>
    <body>
        <h1>Flower-Torch</h1>
        <img src="static/flower-torch.png" class="flowerimage">
        <div class="rcorners1">
            Esse é o Flower-Torch, um modelo de inteligência artificial criado com PyTorch para identificar até 5 espécies diferentes de flores.<br><br>O modelo consegui distinguir entre 
            Margaridas, Dentes-de-leão, Rosas, Girassois e Tulipas.<br><br>Mande uma foto usando o formulário abaixo.
        </div>
        <h2>Upload do Arquivo</h2>
        <form method=post enctype=multipart/form-data id="form_file">
            <input type=file name=file>
            <input type=submit value=Upload>
        </form>
    </body>
    '''

@app.route('/predict/<path>')
def predict(path):

    image_tensor = TRANSF.transform_image('static/'+path)
    model = torch.load(glob.glob('TrainModel/models/*')[0],map_location=torch.device('cpu'))
    
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    classes = ['Margarida', 'Dente-de-leão', 'Rosa', 'Girassol', 'Tulipas']

    result = str(classes[predicted])

    # return json.dumps(dict(result=result))

    pag =f'''
        <!DOCTYPE html>
        <head>
            <title>Result</title>
            <link type="text/css" rel="stylesheet" href={url_for("static", filename="stylesheets/pred_styles.css")}/>
        </head>
        <body>
            <div class="result_image">
                <img src={url_for('static', filename=path)}>
            </div>
            <div class="result_class">
                <h2>{result}</h2>
            </div>
        </body>
        </html>
        '''

    return pag

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")