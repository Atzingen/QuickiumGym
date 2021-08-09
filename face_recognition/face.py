from face_identification import get_people_scores
from keras_vggface.vggface import VGGFace
import cv2
from flask import Flask, request
import numpy as np
import json

app = Flask(__name__)

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg')


@app.route("/", methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        info = request.form['photo']
        img = np.array(json.loads(info))
        cv2.imwrite('tst.jpg', img)
        name = get_people_scores(img, 'true_faces',
                                 model,
                                 just_max=False,
                                 threshold=0.0)
        print(name)                    
        return name

    return "Hello World!"

if __name__ == "__main__":
    app.run(debug=True, port=5007, host='0.0.0.0')
