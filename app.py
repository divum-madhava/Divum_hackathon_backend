# Import all the necessary libraries
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from flask import Flask, request, jsonify

from fer import Video
from fer import FER
from decouple import config
from flask_cors import CORS, cross_origin


app = Flask(__name__)

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'application/json'


@app.route('/video_analysis', methods=['POST'])
def video_analysis():
    video_file = request.files['file']
    raw = video_file.read()
    filename = video_file.filename
    with open(filename, 'wb') as f:
        f.write(raw)
    face_detector = FER(mtcnn=True)
    input_video = Video(filename)
    processing_data = input_video.analyze(face_detector, display=False, save_frames=False, save_video=False,
                                          annotate_frames=False, zip_images=False)
    vid_df = input_video.to_pandas(processing_data)
    vid_df = input_video.get_first_face(vid_df)
    vid_df = input_video.get_emotions(vid_df)
    pltfig = vid_df.plot(figsize=(12, 6), fontsize=12).get_figure()
    plt.legend(fontsize='large', loc=1)
    path = './' + filename + 'emotion.png'
    pltfig.savefig(path)
    '''Push image to s3 and return the url'''
    url=upload_to_s3(filename,path=path)
    os.remove(filename)
    os.remove(path)
    try:
        os.remove('./data.csv')
    except:
        pass
    return jsonify({
        "status": 200,
        "report_graph": url
    })


@app.route('/image_analysis', methods=['POST'])
def image_analysis():
    image_file = request.files['file']
    raw = image_file.read()
    filename = image_file.filename
    url=upload_to_s3(filename,obj=raw)
    with open(filename, 'wb') as f:
        f.write(raw)
    img = cv2.imread(filename)
    detector = FER()
    emotions_list = detector.detect_emotions(img)
    emotion, score = detector.top_emotion(img)
    os.remove(filename)
    return jsonify({
        "status": 200,
        'emotions_list': json.loads(json.dumps(emotions_list,
                                               cls=NumpyEncoder)),
        'average_emotion_type': emotion,
        'average_score': score,
        'url':url
    })


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def upload_to_s3(name,path=None,obj=None):
    import boto3
    client = boto3.client('s3', aws_access_key_id=config('aws_access_key_id'),
                          aws_secret_access_key=config('aws_secret_access_key'))
    if path:
        print(path,"PP")
        with open(path, 'rb') as f:
            data = f.read()
            response = client.put_object(Body=data, Bucket='my-first-public-bucket-test', Key=name,
                                         ContentType='image/png')
    else:
        response = client.put_object(Body=obj, Bucket='my-first-public-bucket-test', Key=name,
                                     ContentType='image/png')

    print('https://my-first-public-bucket-test.s3.ap-south-1.amazonaws.com/'+name)

    return 'https://my-first-public-bucket-test.s3.ap-south-1.amazonaws.com/'+name


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
