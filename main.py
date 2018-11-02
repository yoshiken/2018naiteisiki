# coding: utf-8
import boto3
import json
import cv2

from flask import Flask, render_template

app = Flask(__name__)
bucket = 'yoshiken.8122.jp'
faceclient = boto3.client('rekognition')
s3client = boto3.resource('s3')
photobase = 'rekognition/group'

groupcount = 4
groupname = ['あか', 'うすもも', 'オレンジ', 'うすだいだい', 'きいろ', 'きみどり', 'あお', 'ふじ', 'みず', 'むらさき', 'みどり', 'ピンク']


@app.route('/')
def index():
    gets3img()
    faceDetails = faceSearch()
    return render_template('index.html', faceDetails=faceDetails)


def faceSearch():
    result = [{} for i in range(groupcount)]
    for index in range(groupcount):
        groupno = str(index + 1)
        print(groupno)
        result[index]['groupname'] = groupname[index]
        photo = photobase + groupno + '.jpg'
        result[index]['photoname'] = './img/group' + groupno + '.jpg'
        # rekognitionのスコアを取得
        response = faceclient.detect_faces(Image={'S3Object': {'Bucket': bucket, 'Name': photo}}, Attributes=['ALL'])

        # 使う変数初期化
        scores = []
        totalscore = 0
        humancount = len(response['FaceDetails'])
        result[index]['BoundingBox'] = [[]for l in range(humancount)]
        # print(json.dumps(response, indent=4, sort_keys=True))
        for indextmp, faceDetail in enumerate(response['FaceDetails']):
            result[index]['BoundingBox'][indextmp] = faceDetail['BoundingBox']
            # print(json.dumps(faceDetail, indent=4, sort_keys=True))
            hoge = faceDetail['Emotions']
            for emotion in hoge:
                if emotion['Type'] == 'HAPPY':
                    print(emotion['Confidence'])
                    scores += [emotion['Confidence']]
        for score in scores:
            totalscore += score
        result[index]['avgscore'] = round(totalscore / len(scores), 5)
    return result


def gets3img():
    bucketimg = s3client.Bucket(bucket)
    for groupno in range(1, groupcount+1):
        photo = photobase + str(groupno) + '.jpg'
        print(photo)
        photodownloaddir = 'static/img/group' + str(groupno) + '.jpg'
        bucketimg.download_file(photo, photodownloaddir)
        # im = cv2.imread('static/img/df5d37.jpg')
        # h, w, _ = im.shape

if __name__ == '__main__':
    app.run()
