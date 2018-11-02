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
groupname = ['1班', '2班', '3班', '4班', '5班', '6班', '7班', '8班', '9班', '10班', '11班', '12班']
resizesize = 600


@app.route('/')
def index():
    gets3img()
    faceDetails = faceSearch()
    return render_template('index.html', faceDetails=faceDetails)


def faceSearch():
    result = [{} for i in range(groupcount)]

    # チームごとのループ処理
    for index in range(groupcount):

        # 使う変数初期化
        scores = []
        totalscore = 0
        groupno = str(index + 1)
        photo = photobase + groupno + '.jpg'

        # この時点で決まっている結果を代入
        result[index]['groupname'] = groupname[index]
        result[index]['photoname'] = './img/group' + groupno + 'resize.jpg'

        #print(groupno)

        # rekognitionのスコアを取得
        response = faceclient.detect_faces(Image={'S3Object': {'Bucket': bucket, 'Name': photo}}, Attributes=['ALL'])
        print(json.dumps(response, indent=4, sort_keys=True))

        # rekognitionのスコアで動的に変わる変数
        humancount = len(response['FaceDetails'])
        result[index]['BoundingBox'] = [[]for l in range(humancount)]

        # 顔ごとのループ処理
        for indextmp, faceDetail in enumerate(response['FaceDetails']):
            result[index]['BoundingBox'][indextmp] = faceDetail['BoundingBox']
            # print(json.dumps(faceDetail, indent=4, sort_keys=True))
            hoge = faceDetail['Emotions']

            # 幸福度を取得
            for emotion in hoge:
                if emotion['Type'] == 'HAPPY':
                    scores += [emotion['Confidence']]

        # すべての顔の幸福度の平均を算出
        for score in scores:
            totalscore += score
        result[index]['avgscore'] = round(totalscore / len(scores), 5)

        result[index]['imgmeta'] = imgmeteinfo(groupno + 'resize.jpg')
    return result


def gets3img():
    bucketimg = s3client.Bucket(bucket)
    for groupno in range(1, groupcount+1):
        photo = photobase + str(groupno) + '.jpg'
        photodownloaddir = 'static/img/group' + str(groupno) + '.jpg'
        bucketimg.download_file(photo, photodownloaddir)
        resizeimg(str(groupno))


def resizeimg(groupno):
    img = cv2.imread('static/img/group' + groupno + '.jpg')
    h, w, _ = img.shape
    aspectratio = 600 / h
    img = cv2.resize(img, dsize=None, fx=aspectratio, fy=aspectratio)
    cv2.imwrite('static/img/group' + groupno + 'resize.jpg', img)


def imgmeteinfo(imgname):
    im = cv2.imread('static/img/group' + imgname)
    h, w, _ = im.shape
    return {'width': w, 'height': h}


if __name__ == '__main__':
    app.run()
