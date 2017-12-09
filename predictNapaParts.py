from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import request
import base64
import sys
import json
import numpy as np
import pandas as pd
import geopy.distance
import requests
import tensorflow as tf
from flask import Flask,jsonify,request
from operator import itemgetter, attrgetter
from random import randint
import io
from google.cloud import vision
vision_client = vision.Client("MyProject47443-3090c70d9f52.json")
app = Flask(__name__)
def detectImage(file_name):
    with io.open(file_name,'rb') as image_file:
        content = image_file.read()
        image = vision_client.image(content=content)
    labels = image.detect_labels()
    visionlabel = labels[0].description
    validPart = ['product','hardware','automotive','liquid','car','spray','solvent','lubricant','tire','rim','blue','cylinder','electronics','Technology','glass','electronic device','artifact','bottle','wheel','sphere','metal']
    if any(visionlabel in s for s in validPart):
        isValidPart="true"
        #return (isValidPart,visionlabel)
        return {'isValidPart':isValidPart, 'visionlabel':visionlabel.title()}
    else:
        isValidPart="false"
        return {'isValidPart':isValidPart, 'visionlabel':visionlabel.title()}
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
def getNearByStores(from_lat,from_long,partNum,lineAbbrev):
    distances=[]
    lat1 = str(from_lat)
    lon1 = str(from_long)
    partInqReq={
  "CorrelationID":"481bfd23-aa55-4de0-a7da-2f932ab121ee",
  "SourceSystem":"IBS Portal",
  "CustomerNumber":4757,
  "Part":[
    {
      "LineAbbrev":lineAbbrev,
      "PartNumber":partNum
    }
  ],
  "Customer":{
    "CustomerNumber":801,
    "Password":"fac69fe3f2cc0ca115a8362df910de4b4eee0d61"
  }
}
    headers = {'content-type': 'application/json'}
    URL="http://10.10.10.233:18080/taap/invoicemanagement/partPriceAvailability"
    response=requests.post(url = URL, data = json.dumps(partInqReq),headers=headers)
    partRes = response.json()
    #onhand = randint(1, 100)
    price = partRes['DetailResponse']['Response']['Responder'][0]['Part'][0]['Price']['RegularPrice']
    desc = partRes['DetailResponse']['Response']['Responder'][0]['Part'][0]['PartAttributes']['PartDescription']
    df = pd.read_csv("StoreAddressByLocation.csv", names=['ADR_LN_1', 'ADR_LN_2', 'CTY_NM', 'CY_NM', 'ST_CD', 'PSTL_CD','CTRY_CD','CTY_TX_PCT','CY_TX_PCT','ST_TX_PCT','TM_ZNE_NUM','LATITUDE','LONGITUDE'])
    #data=df[(df['LATITUDE'] > lat1-1)&(df['LONGITUDE'] < lon1)][:10]
    df = df[((df['LATITUDE'] >= str(from_lat-0.5)) & (df['LATITUDE'] <= str(from_lat+0.5)))]
    data = df[((df['LONGITUDE'] >= str(from_long+0.5)) & (df['LONGITUDE'] <= str(from_long-0.5)))]
    coords_1 = (lat1, lon1)
    for index, row in data.iterrows():
        if(index>0):
            coords_2=(row['LATITUDE'],row['LONGITUDE'])
            address=str(row['ADR_LN_1'])+' '+str(row['CTY_NM'])+' '+str(row['ST_CD'])+' '+str(row['PSTL_CD'])
            miles=round(geopy.distance.vincenty(coords_1, coords_2).mi,2)
            onhand = randint(1, 20)
            if(miles < 25):
                distances.append({"address":address,"miles":miles,"onHand":onhand,"price":price,"desc":desc})
    return (sorted(distances, key=itemgetter('miles')))
@app.route('/partsearch/getPartsInfo',methods=['POST'])
def getPartsInfo():
    print("reached")
    #print(request.json())
    lat=request.json['lat']
    longt=request.json['longt']
    partNum=request.json['partNum']
    lineAbbrev=request.json['lineAbbrev']
    partDet=getNearByStores(lat,longt,partNum,lineAbbrev)
    print(pd.Series(partDet).to_json(orient='values'))
    return pd.Series(partDet).to_json(orient='values'), 201
@app.route('/partsearch/uploadImage',methods=['POST'])
def getImage():
    imageData = request.json['image']
    file_name = open('myfile.jpg', 'wb')
    file_name.write(base64.b64decode(imageData))  # python will convert \n to os.linesep
    file_name.close()
    #visionResults=detectImage('myfile.jpg')
    partsData=predictPart('myfile.jpg')
    print("valid part")
    return partsData,201
    #print("visionAPI result"+visionResults['isValidPart'])
    #if(visionResults['isValidPart']=="true"):
        #partsData=predictPart('myfile.jpg')
        #print("valid part")
        #return partsData,201
    #else:
        #print("invalid part")
        #print(json.dumps(visionResults))
        #return json.dumps(visionResults),201
def predictPart(image_file):
    #print("resuest received"+request)
    data = {}
    parts = []
    model_file = "C:\\Users\\398714\\Desktop\\PartPOC\\retrained_graph.pb"
    label_file = "C:\\Users\\398714\\Desktop\\PartPOC\\retrained_labels.txt"
    #prodType = request.json['prodType']
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image(image_file,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        lineAbbrev=labels[i].split()[0].upper()
        partNum=labels[i].split()[1].upper()
        imageName=lineAbbrev+'-'+partNum+'.jpg'
        prediction=results[i]
        if(prediction > 0.5):
            parts.append(
                {
                    "lineAbbrev":lineAbbrev,
                    "partNum":partNum,
                    "prediction":round(prediction*100,2),
                    "imageName":imageName
                }
            )
    print(pd.Series(parts).to_json(orient='values'))
    return pd.Series(parts).to_json(orient='values')
if __name__ == '__main__':
    app.run(host='10.x.xxx.xxx',port=8080,debug=True)
