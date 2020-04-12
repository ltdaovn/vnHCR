#import flask
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
#import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from urllib.request import urlopen

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

	
@app.route('/prediction_iris', methods=['POST','GET'])
def prediction_iris():
	if request.method=='POST':
		loaded_model = load('Iris.DecisionTree.joblib')
		classes = ["Setosa", "Versicolor", "Virginica"]
		X_new = ([[request.form['sep_len'], 
			request.form['sep_wid'],
			request.form['pet_len'],
			request.form['pet_wid']]])
		y_new = loaded_model.predict(X_new)
		return render_template('result.html',results=classes[y_new[0]])

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"got name {name}")

    response = {}

    # Check if user sent a name at all
    if not name:
        response["ERROR"] = "no name found, please send a name."
    # Check if the user entered a number not a name
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    # Now the user entered a valid name
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"

    # Return the response in json format
    return jsonify(response)
	

#http://localhost:5000/prediction?url=http://trolyao.cusc.vn/image/m.jpg


#@app.route('/prediction')
@app.route('/prediction', methods=['GET'])
def predition():
	#url = 'http://trolyao.cusc.vn/image/m.jpg'
	url = ''
	if request.method=='GET':
		url = request.args.get("url", None)
	
	#img = cv2.imread('1.jpg', cv2.IMREAD_COLOR) # đọc ảnh
	#img = url_to_image("http://trolyao.cusc.vn/image/m.jpg");
	#img = url_to_image("http://trolyao.cusc.vn/image/i.jpg");
	#img = url_to_image("http://trolyao.cusc.vn/image/g.jpg");
	img = url_to_image(url);
	img_data = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)  # chuyển kích thước ảnh về 128x128
	thresh_1 = 150 # ngưỡng đặc trưng của ảnh, dươi ngưỡng pixel có màu đen và ngược lại là màu trắng.
	img_agray = cv2.threshold(img_data, thresh_1, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
	thresh_2 = 128
	(thresh, img_bin) = cv2.threshold(img_agray, thresh_2, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
	img_color = cv2.cvtColor(img_bin, 1)
	
	
	
	with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
		model = load_model("./models/weights.01.h5")
		# kien truc mo hinh
		# model.summary()
		names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
				 10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
				 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
				 31: "V", 32: 'W', 33: "X", 34: "Y", 35: "Z"}
	
	"""
	names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
				 10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
				 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
				 31: "V", 32: 'W', 33: "X", 34: "Y", 35: "Z"}
	"""
	#model = load_model("weights.03.h5")

	# dự đoán
	prediction = model.predict(img_color.reshape(1, 128, 128, 3))

	# in ra mãng các giá trị dự đoán
	#print(prediction)

	# lấy phần tử có giá trị lớn nhất
	predict_img = np.argmax(prediction, axis=-1)

	# in ra kết quả dự đoán
	#return("A")
	#return(names.get(predict_img[0]))
	#return render_template('result.html',results=names.get(predict_img[0]))
	#return render_template('result.html',results='A')
	return jsonify(names.get(predict_img[0]))
	#return jsonify(keras.__version__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']

    # Save file
    #filename = 'static/' + file.filename
    #file.save(filename)

    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Detect faces
    faces = detect_faces(image)

    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)
        
        # Draw a rectangle
        for item in faces:
            draw_rectangle(image, item['rect'])
        
        # Save
        #cv2.imwrite(filename, image)
        
        # In memory
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)

	
if __name__ == '__main__':
	#app.run()
	app.run(threaded=False)
	#print(predition())
	