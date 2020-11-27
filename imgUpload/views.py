from django.shortcuts import render
from .forms import ImageUploadForm
#Importing pre trained deep-learning framework , restNet 50 pretrained on image-net data
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
# Create your views here.
def handle_uploaded_file(f):
	with open('img.jpg','wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)

def home(request):
	return render(request,'home.html')

def imageprocess(request):
	form=ImageUploadForm(request.POST,request.FILES)
	if form.is_valid():
		handle_uploaded_file(request.FILES['image'])

		model = ResNet50(weights='imagenet')
		img_path='img.jpg'
		#loading the image
		img = image.load_img(img_path, target_size=(224,224)) 
		#convert image to array , 2D
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		#preprocess data and get ready for prediction
		x = preprocess_input(x)
		#predicting the values
		preds=model.predict(x)
		#predicting the animal
		print('Predicted animal :', decode_predictions(preds, top=3)[0])

		html=decode_predictions(preds,top=3)[0]
		res=[]
		for e in html:
			res.append((e[1],np.round(e[2]*100,2)))
		return render(request,'result.html',{'res':res})

	return render(request,'result.html')
