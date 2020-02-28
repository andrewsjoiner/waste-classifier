
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# import cv2
import pandas as pd
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np
import h5py
import random
import shutil


# In[ ]:


from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
#from imutils import paths
import random


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#dDELETE FILES FROM TRAIN AND TEST FOLDER TO RECREATE IT
import os, shutil
def delete_File(folderpath):
    for filename in os.listdir(folderpath):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

            


# In[ ]:


folder = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/train'

delete_File(folder)


# In[ ]:


#data_dir = 'waste-classification-data-v2\\DATASET\\'
train_data_N = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TRAIN/N'
train_data_o = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TRAIN/O'
train_data_r = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TRAIN/R'
test_data_N = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TEST/N'
test_data_o = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TEST/O'
test_data_r = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TEST/R'
#index =0


# In[ ]:


##### regenerate images for class 'o' or N class to create balanced data ###################
############## ALREADY DONE DO NEED TO RUN ONCE AGAIN
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import os
# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
train_data_N = 'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TRAIN/N'
new_data_N =  'C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input/DATASET/TRAIN/N'
#image = load_img('/floyd/home/datasets/test/00000281.jpg')
paths = os.listdir(train_data_N)
for imagepath in paths:
    image = load_img(train_data_N +"/"+imagepath)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,height_shift_range=0.1, 
                shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
    total = 0

    # construct the actual Python generator
    #print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=new_data_N,save_prefix='new_aug',
                        save_format="jpg")

    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1

        # if we have reached 10 examples, break from the loop
        if total == 3:
            break


# In[ ]:


def add_data_dataframe(dataframe,directory,labelPath,class_id,indx =0):
    #paths = os.listdir('/floyd/home/datasets/input/DATASET/TRAIN/O')
    paths = os.listdir(directory)
    #paths = directory
    print(indx)
    #print(paths)
    #print(paths)


    for image in paths:
        dataframe.loc[indx,'id_code'] = image
        dataframe.loc[indx,'path'] = labelPath +"/"+image
        dataframe.loc[indx,'category'] = class_id
        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([labelPath, image])
#         print(p)
#         print('labelPath +"/"+image')
#         print(labelPath +"/"+image)
#         print(directory +"/"+image)
        shutil.copy2(directory +"/"+image, p)
        indx +=1 
    return indx    
    


# In[ ]:


#loop over the input images
data_table = pd.DataFrame(columns=['id_code','path','category'])
labelpath = "C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/train"
index =0
index = add_data_dataframe(dataframe= data_table,directory=train_data_o,labelPath =labelpath,class_id=1,indx=index)
index = add_data_dataframe(dataframe= data_table,directory=train_data_r,labelPath =labelpath,class_id=2,indx=index)
index = add_data_dataframe(dataframe= data_table,directory=train_data_N,labelPath =labelpath,class_id=0,indx=index)


# In[ ]:


# saving the dataframe 
data_table.to_csv('C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/train.csv') 


# In[ ]:


data_table.path.sample(10)


# In[ ]:


# display the value in count
data_table.category.value_counts().to_frame(name='count').T


# In[ ]:


#after adding new data for class N
# display the value in count
data_table.category.value_counts().to_frame(name='count').T


# In[ ]:


#display the data in percentage
data_table.category.value_counts(normalize=True) * 100


# In[ ]:


#data_table.groupby('category')['path']
grouped = data_table.groupby('category')
df_sample =grouped.apply(lambda x: x.sample(5))


# In[ ]:


def process(filename):    
    image = plt.imread(filename)
    #<something gets done here>
    plt.figure()
    plt.imshow(image)


# In[ ]:


for row in df_sample.iterrows():
    #print(str(row[1][1]))
    process(str(row[1][1]))


# In[ ]:


#loop over the input images
data_table = pd.DataFrame(columns=['id_code','path','category'])
labelpath = "C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/test"
index =0
index = add_data_dataframe(dataframe= data_table,directory=test_data_o,labelPath =labelpath,class_id=1,indx=index)
index = add_data_dataframe(dataframe= data_table,directory=test_data_r,labelPath =labelpath,class_id=2,indx=index)
index = add_data_dataframe(dataframe= data_table,directory=test_data_N,labelPath =labelpath,class_id=0,indx=index)


# In[ ]:


# saving the dataframe 
data_table.to_csv('C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/test.csv') 


# In[ ]:


#display the data in percentage
data_table.category.value_counts(normalize=True) * 100


# In[ ]:


groupedtest = data_table.groupby('category')
df_sampletest =groupedtest.apply(lambda x: x.sample(5))


# In[ ]:


for row in df_sampletest.iterrows():
    process(str(row[1][1]))


# In[ ]:


import zipfile
path_to_zip_file = "C:/Work/waste_classifier/Wastemanagement/Wastemanagement/waste-classification-data-v2.zip"
directory_to_extract_to = "C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/input"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)


# In[ ]:


import hdf5datacreator


# In[ ]:


import finetune_waste


# In[ ]:


import finetune_waste


# In[ ]:


import config
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from sklearn.metrics import classification_report
from hdf5datasetgenerator import HDF5DatasetGenerator
from keras.models import load_model
import numpy as np
#import progressbar
import json

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
#cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()
#
# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.Model_PATH)
#
classNames = {0: "Non-Recyclable", 1: "Organic", 2: "Recyclable"}
# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 32,preprocessors=[sp, mp, iap], classes=len(classNames))

# reset the testing generator and then use our trained model to
# make predictions on the data
predictions = model.predict_generator(testGen.generator(),steps=testGen.numImages // 32, max_queue_size=10)

print(classification_report(testGen.db["labels"],predictions.argmax(axis=1), target_names=classNames))


# In[ ]:


from simpledatasetloader import SimpleDatasetLoader
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
import pandas as pd
import config
import json
from keras.models import load_model

means = json.loads(open(config.DATASET_MEAN).read())
# initialize the image preprocessors

aap = AspectAwarePreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
#cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap,mp, iap])
#imagePaths = list('/floyd/home/datasets/test')
dftest = pd.read_csv('C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/test.csv')

#shuffle dataframe in-place and reset the index
dftest = dftest.sample(frac=1).reset_index(drop=True)

imagePaths = dftest.path.values.tolist()
Labels=dftest.category.values.tolist()
(data, labels) = sdl.load(imagePaths,Labels, verbose=500)
#data = data.astype("float") / 255.0
# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.Model_PATH)

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(data, batch_size=32)
classNames = {0: "Non-Recyclable", 1: "Organic", 2: "Recyclable"}
from sklearn.metrics import classification_report
import numpy as np
print(classification_report(labels,predictions.argmax(axis=1)))              


# In[ ]:


from sklearn.metrics import classification_report
import numpy as np
print(classification_report(labels,predictions.argmax(axis=1)))


# In[ ]:


df = pd.read_csv( config.BASE_PATH +'/train.csv')
df = df.sample(frac=1).reset_index(drop=True)


# In[ ]:


# testing for single image on test data for class zero (0)
import pandas as pd
import config
import json
from keras.models import load_model
from keras.models import model_from_json
import h5py
import cv2
import os
from keras.preprocessing.image import img_to_array
import numpy as np

means = json.loads(open(config.DATASET_MEAN).read())
# initialize the image preprocessors
# load the image for classification
images = []
#folder ="c:\image-classification-keras\examples"
model = load_model(config.Model_PATH)
image = cv2.imread('C:/Work/waste_classifier/Wastemanagement/Wastemanagement/datasets/test/00000281.jpg')
if image is not None:
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    images.append(image)
    testimage = model.predict(image)[0]
print(testimage)


# In[ ]:


import finetune_waste

