import numpy as np
import pickle as pkl
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from flask import Flask, request,render_template,redirect,url_for
from flask_cors import CORS
from sklearn.externals import joblib
import pickle
import flask
import urllib
import pandas as pd
import numpy as np
import gzip
import re
import os
from scipy.sparse import hstack
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from textblob import TextBlob
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, flash, request, redirect, url_for
from flask import Flask, url_for, send_from_directory, request
import gunicorn
import glob
import os
import glob
import pickle
import pandas as pd
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid     
import pytesseract
import cv2
import math
from pythonRLSA import rlsa
import numpy as np
import pickle as pkl
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from flask import Flask, request,render_template,redirect,url_for
from flask_cors import CORS
from sklearn.externals import joblib
import pickle
import flask
import urllib
import pandas as pd
import numpy as np
import gzip
import re
import os
from scipy.sparse import hstack
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from textblob import TextBlob
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, flash, request, redirect, url_for
from flask import Flask, url_for, send_from_directory, request
import gunicorn
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy
from pythonRLSA import rlsa
import math
import os
from flask import Flask, render_template, request
__author__ = 'ibininja'

app = Flask(__name__)



app=flask.Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
CORS(app)



class topic_classifier():
    """
    ***parameter***
    - file_dir: File path of BBC training data
    - topics: tweet topic from one of these topics: 'business','entertainment','politics','sport' and 'tech'
    - classifier: Machine learning multi-class classifier from one of the following classifiers
         +'mulNB': Naive Bayes 
         +'svc': SVC
         +'dec_tree': Decision Tree
         +'rand_forest': Random Forest
         +'random_sample': Random Sample
         +'nearest_cent': Nearest centroid
         +'mlp': Multi-layer Perceptron
    """
    def __init__(self,file_dir,topics,classifier):
        self.file_dir=file_dir
        self.topics=topics
        self.classifier=None
        self.algorithm=classifier
        self.method=None



    #extracting training texts from different folders and dumping them into a dataframe
    def train_topics_gen(self):
        content=[]
        classes=[]
        for topic in self.topics:
            user_set_path = os.path.join(self.file_dir,topic)
            os.chdir(user_set_path)
            files=glob.glob("*.txt")
            for file in files:
                with open(file) as f:
                    content.append(f.read())
                    classes.append(topic)
        DF = pd.DataFrame({'class': classes,'content': content})
        return DF


    # training the classifier using the BBC training data
    def training(self):
        if self.algorithm=='mulNB':
            self.classifier = MultinomialNB()
        elif self.algorithm=='svc':
            self.classifier=OneVsRestClassifier(SVC())
        elif self.algorithm=='dec_tree':
            self.classifier=DecisionTreeClassifier()
        elif self.algorithm=='rand_forest':
            self.classifier=RandomForestClassifier()
        elif self.algorithm=='random_sample':
            self.classifier=RandomForestClassifier()
        elif self.algorithm=='nearest_cent':
            self.classifier=NearestCentroid()
        elif self.algorithm=='mlp':
            self.classifier=MLPClassifier()

        # BBC training dataset
        df=self.train_topics_gen()
        # vectorizing the contents of the data
        self.method = CountVectorizer()
        counts = self.method.fit_transform(df['content'].values)
        targets = df['class'].values
        self.classifier.fit(counts, targets)
        return self



print("Loading classifier model")
pickle_model1 = "finalized_model.sav"
loaded_model = pickle.load(open(pickle_model1, 'rb'))

print("Loading models")
pickle_model = "models/wb_transform.pkl"
clf1= pkl.load(gzip.open(pickle_model, 'rb'))

@app.route("/")
def index():
    return render_template("upload.html")



non_alphanums = re.compile(u'[^A-Za-z0-9]+')
def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

stemmer = SnowballStemmer("english")
def preprocess(df):
    df['author'].fillna('No author', inplace=True)
    df['title'].fillna('No title', inplace=True)
    df['text'].fillna('No text', inplace=True)
    df_author = pd.read_csv('author_cat.csv')
    df['author_cat'] = 1
    df['stemmed_title'] = df['title'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
    df['stemmed_text'] = df['text'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
    df.drop(['title', 'author', 'text'], axis=1, inplace=True)
    return df


@app.route("/upload", methods=['POST'])
def upload():
    
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    return render_template("upload.html",prediction_text="Sucessfully Uploaded")




@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    filenames = [img for img in glob.glob("images/*.png")]
    images = []
    for img in filenames:
       n= cv2.imread(img)
       images.append(n)

    image = cv2.imread(img) # reading the image
    


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert2grayscale
    (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # convert2binary

    contours, hierarchy = cv2.findContours(~binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
# find contours
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)

    mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
    contours, hierarchy = cv2.findContours(~binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    heights = [cv2.boundingRect(contour)[3] for contour in contours] # collecting heights of each contour
    avgheight = sum(heights)/len(heights) # average height
# finding the larger contours
# Applying Height heuristic
    for c in contours:
        [x,y,w,h] = cv2.boundingRect(c)
        if h > 2*avgheight:
            cv2.drawContours(mask, [c], -1, 0, -1)

    x, y = mask.shape
    value = max(math.ceil(x/100),math.ceil(y/100))+20 #heuristic
    mask = rlsa.rlsa(mask, True, False, value) #rlsa application


    contours, hierarchy = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours
    mask2 = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if w > 0.60*image.shape[1]: # width heuristic applied
            title = image[y: y+h, x: x+w] 
            mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
            image[y: y+h, x: x+w] = 255 # nullified the title contour on original image


    title = pytesseract.image_to_string(Image.fromarray(mask2))
    
    im = cv2.imread(img) 
    content = pytesseract.image_to_string(im)
    path_to_dir  = 'images/'  # path to directory you wish to remove
    files_in_dir = os.listdir(path_to_dir)     # get list of files in the directory

    for file in files_in_dir:                  # loop to delete each file in folder
        os.remove(f'{path_to_dir}/{file}')
   # os.remove(img)

    content = content.replace("\n", " ")

    d = {'title': [title],
     'text': [content],
    'author': ["Beekash Mohanty"]}

    df_test = pd.DataFrame(data=d)
#stopwords = {x: 1 for x in stopwords.words('english')}
    non_alphanums = re.compile(u'[^A-Za-z0-9]+')

    def normalize_text(text):
        return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


    print("Loading models")
    pickle_model = "models/wb_transform.pkl"
    clf1= pkl.load(gzip.open(pickle_model, 'rb'))

    stemmer = SnowballStemmer("english")
    def preprocess(df):
       df['author'].fillna('No author', inplace=True)
       df['title'].fillna('No title', inplace=True)
       df['text'].fillna('No text', inplace=True)

    #search author encoded
       df_author = pd.read_csv('author_cat.csv')

    #TODO check at notebook the values for the author and the equal query set the cateory id right
       df['author_cat'] = 1
       df['stemmed_title'] = df['title'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
       df['stemmed_text'] = df['text'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
 
    # drop the title autor and text
       df.drop(['title', 'author', 'text'], axis=1, inplace=True)

       return df

    df= preprocess(df_test)
    vectorizer = HashingVectorizer(normalize_text,decode_error='ignore',n_features=2 ** 23, non_negative=False, ngram_range=(1, 2), norm='l2')

    X_title = vectorizer.transform(df['stemmed_title'])
#X_title = X_title[:, np.array(np.clip(X_title.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    X_text = vectorizer.transform(df['stemmed_text'])
#X_text = X_text[:, np.array(np.clip(X_text.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    X_author = df['author_cat'].values
    X_author = X_author.reshape(-1, 1)

    sparse_merge = hstack((X_title, X_text, X_author)).tocsr()

# Remove features with document frequency <= 100
    mask100 = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 1, 0), dtype=bool)
    X = sparse_merge[:, mask100]
    print(X.shape)
    print('Loading model to predict...')
    print('Loading model to predict...')
 
    y1 = clf1.predict(X)


    bloblist_desc = list()

    df_usa_descr_str=df_test['stemmed_text'].astype(str)
    for row in df_usa_descr_str:
        blob = TextBlob(row)
        bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
        df_usa_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])

    tweet_counts = loaded_model.method.transform(df_test['stemmed_text'])
    predictions = loaded_model.classifier.predict(tweet_counts)
    
    def f(df_usa_polarity_desc):
        if df_usa_polarity_desc['sentiment'] > 0:
            val = "Positive"
        elif df_usa_polarity_desc['sentiment'] == 0:
            val = "Neutral"
        else:
            val = "Negative"
        return val

    df_usa_polarity_desc["Sentiment_Type"] = df_usa_polarity_desc.apply(func=f, axis=1)

    cal = np.round(y1, 5)*100
    if cal > 98 :
        m = "This News is Fake"
    elif cal > 90 and cal < 98:
        m = "This News is more likely a Fake"
    else:
        m = "This News is Real"

    return render_template("upload.html", prediction_text= "Fake Rate={}".format(np.round(y1, 4)*100)+"%"+"->"+m+" "+"   Sentiments="+df_usa_polarity_desc["Sentiment_Type"].values+" "+"Category="+predictions)


if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
#C:\Program Files\Tesseract-OCR    