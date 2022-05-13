from flask import Flask,render_template,request
from summarizer import Summarizer,TransformerSummarizer
import nltk
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
#nltk.download('punkt')   # one time execution
#nltk.download('stopwords')  # one time execution
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flask import request as req
app = Flask(__name__)

# @app.route("/about/")
# def about():
#     return render_template("about.html")

@app.route("/",methods = ["GET", "POST"])
def main():
    if req.method == "POST":
        if req.form.get('action1') == 'KMEANS':
            article = req.form.get("article")
            return render_template('main.html', result = kmeans_model(str(article)))
        elif req.form.get('action2') == 'BERT':
            article = req.form.get("article")
            return render_template('main.html', result = bert_model(str(article)))
        elif req.form.get('action3') == 'BERT':
            article = req.form.get("article")
            return render_template('main.html', result = bert(str(article)))
    elif req.method == 'GET':
        return render_template('main.html')
    
    return render_template("main.html")


# @app.route("/index/")
# def index():
#     return render_template('index.html')

@app.route("/about/")
def about():
    return render_template('about.html')
def kmeans_model(text):
    #importing the libraries

# data_path = 'Summary.txt'
# raw_data = open(data_path, 'r').read()
#sentence tokenization



    sentence = sent_tokenize(text)

# cleaning the sentences

    corpus = []
    for i in range(len(sentence)):
        sen = re.sub('[^a-zA-Z]', " ", sentence[i])  
        sen = sen.lower()                            
        sen=sen.split()                         
        sen = ' '.join([i for i in sen if i not in stopwords.words('english')])   
        corpus.append(sen)
    

#creating word vectors

    n=300
    all_words = [i.split() for i in corpus]
    model = Word2Vec(all_words, min_count=1,vector_size= n)

# creating sentence vectors

    sen_vector=[]
    for i in corpus:
    
        plus=0
        for j in i.split():
            plus+=model.wv[j]
        plus = plus/len(plus)
    
        sen_vector.append(plus)
    
#performing k-means  
    
    n_clusters = 5
    kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(sen_vector)

#finding and printing the nearest sentence vector from cluster centroid


    my_list=[]
    for i in range(n_clusters):
        my_dict={}
    
        for j in range(len(y_kmeans)):
        
            if y_kmeans[j]==i:
                my_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],sen_vector[j])
        min_distance = min(my_dict.values())
        my_list.append(min(my_dict, key=my_dict.get))

    

                            
# print(my_list)
# print(y_kmeans)
    ans=""
    for i in sorted(my_list):
        ans+= (sentence[i])

    return ans
def bert_model(text):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(text, min_length=10))
    return (bert_summary)

def bert(text):
    import requests

    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    # API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer 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"}

    def query(payload):
	    response = requests.post(API_URL, headers=headers, json=payload)
	    return response.json()
	
    output = query({
        "inputs":text,
        # "parameters":{"min_length":50, "max_length":150}
    })
    return output;

if __name__ == "__main__":
    app.run(debug=True)


