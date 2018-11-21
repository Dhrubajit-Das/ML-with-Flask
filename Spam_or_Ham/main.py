from flask import Flask,render_template,request,url_for
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/",methods=['POST'])
def predict_sentiment():
	# Link to dataset from github
	# url = "https://raw.githubusercontent.com/Jcharis/Machine-Learning-Web-Apps/master/Youtube-Spam-Detector-ML-Flask-App/YoutubeSpamMergedData.csv"
	
	df= pd.read_csv("spamham.csv")
	#df= pd.read_csv("spamham.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	df_x = df_data['CONTENT']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	

	if request.method == 'POST':
		print (request.form)

		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		pred = clf.predict(vect)
	return render_template('results.html',prediction = pred, comment = comment)
	
@app.route("/")
def go_back():
	return render_template("index.html")


if __name__ == '__main__':
	app.run(debug=True)


