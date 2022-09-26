# Naive Bayes classifier for Fake News recognition

Fake news are defined by the New York Times as ”a made-up story with an intention to deceive”, with the intent to confuse or deceive people.
They are everywhere in our daily life, and come especially from social media platforms and applications in the online world.
Being able to distinguish fake contents form real news is today one of the most serious challenges facing the news industry.

Naive Bayes classifiers [1] are powerful algorithms that are used for text data analysis and are connected to classification tasks of text in multiple classes.

The goal of the project is to implement a Multinomial Naive Bayes classifier in R and test its performances in the classification of social media posts.
The suggested data set is available on Kaggle [2].
Possible suggested lables for classifying the text are the following:
* True - 5
* Not-Known - 4
* Mostly-True - 3
* Half-True - 2
* False - 1
* Barely-True - 0

The Kaggle dataset [2] consists of a training set wth 10,240 instances and a test set wth 1,267 instances.

## Group members
- [Paolo Zinesi](https://github.com/PaoloZinesi)
- [Nicola Zomer](https://github.com/NicolaZomer)

## Notebook 
In the notebook [NB_fake_news.ipynb](https://github.com/PaoloZinesi/NaiveBayesClassifier_Fake_News_Recognition/blob/main/NB_fake_news.ipynb) we implement by scratch the NB algorithm, after applying a cleaning procedure to our dataset. We also implement the concept of feature selection, in particular using as utility measure the Mutual Information. The models are trained and tested using the Kaggle dataset [2], and for each of them the accuracy is computed. We also train the same model using the R function  `naiveBayes()` from the library `e1071` and compare the results.

## App
After training the models, using the R package [Shiny](https://shiny.rstudio.com/) we built an interactive web app, which, given as input a social media post:
- Returns the score associated with each class as barplot
- Classifies the input text

The app is available at the following link: [app](https://paolozinesi.shinyapps.io/fake-news-recognition/).



## References
[1] C. D. Manning, Chapter 13, Text Classification and Naive Bayes, in Introduction to Information Retrieval, Cambridge University Press, 2008.

[2] Fake News Content Detection, KAGGLE data set: https://www.kaggle.com/datasets/anmolkumar/fake-news-content-detection?select=train.csv