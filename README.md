## Edit Names: KNN Edit Distance Classifier

We estimate a knn classifier with an edit distance based distance metric to predict the race and ethnicity of *unseen* names. (We don't try to learn the naive Bayes classifier with k = 0 and name in the corpus.) The accuracy of the knn classifier is XX%. The OOS confusion matrix:


### Workflow

1. Take the entire FL dataset and group_by last name producing the following cols: `last_name, prop_white, prop_hispanic, ..., total_n`
2. Split the grouped data into train and test
3. Using the train set, learn the optimal k
4. Because calculating the Levenshtein distance is very expensive, we follow the following strategy:
	* We use cosine distance on tf-idf based bichar tokens to filter down to 5k names that are closest to the name
	* We estimate Levenshtein distance to 5k names and pick the k closest. (Where there are more names that are as far away as k, we include all those names till 5k.)  
5. We try ~binary search to find the optimal k, starting with extreme choices 3 and 3000, assuming monotonicity.
6. When predicting, we take a weighted prediction based on the n of each of the names. For instance, say the closest names are:
	ABC, n = 100, p_white = 100
	BBC, n = 10, p_white = 0
	then prediction = (100*1 + 10*0)/110
7. compare performance to the LSTM model

 
### Scripts

* [KNN Edit Distance Classifier Notebook](scripts/knn_edit_classifier.ipynb)
* [KNN Cosine Distance Classifier Notebook](scripts/knn_cosine_classifier.ipynb)

### Future

To make search for k-nearest [edit distance] neighbors faster, we plan to implement a BK-Tree backend.

### Authors

Bashar Naji and Gaurav Sood
