## Edit Names: KNN Edit Distance Classifier

### Workflow

1. Take the entire FL dataset and group_by last name producing the following cols: `last_name, prop_white, prop_hispanic, ..., total_n`
2. Split the grouped data into train and test
3. Using the train set, learn the optimal k.
4. Distance metrics: We try different string distance metrics. We start with 'levenshtein distance' on the strings. We then use cosine distance on bi-chars. We also try cosine distance on popular bi-chars and a tf-idf matrix.
5. use the optimal k to estimate generalization error on the test set. We try binary search to find the optimal k, starting with extreme choices 3 and 3000, assuming monotonicity. 

When predicting, we take a weighted prediction based on the n of each of the names. For instance, say the closest names are:
	ABC, n = 100, p_white = 100
	BBC, n = 10, p_white = 0
then prediction = (100*1 + 10*0)/110
6. compare performance to LSTM based methods

### Authors

Bashar Naji and Gaurav Sood
