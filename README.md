## Edit Names: KNN Edit Distance Classifier

Using the [Florida Voting Registration Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UBIG3F), we estimate a knn classifier with an edit distance based distance metric to predict the race and ethnicity of *unseen* names. (We don't try to learn the naive Bayes classifier with k = 0 and the name in the 'look-up' corpus.) 

For one set of analyses, we assume that the true label for a name is the modal race/ethnicity that people with that last name identify with. The OOS confusion matrix:

|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| asian         | 0.54      | 0.21   | 0.30     | 1504    |
| hispanic      | 0.85      | 0.78   | 0.81     | 10775   |
| multi_racial  | 0.02      | 0.00   | 0.00     | 364     |
| native_indian | 0.02      | 0.02   | 0.02     | 111     |
| nh_black      | 0.59      | 0.42   | 0.49     | 4483    |
| nh_white      | 0.79      | 0.92   | 0.85     | 25614   |
| other         | 0.18      | 0.03   | 0.04     | 582     |
| accuracy      |           |        | 0.78     | 43433   | 
| macro avg     | 0.43      | 0.34   | 0.36     | 43433   |
| weighted avg  | 0.76      | 0.78   | 0.76     | 43433   |

For another set of analyses, we use the distribution (what proportion of people with last name X are nh_black, nh_white, etc.) and compute the RMSE. The generalization RMSE (for cosine distance only; see below for link to nb) is .16.

### Workflow

1. Take the entire FL dataset and group_by last name producing the following cols: `last_name, prop_white, prop_hispanic, ..., total_n`
2. Split the grouped data into train, validation, and test sets
3. Using the train and the validation sets, learn the optimal k
4. Because calculating the Levenshtein distance is expensive, we follow the following strategy:
	* We use cosine distance on tf-idf based bichar tokens to filter down to names that have a cosine similarity of .6 or greater.
	* We estimate Levenshtein distance to those names and pick the k closest. (Where there are more names that are as far away as k, we include all those names till we hit all the records passed by previous step.)  
5. We tried a few choices for k: 3, 5, 25.
6. When predicting, we take a simple average and predict the one with the max probability. 

### Script

* [KNN Edit Distance Classifier Notebook](notebooks/knn_cosine_levenshtein_threadpool.ipynb)

### Variations

* **What Happens When We Just Use Cosine Distance?** As the [notebook](notebooks/knn_cosine_threadpool.ipynb) shows, results are roughly the same.
* **What Happens If We Use Weighted Mean Instead of a Simple Average?** As the notebooks for [cosine distance](notebooks/knn_cosine_threadpool_with_weighted_mean.ipynb) and [levenshtein with cosine distance](notebooks/knn_cosine_levenshtein_threadpool_with_weighted_mean.ipynb) show, the results look pretty much the same. (If you are confused about what that means, take a look at this [notebook](notebooks/compare_simple_weighted_mean.ipynb). Here's a quick example: say the closest names are: ABC, n = 100, p_white = 100; BBC, n = 10, p_white = 0 then prediction = (100*1 + 10*0)/110.)
* **What If We Use RMSE?** The [notebook](notebooks/knn_cosine_threadpool_rmse.ipynb) provides RMSE for cosine distance based knn.
* **What is the Baseline Performance When We Predict k Most Popular Names?** See [notebook](notebooks/knn_popular_names.ipynb). RMSE is .3 and accuracy is 59%.
* **Using Minhash LSH/Jaccard for FL 2022** See [notebook](notebooks/knn_last_name_2022_jaccard_lsh.ipynb). We see a 5 point hit to accuracy.


### Authors

Suriyan Laohaprapanon, Bashar Naji, and Gaurav Sood

