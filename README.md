## Edit Names: KNN Edit Distance Classifier

### Workflow

1. Take the entire FL dataset and group_by last name producing the following cols: `last_name, prop_white, prop_hispanic, ..., total_n`

2. Split the grouped data into train and test
3. Using the train set, learn the optimal k.
4. use the optimal k to estimate generalization error on the test set.
5. compare performance to LSTM


### Authors

Bashar Naji and Gaurav Sood
