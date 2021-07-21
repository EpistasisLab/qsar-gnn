# !/usr/iinfenv python
# created by Yun Hao @MooreLab 2019
# This script learns a simple classfication model based on specified hyperparameters, then evalutes its performance 


# Module
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/')
import simple_learning


## Main function 
def main(argv):
	## 0. Input arguments 
		# argv 1: input file that contains training feature-response data 
		# argv 2: input file that contains testing feature-response data
		# argv 3: name of label(response) column
		# argv 4: prefix of output file name  
		# argv 5: classification methods to be used 'RandomForest' or 'XGBoost' 
		# argv 6: string that contains hyperparamter setting of classifier. Format: n_estimators:XX,criterion:XX,max_features:XX,min_samples_split:XX,min_samples_leaf:XX,bootstrap:XX (for 'Randomforest'); n_estimators:XX,max_depth:XX,learning_rate:XX,subsample:XX,min_child_weight:XX (for 'XGBoost')

	## 1. Read in input training and testing files
	# read in training data
	outcome_col = argv[3]
	train_data_df = pd.read_csv(argv[1], sep = '\t', header = 0, index_col = 0)
	train_X_data, train_y_data = train_data_df.drop(outcome_col, axis = 1).values, train_data_df[outcome_col].values
	# read in testing data 
	test_data_df = pd.read_csv(argv[2], sep = '\t', header = 0, index_col = 0)
	test_X_data, test_y_data = test_data_df.drop(outcome_col, axis = 1).values, test_data_df[outcome_col].values
	
	## 2. Learn a classification model then evaluate its performance 
	# build classifier with training data, compute performance metrics with training & testing data  
	simple_learner, hyper_str, train_perf, test_perf = simple_learning.build_simple_classifier(train_X_data, test_X_data, train_y_data, test_y_data, argv[5], argv[6])
	# write performance metrics to output file 
	output_perf_list = simple_learning.generate_simple_performance_file(train_X_data.shape[0], test_X_data.shape[0], argv[6], train_perf, test_perf)
	output_file = argv[4] + '_md_' + argv[5] + hyper_str + '_performance.txt'
	perf_op = open(output_file, 'w')
	for opl in output_perf_list:
		perf_op.write('%s\n' % opl)
	perf_op.close()

	return 1	


## call main function
main(sys.argv)
