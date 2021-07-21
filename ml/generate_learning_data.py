# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script generates feature-response datasets for learning task from group-sample relationships, then split each dataset into train set and test set  


## Module
import sys
sys.path.insert(0, 'src/')
import data


## Main function 
def main(argv):
	## 0. Input arguments: 
		# argv 1: input file that contains group-sample relationships (three columns, 1: group, 2: sample, 3: label) 
		# argv 2: input file that contains computed features of samples 
		# argv 3: folder to store the output files  
		# argv 4: minimum number of samples required 
		# argv 5: proportion of test data   

	## 1. Generate dataset
	generate_data = data.generate_learning_dataset(argv[1], argv[2], int(argv[4]), float(argv[5]), argv[3])

	return 1


## Call main function
main(sys.argv)

