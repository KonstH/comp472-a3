from NB_BOW_OV import NBC_OV
from NB_BOW_FV import NBC_FV
import argparse

# Sets up the arguments that can be passed in the terminal
parser = argparse.ArgumentParser(description="Use Naive Bayes Classifiers to predict tweet information verifiableness")
parser.add_argument('-trn', '--trainfilename', type=str, default='covid_training.tsv', metavar='', help="Name of training input file (default: covid_training.tsv)")
parser.add_argument('-tst', '--testfilename', type=str, default='covid_test_public.tsv', metavar='', help="Name of test input file (default: covid_test_public.tsv)")
parser.add_argument('-s', '--smoothing', type=float, default=0.01, metavar='', help="Smoothing value (default: 0.01)")
parser.add_argument('-v', '--nbversion', type=str, metavar='', help="Version of NB Classifier to run. Options: OV or FV (default: runs both versions)")
args = parser.parse_args()

# Initializes both NBC Models
OV_Model = NBC_OV(args.trainfilename, args.testfilename, args.smoothing)
FV_Model = NBC_FV(args.trainfilename, args.testfilename, args.smoothing)

# Runs models based on command line arguments
if(args.nbversion == 'OV'):
  print("Running only NBC_OV...")
  OV_Model.run()

elif(args.nbversion == 'FV'):
  print("Running only NBC_FV...")
  FV_Model.run()

else:
  print("Running NBC_OV and NBC_FV...")
  OV_Model.run()
  FV_Model.run()
