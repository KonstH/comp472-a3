from NB_BOW_OV import BOW_OV
from NB_BOW_FV import BOW_FV
import argparse

# Sets up the arguments that can be passed in the terminal
parser = argparse.ArgumentParser(description="Use Naive Bayes Classifiers to predict tweet information verifiableness")
parser.add_argument('-trn', '--trainfilename', type=str, default='covid_training.tsv', metavar='', help="Name of training input file (default: covid_training.tsv)")
parser.add_argument('-tst', '--testfilename', type=str, default='covid_test_public.tsv', metavar='', help="Name of test input file (default: covid_test_public.tsv)")
parser.add_argument('-s', '--smoothing', type=float, default=0.01, metavar='', help="Smoothing value (default: 0.01)")
parser.add_argument('-v', '--nbversion', type=str, metavar='', help="Version of NB Classifier to run. Options: OV or FV (default: runs both versions)")
args = parser.parse_args()

trainf_name = "covid_training.tsv"
testf_name = "covid_test_public.tsv"

if(args.nbversion == 'OV'):
    print("Running BOW_OV...")
    BOW_OV(args.trainfilename, args.testfilename, args.smoothing)
elif(args.nbversion == 'FV'):
    print("Running BOW_FV...")
    BOW_FV(args.trainfilename, args.testfilename, args.smoothing)
else:
    print("Running BOW_OV and BOW_FV...")
    BOW_OV(args.trainfilename, args.testfilename, args.smoothing)
    BOW_FV(args.trainfilename, args.testfilename, args.smoothing)
