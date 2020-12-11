from NB_BOW_OV import BOW_OV
from NB_BOW_FV import BOW_FV
from utils import get_tweets

trainf_name = "covid_training.tsv"
testf_name = "covid_test_public.tsv"

BOW_OV(trainf_name, testf_name, 0.01)
BOW_FV(trainf_name, testf_name, 0.01)
