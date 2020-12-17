from utils import getOV, split_into_classes, getWords, get_tweets
import math

class NBC_OV:
  def __init__(self, trainf_name, testf_name, smoothing_factor):
    self.vocab, self.vocabLength = getOV(trainf_name)
    self.train_tweets = get_tweets(trainf_name)
    self.tweets_labeled_yes, self.tweets_labeled_no = split_into_classes(trainf_name)
    self.test_tweets = get_tweets(testf_name)
    self.smoothing_factor = smoothing_factor

  # Returns the prior for each class
  def getPriors(self):
    prior_yes = len(self.tweets_labeled_yes) / len(self.train_tweets)
    prior_no = len(self.tweets_labeled_no) / len(self.train_tweets)

    return(prior_yes, prior_no)
  
  # Returns the conditionals for all the words in the vocabulary
  def getConditionals(self):
    words_in_yes, nb_words_in_yes = getWords(self.tweets_labeled_yes)
    words_in_no, nb_words_in_no = getWords(self.tweets_labeled_no)

    smoothed_nb_words_yes = nb_words_in_yes + (self.smoothing_factor * self.vocabLength)
    smoothed_nb_words_no = nb_words_in_no + (self.smoothing_factor * self.vocabLength)

    conditionals_yes = { word:((words_in_yes.count(word) + self.smoothing_factor)/(smoothed_nb_words_yes)) for word in self.vocab }
    conditionals_no = { word:((words_in_no.count(word) + self.smoothing_factor)/(smoothed_nb_words_no)) for word in self.vocab }

    return(conditionals_yes, conditionals_no)

  # Returns the predictions for all tweets in the test set
  def getPredictions(self, prior_yes, prior_no, cond_yes, cond_no):

    # Computes the scores of all test tweets given class "yes"
    tweet_scores_yes = []
    for tweet in self.test_tweets:
      words_in_tweet = tweet[1].lower().split(' ')
      tweet_score_yes = math.log10(prior_yes)
      for word in words_in_tweet:
        if word in cond_yes:
          tweet_score_yes += math.log10(cond_yes[word])
      tweet_scores_yes.append(tweet_score_yes)

    # Computes the scores of all test tweets given class "no"
    tweet_scores_no = []
    for tweet in self.test_tweets:
      words_in_tweet = tweet[1].lower().split(' ')
      tweet_score_no = math.log10(prior_no)
      for word in words_in_tweet:
        if word in cond_no:
          tweet_score_no += math.log10(cond_no[word])
      tweet_scores_no.append(tweet_score_no)

    # Create list of tuples containing the test tweet IDs, their final score and their predicted class
    predictions = []
    for i, tweet in enumerate(self.test_tweets):
      if(tweet_scores_yes[i] > tweet_scores_no[i]):
        predictions.append((tweet[0], tweet_scores_yes[i], 'yes'))
      else:
        predictions.append((tweet[0], tweet_scores_no[i], 'no'))

    return(predictions)
  
  # Returns the accuracy and per-class precision, recall and f1-measure
  def getMetrics(self, predictions, actual):

    # Accuracy calculation
    correct = 0
    for i, tweet in enumerate(predictions):
      if (tweet[2] == actual[i][2]):
        correct += 1
    acc = correct / len(predictions)

    # Precision calculations
    tp = 0
    tp_and_fp = 0
    for i, tweet in enumerate(predictions):
      if(tweet[2] == 'yes' and tweet[2] != actual[i][2]):
        tp_and_fp += 1
      elif (tweet[2] == 'yes' and tweet[2] == actual[i][2]):
        tp += 1
        tp_and_fp += 1
    if(tp_and_fp == 0):
      yes_p = 0
    else:
      yes_p = (tp / tp_and_fp)

    tp = 0
    tp_and_fp = 0
    for i, tweet in enumerate(predictions):
      if(tweet[2] == 'no' and tweet[2] != actual[i][2]):
        tp_and_fp += 1
      elif (tweet[2] == 'no' and tweet[2] == actual[i][2]):
        tp += 1
        tp_and_fp += 1
    if(tp_and_fp == 0):
      no_p = 0
    else:
      no_p = (tp / tp_and_fp)

    # Recall calculations
    tp = 0
    tp_and_fn = 0
    for i, tweet in enumerate(predictions):
      if(actual[i][2] == 'yes' and tweet[2] != actual[i][2]):
        tp_and_fn += 1
      elif(actual[i][2] == 'yes' and tweet[2] == actual[i][2]):
        tp += 1
        tp_and_fn += 1
    if(tp_and_fn == 0):
      yes_r = 0
    else:
      yes_r = (tp / tp_and_fn)    

    tp = 0
    tp_and_fn = 0
    for i, tweet in enumerate(predictions):
      if(actual[i][2] == 'no' and tweet[2] != actual[i][2]):
        tp_and_fn += 1
      elif(actual[i][2] == 'no' and tweet[2] == actual[i][2]):
        tp += 1
        tp_and_fn += 1
    if(tp_and_fn == 0):
      no_r = 0
    else:
      no_r = (tp / tp_and_fn)    

    # F1-measure calculations
    if(yes_p == 0 or yes_r == 0):
      yes_f = 0
    else:
      yes_f = (2*yes_p*yes_r)/(yes_p + yes_r)

    if(no_p == 0 or no_r == 0):
      no_f = 0
    else:
      no_f = (2*no_p*no_r)/(no_p + no_r)

    # Rounding results to 4 decimals and converting to strings
    acc = '{:0.4f}'.format(acc)
    yes_p = '{:0.4f}'.format(yes_p)
    yes_r = '{:0.4f}'.format(yes_r)
    yes_f = '{:0.4f}'.format(yes_f)
    no_p = '{:0.4f}'.format(no_p)
    no_r = '{:0.4f}'.format(no_r)
    no_f = '{:0.4f}'.format(no_f)

    return (acc, yes_p, yes_r, yes_f, no_p, no_r, no_f)

  # Exports the Model's trace into file
  def exportTrace(self, predictions):
    with open('trace_NB-BOW-OV.txt', 'w') as f:
      for i, tweet in enumerate(predictions):
        if (tweet[2] == self.test_tweets[i][2]):
          outcome = 'correct'
        else:
          outcome = 'wrong'

        f.write(str(tweet[0]) + '  ' + str(tweet[2]) + '  ' + format(tweet[1], ".2E") + '  ' + str(self.test_tweets[i][2]) + '  ' + outcome)
        f.write('\n')

    f.close()
    print('NB BOW OV trace exported to file: \"trace_NB-BOW-OV.txt\"')
  
  # Exports the Model's performance metrics into file
  def exportMetrics(self, predictions, actual):
    acc, yes_p, yes_r, yes_f, no_p, no_r, no_f = self.getMetrics(predictions, actual)
    
    with open('eval_NB-BOW-OV.txt', 'w') as f:
      f.write(acc + '\n')
      f.write(yes_p + '  ' + no_p + '\n')
      f.write(yes_r + '  ' + no_r + '\n')
      f.write(yes_f + '  ' + no_f)
    
    f.close()
    print('NB BOW OV metrics exported to file: \"eval_NB-BOW-OV.txt\"')
  
  # Runs the Model and exports results into files
  def run(self):
    prior_yes, prior_no = self.getPriors()
    cond_yes, cond_no = self.getConditionals()
    predictions = self.getPredictions(prior_yes, prior_no, cond_yes, cond_no)
    self.exportTrace(predictions)
    self.exportMetrics(predictions, self.test_tweets)

if __name__ == "__main__":
  print("\nThis is the NB_BOW_OV class. It defines a Naive Bayes Classifier using an unfiltered vocabulary.")
