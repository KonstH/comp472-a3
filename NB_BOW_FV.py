from utils import getFV, split_into_classes, getWords, get_tweets
import math

def BOW_FV(trainf_name, testf_name, smoothing_factor):
  # Capture the vocabulary of the tweets file provided
  vocab, length_vocab = getFV(trainf_name)

  # Separate the tweets from the training set into all classes
  tweets_labeled_yes, tweets_labeled_no = split_into_classes(trainf_name)

  all_train_tweets = get_tweets(trainf_name)
  all_test_tweets = get_tweets(testf_name)

  # Calculate priors
  prior_yes = len(tweets_labeled_yes) / len(all_train_tweets)
  prior_no = len(tweets_labeled_no) / len(all_train_tweets)

  # Calculate conditionals
  words_in_yes, nb_words_in_yes = getWords(tweets_labeled_yes)
  words_in_no, nb_words_in_no = getWords(tweets_labeled_no)

  smoothed_nb_words_yes = nb_words_in_yes + (smoothing_factor * length_vocab)
  smoothed_nb_words_no = nb_words_in_no + (smoothing_factor * length_vocab)

  conditionals_yes = { word:((words_in_yes.count(word) + smoothing_factor)/(smoothed_nb_words_yes)) for word in vocab }
  conditionals_no = { word:((words_in_no.count(word) + smoothing_factor)/(smoothed_nb_words_no)) for word in vocab }

  # Go through entire training set tweets and classify them based on predicted class
  tweet_scores_yes = []
  for tweet in all_test_tweets:
    words_in_tweet = tweet[1].lower().split(' ')
    tweet_score_yes = math.log10(prior_yes)
    for word in words_in_tweet:
      if word in conditionals_yes:
        tweet_score_yes += math.log10(conditionals_yes[word])
    tweet_scores_yes.append(tweet_score_yes)

  tweet_scores_no = []
  for tweet in all_test_tweets:
    words_in_tweet = tweet[1].lower().split(' ')
    tweet_score_no = math.log10(prior_no)
    for word in words_in_tweet:
      if word in conditionals_no:
        tweet_score_no += math.log10(conditionals_no[word])
    tweet_scores_no.append(tweet_score_no)

  # Create list of tuples with the original training set tweet IDs and their predicted class
  final_scores = []
  for i, tweet in enumerate(all_test_tweets):
    if(tweet_scores_yes[i] > tweet_scores_no[i]):
      final_scores.append((tweet[0], tweet_scores_yes[i], 'yes'))
    else:
      final_scores.append((tweet[0], tweet_scores_no[i], 'no'))
  
  # Export the tweet ids with their predicted class
  with open('trace_NB-BOW-FV.txt', 'w') as f:
    for i, tweet in enumerate(final_scores):
      if (tweet[2] == all_test_tweets[i][2]):
        outcome = 'correct'
      else:
        outcome = 'wrong'

      f.write(str(tweet[0]) + ' ' + "{:e}".format(tweet[1]) + ' ' + str(tweet[2]) + ' ' + str(all_test_tweets[i][2]) + ' ' + outcome)
      f.write('\n')
      
  f.close()
