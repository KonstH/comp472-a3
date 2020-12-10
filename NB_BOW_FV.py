from utils import getFV, split_into_classes, getWords
import math

def BOW_FV(f_name, smoothing_factor):
  # Capture the vocabulary of the tweets file provided
  vocab, length_vocab = getFV(f_name)

  # Separate the tweets from the training set into all classes
  all_tweets, tweets_labeled_yes, tweets_labeled_no = split_into_classes(f_name)

  # Calculate priors
  prior_yes = len(tweets_labeled_yes) / len(all_tweets)
  prior_no = len(tweets_labeled_no) / len(all_tweets)

  # Calculate conditionals
  words_in_yes, nb_words_in_yes = getWords(tweets_labeled_yes)
  words_in_no, nb_words_in_no = getWords(tweets_labeled_no)

  smoothed_nb_words_yes = nb_words_in_yes + (smoothing_factor * length_vocab)
  smoothed_nb_words_no = nb_words_in_no + (smoothing_factor * length_vocab)

  conditionals_yes = { word:((words_in_yes.count(word) + smoothing_factor)/(smoothed_nb_words_yes)) for word in vocab }
  conditionals_no = { word:((words_in_no.count(word) + smoothing_factor)/(smoothed_nb_words_no)) for word in vocab }

  # Go through entire training set tweets and classify them based on predicted class
  tweet_scores_yes = []
  for tweet in all_tweets:
    words_in_tweet = tweet[1].lower().split(' ')
    tweet_score_yes = math.log10(prior_yes)
    for word in words_in_tweet:
      tweet_score_yes += math.log10(conditionals_yes[word])
    tweet_scores_yes.append(tweet_score_yes)

  tweet_scores_no = []
  for tweet in all_tweets:
    words_in_tweet = tweet[1].lower().split(' ')
    tweet_score_no = math.log10(prior_no)
    for word in words_in_tweet:
      tweet_score_no += math.log10(conditionals_no[word])
    tweet_scores_no.append(tweet_score_no)

  # Create list of tuples with the original training set tweet IDs and their predicted class
  final_scores = []
  for i, tweet in enumerate(all_tweets):
    if(tweet_scores_yes[i] > tweet_scores_no[i]):
      final_scores.append((tweet[0], 'yes'))
    else:
      final_scores.append((tweet[0], 'no'))
  
  # Export the tweet ids with their predicted class
  with open('predicted.txt', 'w') as f:
    for i in final_scores:
      f.write(str(i))
      f.write('\n')

  # Export the tweet ids with their actual class
  with open('actual.txt', 'w') as f:
    for i in all_tweets:
      f.write(str((i[0], i[2])))
      f.write('\n')
      
  f.close()
