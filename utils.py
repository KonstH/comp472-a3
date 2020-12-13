import csv

"""
  Splits tweets into lists based on the class they belong to
"""
def split_into_classes(fname):
  all_yesses = []
  all_nos = []

  with open(fname, "r") as f:
    file = csv.reader(f, delimiter='\t')
    next(file)  # skip first line
    for line in file:
      if(line[2] == 'yes'):
        all_yesses.append((line[0],line[1],line[2]))
      else:
        all_nos.append((line[0],line[1],line[2]))

  return(all_yesses, all_nos)

"""
  Returns all tweets (id, content, class) from the given tsv file
"""
def get_tweets(fname):
  tweets = []

  with open(fname, "r") as f:
    file = csv.reader(f, delimiter='\t')
    next(file)  # skip first line
    for line in file:
      tweets.append((line[0],line[1],line[2]))

  return(tweets)

"""
  Takes tsv file, computes and returns the unfiltered vocabulary, along with its length, based on the content of its tweets.
"""
def getOV(fname):
  words = []

  with open(fname, "r") as f:
    file = csv.reader(f, delimiter='\t')
    next(file)  # skip first line
    for line in file:
      words_in_line = line[1].lower().split(' ')
      for word in words_in_line:
        words.append(word)

  vocabulary = { i:words.count(i) for i in set(words) }
  vocabulary_size = len(vocabulary)

  return(vocabulary, vocabulary_size)

"""
  Takes tsv file, computes and returns the filtered vocabulary, along with its length, based on the content of its tweets.
  The words which only appear once are filtered out of the vocabulary.
"""
def getFV(fname):
  words = []

  with open(fname, "r") as f:
    file = csv.reader(f, delimiter='\t')
    next(file)  # skip first line
    for line in file:
      words_in_line = line[1].lower().split(' ')
      for word in words_in_line:
        words.append(word)

  vocabulary = { word:words.count(word) for word in set(words) }

  # Filter vocabulary to only contain words which appear at least twice
  for word, freq in vocabulary.copy().items():
    if(freq == 1):
      vocabulary.pop(word)
  
  vocabulary_size = len(vocabulary)

  return(vocabulary, vocabulary_size)

"""
  Returns all words from the tweets belonging to the given category (yes / no), along with their # of occurences
"""
def getWords(category):
  words_in_category = []

  for tweet in category:
    words_in_tweet = tweet[1].lower().split(' ')
    for word in words_in_tweet:
      words_in_category.append(word)

  nb_words_in_category = len(words_in_category)

  return (words_in_category, nb_words_in_category)

if __name__ == "__main__":
  print("\nThis file contains util functions used throughout the project.")
