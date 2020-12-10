import csv

"""
  Splits tweets into lists based on class they belong to
"""
def split_into_classes(fname):
  all_tweets = []
  all_yesses = []
  all_nos = []

  with open(fname, "r") as f:
    file = csv.reader(f, delimiter='\t')
    next(file)  # skip first line
    for line in file:
      all_tweets.append((line[0],line[1],line[2]))
      if(line[2] == 'yes'):
        all_yesses.append((line[0],line[1],line[2]))
      else:
        all_nos.append((line[0],line[1],line[2]))

  return(all_tweets, all_yesses, all_nos)

"""
  Takes tsv file and computes the vocabulary of all its tweet contents
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
        #words.append(word.replace('.', '').replace(',', '').replace(':', '').replace('!', '').replace('?', '').replace(';', ''))

  vocabulary = { i:words.count(i) for i in set(words) }
  vocabulary_size = len(vocabulary)

  return(vocabulary, vocabulary_size)

"""
  Takes tsv file and computes the filtered vocabulary of all its tweet contents
  The words which only appear once are not included.
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
        #words.append(word.replace('.', '').replace(',', '').replace(':', '').replace('!', '').replace('?', '').replace(';', ''))

  vocabulary = { word:words.count(word) for word in set(words) }

  # Filter vocabulary to only contain words which appear at least twice
  for word, freq in vocabulary.copy().items():
    if(freq == 1):
      vocabulary.pop(word)

  return(vocabulary)

"""
  Returns all words from the tweets labeled in the passed category
"""
def getWords(category):
  words_in_category = []

  for tweet in category:
    words_in_tweet = tweet[1].lower().split(' ')
    for word in words_in_tweet:
      words_in_category.append(word)

  nb_words_in_category = len(words_in_category)

  return (words_in_category, nb_words_in_category)