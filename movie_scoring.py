from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
import operator
import math
import numpy as np
import csv
import re

class MovieScoring(object): 
	
	def __init__(self):
		# dictionaries = self.getMovieDictionaries()
		# self.movie_to_genre = dictionaries[0]
		# self.movie_to_categories = dictionaries[1]
		# self.movie_to_attributes = dictionaries[2]
		# self.movie_to_summaries = dictionaries[3]
		# self.movie_to_cast = dictionaries[4]
		# self.wiki_links = dictionaries[5]

		with open("wiki_links.txt", "r") as output:
			self.wiki_links = json.load(output)
			output.close()
		with open("movie_genre.txt", "r") as output:
			self.movie_to_genre = json.load(output)
			output.close()
		with open("movie_categories.txt", "r") as output:
			self.movie_to_categories = json.load(output)
			output.close()
		with open("movie_attributes.txt", "r") as output:
			self.movie_to_attributes = json.load(output)
			output.close()
		# with open("movie_summaries.txt", "r") as output:
		# 	self.movie_to_summaries = json.load(output)
		# 	output.close()
		with open("movie_cast.txt", "r") as output:
			self.movie_to_cast = json.load(output)
			output.close()

		#List of all movies
		self.all_movies = list(self.movie_to_genre.keys())

		#Initialize the Cosine parts
		# self.movies_tokens = self.token(self.movie_to_summaries)
		# self.inv_idx = self.buildInvertedIndex(self.movies_tokens)
		# self.idf = self.compute_idf(self.inv_idx, len(self.movie_to_summaries), min_df=18, max_df_ratio=0.05)
		# self.inv_idx = {key: val for key, val in self.inv_idx.items() if key in self.idf}
		# self.doc_norms = self.compute_doc_norms(self.inv_idx, self.idf, len(self.movie_to_summaries))

		with open("inv_idx.txt", "r") as output:
			self.inv_idx = json.load(output)
			output.close()
		
		with open("idf.txt", "r") as output:
			self.idf = json.load(output)
			output.close()
		
		with open("doc_norms.txt", "r") as output:
			self.doc_norms = json.load(output)
			output.close()

	def getCSVs(self):
		with open("inv_idx.txt", "w") as output:
			output.write(json.dumps(self.inv_idx))
			output.close()
		with open("idf.txt", "w") as output:
			output.write(json.dumps(self.idf))
			output.close()
		with open("doc_norms.txt", "w") as output:
			output.write(json.dumps(self.doc_norms))
			output.close()
		
		with open("wiki_links.txt", "w") as output:
			output.write(json.dumps(self.wiki_links))
			output.close()
		with open("movie_genre.txt", "w") as output:
			output.write(json.dumps(self.movie_to_genre))
			output.close()
		with open("movie_categories.txt", "w") as output:
			output.write(json.dumps(self.movie_to_categories))
			output.close()
		with open("movie_attributes.txt", "w") as output:
			output.write(json.dumps(self.movie_to_attributes))
			output.close()
		with open("movie_summaries.txt", "w") as output:
			output.write(json.dumps(self.movie_to_summaries))
			output.close()
		with open("movie_cast.txt", "w") as output:
			output.write(json.dumps(self.movie_to_cast))
			output.close()

	
	def getMovieDictionaries(self):
		# Read in the csv of movies
		with open('list_final_predicted_v2.csv') as file:
			movies = list(csv.DictReader(file))
			file.close()
			# garbage collect
		
		# actors = list(csv.DictReader(open('new_only_cast.csv')))

		# Dictionary with movie title as key and list of genres as values
		movie_to_genre = {}
		# Dictionary with movie title as key and list of categories as values
		movie_to_categories = {}
		# Dictionary with movie title as key and list of attributes as values
		movie_to_attributes = {}
		# Dictionary with movie title as key and list of summaries as values
		movie_to_summaries = {}
		# Dictionary with movie title as key and list of actors as value
		movie_to_cast = {}
		wiki_links = {}

		# Generate dictionaries
		for each_movie in movies:
			#Creates the genre list
			if each_movie["Genres"] == "":
				movie_to_genre[str(each_movie["Title"]).lower()] = []
			else:
				movie_to_genre[str(each_movie["Title"]).lower()] = eval(each_movie["Genres"])
			
			if eval(each_movie["categories"]) != ['']:
				movie_to_categories[str(each_movie["Title"]).lower()] = eval(each_movie["categories"])
			else:
				movie_to_categories[str(each_movie["Title"]).lower()] = []
			
			movie_to_attributes[str(each_movie["Title"]).lower()] = eval(each_movie["attributes"])
			movie_to_summaries[str(each_movie["Title"]).lower()] = (each_movie["Plot"])
			wiki_links[str(each_movie["Title"]).lower()] = (each_movie["Wiki Page"])

			if each_movie["Cast"] == "":
				movie_to_cast[str(each_movie["Title"]).lower()] = []
			else:
				movie_to_cast[str(each_movie["Title"]).lower()] = each_movie["Cast"].lower().split(', ')
		
		return [movie_to_genre, movie_to_categories, movie_to_attributes, movie_to_summaries, movie_to_cast, wiki_links]
		
	def getMovieAndFoodWords(self, user1_movies, user2_movies, user1_keywords, user2_keywords, user1_actors, user2_actors):
		# List of inputted key words
		input_keywords_list = self.getKeywords(user1_keywords, user2_keywords)
		
		# List of all movie names inputted by both users 
		input_movie_list = self.getMovieNames(user1_movies, user2_movies)

		input_movie_list, input_keywords_list = self.cleanInputMovieList(input_movie_list, input_keywords_list, self.all_movies)

		#List of all actros inputted and in inputtted movie -- this can only be done after we have cleaned input movie list
		input_actor_list = self.getActors(user1_actors, user2_actors, input_movie_list, self.movie_to_cast)

		#All the genres in the movies that the user inputted	
		input_movie_genres = self.getMovieGenres (input_movie_list, self.movie_to_genre)

		#GENRE SCORES
		if len(input_movie_genres) == 0:
			genre_score_array = np.zeros((len(self.all_movies)))
		else:
			#All unique genres in the movies that the user inputted	
			unique_input_movie_genres = list(set(input_movie_genres))
			#Genre Scores of All movies
			genre_score_array = self.getGenreScore(unique_input_movie_genres,input_movie_genres, input_movie_list, self.all_movies, self.movie_to_genre)
			genre_score_array = genre_score_array/(max(genre_score_array))

		#KEYWORDS SCORES && KEYWORDS IN TITLE SCORE
		if len(input_keywords_list) == 0:
			keywords_score_array = np.zeros((len(self.all_movies)))
			title_score_array = np.zeros((len(self.all_movies)))
		else:
			keywords_score_array = []
			keywords_dict = self.getCosine(input_keywords_list)

			title_score_array = []
			title_dict = self.getTitleScore(self.all_movies, input_keywords_list)
			p = 0 
			t = 0
			for each_movie in self.all_movies:
				if each_movie in keywords_dict:
					p+=1
					keywords_score_array.append(keywords_dict[each_movie])
				else:
					keywords_score_array.append(0)
				
				if each_movie in title_dict:
					t+=1
					title_score_array.append(title_dict[each_movie])
				else:
					title_score_array.append(0)

			if p > 0:
				keywords_score_array = (np.asarray(keywords_score_array))/(max(keywords_score_array)/2)
			if t > 2:
				title_score_array =  (np.asarray(title_score_array))/(max(title_score_array))

		#ACTOR SCORE
		if len(input_actor_list) == 0:
			actors_score_array = np.zeros((len(self.all_movies)))
		else:
			actors_score_array = []
			actors_dict = self.getActorsScore(self.movie_to_cast, input_actor_list)
			
			a = 0
			for each_movie in self.all_movies:
				if each_movie in actors_dict:
					a += 1
					actors_score_array.append(actors_dict[each_movie]*2)
				else:
					actors_score_array.append(0)

			if a > 2:
				actors_score_array = (np.asarray(actors_score_array))/(max(actors_score_array))
		

		#TOTAL SCORE
		total_score = genre_score_array+keywords_score_array+title_score_array+actors_score_array
		#DEFAULT MOVIE
		if (np.sum(total_score)<=2):
			return ["Brave", ['Cake'], ['romantic'], "https://en.wikipedia.org/wiki/Brave_(2012_film)",
					"Inside Out", [], ['romantic'], "https://en.wikipedia.org/wiki/Inside_Out_(2015_film)",
					"The Legend of Tarzan", ['British', ' Indian', ' African', ' Belgian'], ['dinner'], "https://en.wikipedia.org/wiki/The_Legend_of_Tarzan_(film)"]
		
		total_score = ((np.asarray(total_score))/(max(total_score)))*5
		total_score_rank =  np.argsort(total_score)

		#FINDING BEST MOVIE FROM ALL RANKED MOVIES
		movie = []
		#counter for going from the back of the list
		i = 1
		#counter for only having 3 movies returned
		x = 0 
		while i<len(self.all_movies) and x<3:
			index_movie = total_score_rank[len(self.all_movies)-i]
			if self.checkInputMovie(self.all_movies[index_movie], input_movie_list):
				movie_name = self.all_movies[index_movie]
				movie.append(movie_name)
				movie.append(self.movie_to_categories[movie_name])
				movie.append(self.movie_to_attributes[movie_name])
				movie.append(self.wiki_links[movie_name])
				x +=1
			i += 1

		# #prints the total scores of all movie (use to check)
		# for i in range(0,len(self.all_movies)):
		# 	index_movie = total_score_rank[i]
		# 	print(self.all_movies[index_movie], total_score[index_movie])
		
		return movie
		#return["","",""]

	def checkInputMovie(self, movie, input_movies):
		for mov in input_movies:
			if (movie in mov) or (mov in movie):
				return False
		return True

	def getKeywords(self, user1_keywords,user2_keywords):
		""" Give the inputted keywords of both users returns a list with all the keywords
		"""
		if len(user1_keywords) == 0 and len(user2_keywords) == 0 :
			return []
		elif len(user1_keywords) == 0:
			return re.split('\s[,]\s|\s',user2_keywords.lower())
		elif len(user2_keywords) == 0 :
			return re.split('\s[,]\s|\s',user1_keywords.lower())
		else:
			keywords = user1_keywords.lower()+ ", " + user2_keywords.lower()
			return re.split('\s[,]\s|\s',keywords)


	def getMovieNames(self, user1_movies, user2_movies):
		"""Given a the user input of movie names (which are split by commas)
		the function returns the movie names as a list
		
		Parameters:
		movie_name: a string that has one (or multiple) movie titles
		
		Returns: a list with all the movie titles"""
		if len(user1_movies) == 0 and len(user2_movies) == 0 :
			return []
		elif len(user1_movies) == 0:
			return user2_movies.lower().split(', ')
		elif len(user2_movies) == 0 :
			return user1_movies.lower().split(', ')
		else:
			movies = user1_movies+ ", " + user2_movies
			return movies.lower().split(', ')

	def cleanInputMovieList(self, input_movie_list, input_keywords_list, all_movies):
		"""
		Check if all inputted movies are in the db otherwise appends them to keywords
		"""
		cleaned_input_movie_list = []
		for each_movie in input_movie_list:
			if each_movie in all_movies:
				cleaned_input_movie_list.append(each_movie)
			input_keywords_list += each_movie.split(' ')
		return cleaned_input_movie_list, input_keywords_list

	def getActors(self, user1_actors, user2_actors, input_movie_list, movie_to_cast):
		"""
		Gets a list of actors that are inputted and are in the inputted movies
		"""
		movie_actors = ['leonardo dicaprio', 'kate winslet', 'gloria stuart', 'will smith', 'salma hayek', 'bai ling', 'nick stahl', 'm.c. gainey', 'carolyn hennesy', 'tom hanks', 'eddie deezen', 'peter scolari', 'j.k. simmons', 'james franco', 'kirsten dunst', 'jim broadbent', 'kiran shah', 'shane rangi', 'naomi watts', 'thomas kretschmann', 'evan parke', 'johnny depp', 'orlando bloom', 'jack davenport', 'christopher lee', 'eva green', 'kristin scott thomas', 'peter dinklage', 'pierfrancesco favino', 'damián alcázar', 'harrison ford', 'ray winstone', 'robert downey jr.', 'jeff bridges', 'jon favreau', 'robin wright', 'colin firth', 'gary oldman', 'joseph gordon-levitt', 'dennis quaid', 'leo howard', 'alan rickman', 'daniel radcliffe', 'rupert grint', 'amy poehler', 'rainn wilson', 'stephen colbert', 'christian bale', 'bryce dallas howard', 'common', 'glenn morshower', 'kevin dunn', 'ramon rodriguez', 'john ratzenberger', 'delroy lindo', 'jess harnell', 'anne hathaway', 'tom hardy', 'scarlett johansson', 'brad garrett', 'donna murphy', 'don rickles', 'joe mantegna', 'eddie izzard', 'chloë grace moretz', 'sam claflin', 'stephen graham', 'lester speight', 'liam neeson', 'alexander skarsgård', 'tadanobu asano', 'kelly macdonald', 'julie walters', 'michael stuhlbarg', 'nicole scherzinger', 'don cheadle', 'henry cavill', 'christopher meloni', 'harry lennix', 'tim holmes', 'mila kunis', 'chris evans', 'hayley atwell', 'jennifer lawrence', 'hugh jackman', 'judy greer', 'kodi smit-mcphee', 'bradley cooper', 'vin diesel', 'djimon hounsou', 'matthew mcconaughey', 'mackenzie foy', 'channing tatum', 'eddie redmayne', 'jason statham', 'paul walker', 'chris hemsworth', 'chris bauer', 'thomas robinson', 'omar sy', 'mindy kaling', 'phyllis smith', 'lauren cohan', 'alan d. purwin', 'bill murray', 'garry shandling', 'michael fassbender', 'tye sheridan', 'christoph waltz', 'casper crump', 'robin atkin downes', 'ike barinholtz']
		actors_input = []
		if len(user1_actors) != 0:
			actors_input += user1_actors.lower().split(', ')
		if len(user2_actors) != 0 :
			actors_input += user2_actors.lower().split(', ')

		actors = []
		for each in actors_input:
			if each in movie_actors:
				actors.append(each)

		for each_movie in input_movie_list:
			if each_movie in movie_to_cast:
				actors += movie_to_cast[each_movie]
		return actors

	def getMovieGenres (self, input_movie_list, movie_to_genre):
		"""Returns a list of all the genres in all of the inputted movies"""
		input_movie_genres = []
		for each_movie in input_movie_list:
			input_movie_genres = input_movie_genres +  movie_to_genre[each_movie]
		return input_movie_genres

	def getGenreScore(self, unique_input_movie_genres,input_movie_genres,input_movie_list, all_movies, movie_to_genre):
		"""Return a the genre score"""
		if input_movie_genres == "":
			return 0

		#Create a numpy matric that is movie_genres by all movies
		genresBYmovies = np.zeros((len(unique_input_movie_genres), len(all_movies)))

		#Set a position 1 if the genre appears in a certain movie
		for m in range(0,len(all_movies)):
			movie_genres = movie_to_genre[all_movies[m]]
			for g in movie_genres:
				if g in unique_input_movie_genres:
					genresBYmovies[unique_input_movie_genres.index(g)][m]=1

		genre_count = []

		#Counts the number of times a genre was inputted
		for i in range(0,len(unique_input_movie_genres)):
			genre_count.append(input_movie_genres.count(unique_input_movie_genres[i]))

		movie_scoring = np.sum((genresBYmovies.transpose())*genre_count, axis=1)
		return movie_scoring

	#Cosine Similarity

	def token(self, movie_summaries):
		result = {}
		for mov in movie_summaries:
			# temp = re.sub(r'[^\w\s]','',movie_summaries[mov].lower())
			temp = movie_summaries[mov].lower().split(' ')
			li = []
			for t in temp:
				if '(' not in t and ')' not in t and '.' not in t and ',' not in t and len(t) > 2:
					li.append(t)
			result[mov] = li
		return result

	def tokenize_plot(self, movie_summaries):
		result = {}
		#use treebank tokenizer?
		tokenizer = TreebankWordTokenizer()
		for mov in movie_summaries:
			result[mov] = tokenizer.tokenize(movie_summaries[mov].lower())
		return result
		
	#Build the inverted index - term - list of movies it is in
	def buildInvertedIndex(self, movie_summaries):
		#input should be the a dictionary with key as movie title and value as movie_summaries
		#takes in the list of words that the user inputted
		"""
		This just builds an inverted index with the movie summaries. {word: [(movie title, count of word), ...]}
		
		"""
		inverted_index = {}
		for mov in movie_summaries:
			words = Counter(movie_summaries[mov]) #dictionary of word and its count
			for word in words.keys():
				if word in inverted_index.keys():
					inverted_index[word].append((mov, words[word]))
				else:
					inverted_index[word] = [(mov, words[word])]
		return inverted_index
		#inverted index only has the query term words

	def compute_idf(self, inv_idx, n_docs, min_df=18, max_df_ratio=0.05):
		"""
		This just builds computes idf for the movie summaries. {word: idf value}
		
		"""
		idf_dict = {}
		for key in inv_idx.keys():
			ratio = len(inv_idx[key])/n_docs
			if ((len(inv_idx[key]) >= min_df) and (ratio <= max_df_ratio)):
				idf = [n_docs/(1 + len(inv_idx[key]))]
				idf_dict[key] = np.log2(idf)[0]
		return idf_dict

	def compute_doc_norms(self, index, idf, n_docs):
		"""
		norms: {movie title: norm of movie summary}
		"""
		norms = {}
		sum = 0
		for word in idf.keys():
			words = index[word]
			for t in words:
				power = math.pow(idf[word]*t[1], 2)
				if t[0] in norms:
					norms[t[0]] += power
				else:
					norms[t[0]] = power
		
		for n in norms:
			norms[n] = math.sqrt(norms[n])
		return norms

	def cosine_score(self, query_words, inv_index, idf, doc_norms):
		"""
		results, list of tuples (score, movie title) Sorted list of results such that the first element has
		the highest score.
		"""

		results = {}

		#get the counts for each word in user input/query
		count_query = {} #{word: count}
		for term in query_words:
			if term in count_query.keys():
				count_query[term] += 1
			else:
				count_query[term] = 1
		
		total = 0
		#WHAT TO DO HERE SINCE MOST OFTEN IT WONT HAVE IDF?
		#If the query has no words that relate, then it would just be 0?
		query_numbers = {} #{word in query: tf*idf, ...} 
		for key in count_query.keys():
			if key in idf.keys():
				top = count_query[key] * idf[key]
				query_numbers[key] = top
				total += top**2

		query_den = math.sqrt(total)
		
		doc_dict = {} #{movie title: {word: count in movie, word: count in movie ...}
		for word in inv_index.keys():
			for tup in inv_index[word]:
				if tup[0] in doc_dict.keys():
					val = doc_dict[tup[0]]
					val[word] = tup[1]
					doc_dict[tup[0]] = val              
				else:
					val = {}
					val[word] = tup[1]
					doc_dict[tup[0]] = val
		
		for document in doc_dict.keys():
			num = 0
			for word in query_numbers.keys():
				if word in doc_dict[document].keys():
					num += query_numbers[word]*doc_dict[document][word]*idf[word]
					den = query_den * doc_norms[document]
					results[document] = num/den
					# final_tuple = (num/den, document)
					# results += [final_tuple]
		
		# results.sort(key=lambda x: x[0])
		# results.reverse()
		return results

	def getCosine(self, query_words):
		""" Main function for cosine! """
		return self.cosine_score(query_words, self.inv_idx, self.idf, self.doc_norms)
		
	def getAllMovies(self):
		"""Used for drop down for movie.
		
		Returns: a list with all the movie titles"""

		movies = list(csv.DictReader(open('new.csv')))
		all_movies = []
		for each_movie in movies:
			all_movies.append(str(each_movie["Title"].lower()))
		
		with open("all_movies.txt", "w") as output:
			for mov in all_movies:
				output.write(mov+ '\n')

	def getActorsScore(self, movie_to_cast, inputted_movie_actors):
		"""
		Assumes parameter is a list of words that are divided based on commas (list of actors), all lowercase.
		
		Returns: sorted list: [(movie title, score), ...]. Will be empty if no inputed actors
		match any of the movie cast members.

		"""
		movie_act_dict = {}

		for each_movie in movie_to_cast:
			for each_actor in movie_to_cast[each_movie]:
				for each_inputted_actor in inputted_movie_actors:
					if each_inputted_actor == each_actor:
						if each_movie in movie_act_dict:
							movie_act_dict[each_movie] += 1
						else:
							movie_act_dict[each_movie] = 1

		# for mov in movies_to_cast:
		# 	for actor in movie_actors:
		# 		if mov in movie_act_dict:
		# 			movie_act_dict[mov] += 1
		# 		else:
		# 			movie_act_dict[mov] = 1
		# movie_act_dict_sorted = sorted(movie_act_dict.items(), key=operator.itemgetter(1))
		return movie_act_dict

	def getTitleScore(self, movie_titles, keywords):
		"""
		Parameter: list of movie titles, list of user inputed keywords all lowercase
		
		Returns: sorted list: [(movie title, score), ...]. Will be empty if no keywords
						match the words in the movie titles.
		"""
		title_dict = {}
		for mov in movie_titles:
			#tokenize movie title based on space
			title = mov.lower().split(" ")
			for word in keywords:
				if word in title:
					if mov in title_dict:
						title_dict[mov] += 1
					else:
						title_dict[mov] = 1
		return title_dict

	# def test(self):
	# 	"""
	# 	Used to test that cosine works. Cosine will return an empty list if there's
	# 	no matching.
	# 	"""

		# all_movies = {}

		# with open('new_cast.csv', mode='r') as csv_file:
		# 	csv_reader = csv.DictReader(csv_file)
		# 	for row in csv_reader:
		# 		if row["Cast"] != "":
		# 			all_movies[row["Title"].lower()] = row["Cast"]

		# a = ['maleficent', 'guardians of the galaxy', 'the legend of tarzan']
		# print(getTitleScore(a, []))

		
		# i = getCosine(all_movies, ["christmas", "princess"])
		# print(i[0])
		# print(all_movies[i[0][1]])
		
		# with open("possible_inputs.txt", "w") as output:
		# 	for mov in i:
		# 		output.write(mov+ '\n')

# print("Method")
# movieClass = MovieScoring()
# for mov in movieClass.movie_to_cast:
# 	if movieClass.movie_to_cast[mov] != []:
# 		print(mov)
# movieClass.getCSVs()
# print(len(movieClass.all_movies))
# print(len(movieClass.inv_idx))
# print(len(movieClass.movie_to_cast))


#CASES:
#check that the movie returned isnt a movie inputted
#pre-computed inverted index
#return a default movie!

#make sure that movie inputed is in database --> make sure its a dropdown list of movies

#movie if it doesn't have a genre --> genre score should be 0
#movie keywords - title and summary have cosine sim score of 0 

#if genre + cosine sim score = 0 --> default movie

#for food, if there is no restaurant, display popcorn is always a good option

# Questions:
# movie categories/attributes if empty
# how are we dividing user input (actors have to be full names & exact)

#Create a class to pre-compute the inverted index