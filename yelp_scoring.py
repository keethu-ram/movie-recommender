import numpy as np
import json
import operator
import csv
import re
import pyzipcode 

zcdb = pyzipcode.ZipCodeDatabase()

class YelpScoring(object): 

  def __init__(self):
    self.data = self.create_python_dict()
    self.restaurants = self.data[0]
    self.restaurant_dict = self.data[1]
    self.zipcodes = self.data[2]


  def create_python_dict(self):
    """ Converts json (the yelp data set) to a list of restaurants and python 
    dictionary to be used throughout the program.
    
    Returns: a tuple (list with each element as restaurant dictionary, python 
    dictionary with business id of restaurant as key and the restaurant dictionary as value)
    """
    
    tempList = []
    tempDict = {}
    zipcodes = set()

    with open('yelp-restaurants.txt') as f:
      for jsonObj in f.readlines():
        obj = json.loads(jsonObj)
        zipcodes.add(obj['postal_code'])
        tempList.append(obj) 
        tempDict[obj["business_id"]] = obj
      f.close()  
    
    zipcodes = list(zipcodes)

    return [tempList, tempDict, zipcodes]


  def run(self, mov_attributes, mov_categories, zipcode1, zipcode2):
    """" Runs the main code. If zipcode 2 is None, there will be
    one output of a restaurant, otherwise there will be two separate restaurants.
    
    Parameters: zipcode2 should be None if there is only one user location 
    Returns: dictionary of what is printed on app - restuarant, location, etc. """

    # TODO - what to print if any of the values are empty

    restaurant_locations = self.restaurant_location_by_zip(self.restaurants, zipcode1)
    restaurant_scores, key_words_dict = self.find_restaurants(mov_categories, mov_attributes, self.restaurants)
    #key_words_dict is a dictionary with business id as key and value is the key words that intersect with the movie

    final_result = self.combine_location(restaurant_scores, restaurant_locations)
    #final result is a dictionary with id of restaurant and the score

    if len(final_result) == 0:
        user1_result = [{'restaurant1': "Could not find restaurant :( but popcorn is always a good option! ", 'score1': '', 'city1': '', 'state1': '',
        'matchings': '', 'star': '', 'zipcode': ''}]

    else:
      first_elem = list(final_result.keys())[0] #gets the first element id of the final result
      first_elem_score = final_result[first_elem] #gets the score
      i = key_words_dict[first_elem]

      if len(i) == 0:
        i = ''
      else:
        i = list(i)
      
      user1_result = [{'restaurant1': self.restaurant_dict[first_elem]['name'], 'score1': str(first_elem_score), 
      'city1': self.restaurant_dict[first_elem]['city'], 'state1': self.restaurant_dict[first_elem]['state'],
      'matchings': i, 'star': self.restaurant_dict[first_elem]['stars'], 'zipcode': self.restaurant_dict[first_elem]['postal_code']}]

    if zipcode2 is None: 
      user2_result = []

    else: # Find two restaurants
      restaurant_locations2 = self.restaurant_location_by_zip(self.restaurants, zipcode2)
      restaurant_scores2, key_words_dict2 = self.find_restaurants(mov_categories, mov_attributes, self.restaurants)
      final_result2 = self.combine_location(restaurant_scores2, restaurant_locations2)

      if len(final_result2) == 0:

        user2_result = [{'restaurant2': "Could not find restaurant :( but popcorn is always a good option! ", 'score2': '', 'city2': '', 'state2': '',
        'matchings': '', 'star': '', 'zipcode': ''}]
      
      else:
        first_elem2 = list(final_result2.keys())[0]
        first_elem_score2 = final_result2[first_elem2]
        i2 = key_words_dict2[first_elem2]

        if len(i2) == 0:
          i2 = ''
        else:
          i2 = list(i2)

        user2_result = [{'restaurant2': self.restaurant_dict[first_elem2]['name'], 'score2': str(first_elem_score2), 
        'city2': self.restaurant_dict[first_elem2]['city'], 'state2': self.restaurant_dict[first_elem2]['state'],
        'matchings': i2, 'star': self.restaurant_dict[first_elem2]['stars'], 'zipcode': self.restaurant_dict[first_elem2]['postal_code']}]
      
    return user1_result + user2_result
        
    
  def intersection_fun(self, set1, set2):
    """" Gets the intersection of the two sets.
    Returns: a set of the words in both sets. """

    result = set()
    for word1 in set1:
      for word2 in set2:
        if word1.lower() in word2.lower():
          result.add(word2.title())
        elif word2.lower() in word1.lower():
          result.add(word1.title())
    return result

  def jaccard_sim(self, set1, set2):
    """" Gets the simple Jaccard Similarity between the two sets.
    Returns: a tuple with a simple Jaccard Similarity score for set1 and set2
    and the set of intersecting words. """

    #CHECK - condition when denominator is 0, when BOTH restaurant and movie sets have empty categories or empty attributes
    #fix with smoothing!

    intersection = self.intersection_fun(set1, set2)
    numerator = len(intersection)
    denominator = (len(set1) + len(set2)) - numerator
    return (float(numerator) / (1.0 + float(denominator)), intersection)


  def restaurant_location_by_zip(self, restaurants, zipcode):
    """" Takes in a dictionary of restaurants, a string zipcode.
    Returns: an ordered list of restaurant dictionaries that are near the location,
    where the beginning elements are the closest and the later are farther """

    # Get all zipcodes in different sizes of radius from the input zipcode

    radius_2 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 1)]) # All zipcodes with 2 miles of input
    radius_5 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 2)]) - radius_2   # All zipcodes within 2 < zipcode <= 5
    radius_10 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 3)]) - radius_2 - radius_5 # All zipcodes within 5 < zipcode <= 10
    radius_15 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 4)]) - radius_2 - radius_5 - radius_10
    radius_20 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 5)]) - radius_2 - radius_5 - radius_10 - radius_15
    radius_25 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 10)]) - radius_2 - radius_5 - radius_10 - radius_15 - radius_20
    radius_30 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 15)]) - radius_2 - radius_5 - radius_10 - radius_15 - radius_20 - radius_25
    radius_45 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 20)]) - radius_2 - radius_5 - radius_10 - radius_15 - radius_20 - radius_25 - radius_30
    radius_50 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 25)]) - radius_2 - radius_5 - radius_10 - radius_15 - radius_20 - radius_25 - radius_30 - radius_45
    radius_60 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 30)]) - radius_2 - radius_5 - radius_10 - radius_15 - radius_20 - radius_25 - radius_30 - radius_45 - radius_50

    # radius_5 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 5)])   # All zipcodes within 2 < zipcode <= 5
    # radius_20 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 20)]) - radius_5
    # radius_30 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 30)]) - radius_5 - radius_20
    # radius_50 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 50)]) - radius_5 - radius_20 - radius_30
    # radius_60 = set([z.zip for z in zcdb.get_zipcodes_around_radius(zipcode, 60)]) - radius_5 - radius_20 - radius_30 - radius_50

    res_same_zip = []
    res_within_2 = []
    res_within_5 = []
    res_within_10 = []
    res_within_15 = []
    res_within_20 = []
    res_within_25 = []
    res_within_30 = []
    res_within_45 = []
    res_within_50 = []
    res_within_60 = []

    for res in restaurants:
      zcode = res["postal_code"]
      if zcode == zipcode:
        res["radius"] = 0
        res_same_zip.append(res)
      elif zcode in radius_2:
        res["radius"] = 2
        res_within_2.append(res)
      elif zcode in radius_5:
        res["radius"] = 5
        res_within_5.append(res)
      elif zcode in radius_10:
        res["radius"] = 10
        res_within_10.append(res)
      elif zcode in radius_15:
        res["radius"] = 15
        res_within_15.append(res)
      elif zcode in radius_20:
        res["radius"] = 20
        res_within_20.append(res)
      elif zcode in radius_25:
        res["radius"] = 25
        res_within_25.append(res)
      elif zcode in radius_30:
        res["radius"] = 30
        res_within_30.append(res)
      elif zcode in radius_45:
        res["radius"] = 45
        res_within_45.append(res)
      elif zcode in radius_50:
        res["radius"] = 50
        res_within_50.append(res)
      elif zcode in radius_60:
        res["radius"] = 60
        res_within_60.append(res)
      
    result = (res_same_zip + res_within_2 + res_within_5 + res_within_10 + res_within_15 + res_within_20 +
    res_within_25 + res_within_30 + res_within_45 + res_within_50 + res_within_60) 

    return result


  def find_restaurants(self, mov_categories, mov_attributes, restaurants):
    """" Gets the score for each restaurant based on its Jaccard Similarity scores.
    
    Returns: a tuple (sorted dictionary of restuarants with id of restaurant and 
    the similarity score to the mov_categories and mov_attributes, dictionary with id as key 
    and value as the list of matching attributes and categories). """

    # TODO - need to redistribute the weight of words/categories/attributes

    result = {}
    output = {}
    for r in restaurants:
      res_categories = set()
      res_attributes = set()
      if r['attributes'] is not None: #used to check if restaurant has this as empty
        res_attributes = r['attributes']
      if r['categories'] is not None:
        res_categories = self.tokenize_categories(r['categories'])
      jac_attribute = self.jaccard_sim(set(mov_attributes), res_attributes)
      jac_cattegory = self.jaccard_sim(set(mov_categories), res_categories)
      sim_attribute = jac_attribute[0]
      sim_category = jac_cattegory[0]

      result[r['business_id']] = sim_attribute + (3 * sim_category)
      output[r['business_id']] = jac_attribute[1].union(jac_cattegory[1])
    
    sorted_dict = {r: result[r] for r in sorted(result, key=result.get, reverse=True)}
    return (sorted_dict, output)

  def combine_location(self, restaurant_scores, restaurant_locations):
    """" Takes in a sorted dictionary of restaurant ids and similarity scores
    and an ordered list of restaurants in a specific location range. 

    Returns: a dictionary of restaurants sorted on the highest matching scores.
    Will only include restaurants from the restaurant locations input.
    """
    # 11 bins for zipcodes - those w radius 0 get 11 points, with radius 2 (next bin) get 10 points
    weight = {0: 11, 2: 10, 5: 9, 10: 8, 15: 7, 20: 6, 25: 5, 30: 4, 45: 3, 50: 2, 60: 1}

    result = {} 
    for r in restaurant_locations:
      score = restaurant_scores[r['business_id']] * 3 + (weight[r['radius']] / 11) + (r['stars'] / 5)
      sentiments = r['avg_sentiments']

      # ADD SENTIMENT TO SCORE 
      score = score + sentiments['avg_pos'] - sentiments['avg_neg'] + 0.5*sentiments['avg_neu']
      result[r['business_id']] = score
    
    sorted_result = {r: result[r] for r in sorted(result, key=result.get, reverse=True)}
    return sorted_result


  def tokenize_categories(self, categories):
    """" Takes in a category string and tokenizes it into separate words. 
    Returns: a set of token words. """

    new_categories = set()
    c = set(categories.split(", "))
    new_categories = new_categories.union(c)
    
    return new_categories

  def tokenize_attributes(self, attributes):
    """" Takes in an attribute dictionary and tokenizes it into separate, key words. 
    Returns: a set of token words. """

    #CHECK - if this includes all attributes from yelp-restaurants that might be necessary

    new_attributes = set()
    attrs = eval(str(attributes))

    for key in attrs.keys():

      if isinstance(eval(attrs[key]), bool):
        new_attributes.add(key)

      else:

        if isinstance(eval(attrs[key]), dict):
          val = eval(attrs[key])
          for k in val.keys():
            if isinstance(val[k], bool):
              if val[k] == True:
                new_attributes.add(k)
            if isinstance(val[k], dict):
              val2 = val[k]
              for k2 in val2.keys():
                if isinstance(val2[k2], bool):
                  if k2 == True:
                    new_attributes.add(k2)
    
    return new_attributes

  def get_key_words(self):
    """" Used to create the key words for the restaurant data set.
    Returns: the category and attribute final list in csv called 'output'. 
    """
    
    restaurants, restaurant_dict = self.create_python_dict()[0:2]
    result_cat = set()
    result_att = set()
    for res in restaurants:
      if res['categories'] is not None:
        c = self.tokenize_categories(res['categories'])
        result_cat = result_cat.union(c)
      if res['attributes'] is not None:
        result_att = result_att.union(self.tokenize_attributes(res['attributes']))
    
    with open('output.csv', mode='w') as csv_file:
      fieldnames = ['category', 'attribute']
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      writer.writeheader()
      length = max(len(result_att), len(result_cat))
      for c in result_cat:
        writer.writerow({'category': c})
      for a in result_att:
        writer.writerow({'attribute': a})

  def get_restaurants_az(self):

    with open('yelp-restaurants.txt') as f:
      outF = open('yelp-restaurant.txt', 'w')
      for line in f.readlines():
        obj = json.loads(line)
        if obj["attributes"]:
          new_attributes = []
          attributes = eval(str(obj["attributes"]))
          for key in attributes.keys():
            if isinstance(eval(attributes[key]),bool):
              if eval(attributes[key]) == True:
                new_attributes.append(key)
            elif isinstance(eval(attributes[key]), dict):
              val = eval(attributes[key])
              for k in val.keys():
                if isinstance(val[k], bool):
                  if val[k] == True:
                    new_attributes.append(k)
                if isinstance(val[k], dict):
                  val2 = val[k]
                  for k2 in val2.keys():
                    if isinstance(val2[k2], bool):
                      if k2 == True:
                        new_attributes.append(k2)
          obj['attributes'] = new_attributes
        outF.write(json.dumps(obj))
        outF.write('\n')

    
    # with open('yelp-categories.txt', 'w') as f:
    #   for cat in categories:
    #     f.write(cat)
    #     f.write("\n")
    #   f.close()


#TESTING

# run(['romantic, hipster'],['Family', 'bagel'],'Phoenix', 'AZ', None, None)
# run(['romantic'],['Family', 'Kid'],'Phoenix', 'AZ', 'Las Vegas', 'NV')
# run(['trendy'],['Waffle'],'Scottsdale', 'AZ', 'Mesa', 'AZ')
# run([],[],'Mesa', 'AZ', 'Mesa', 'AZ')


#bugs
# run(['trendy', 'hipster'],['Kid'],'Mesa', 'AZ', 'Mesa', 'AZ')

# print("Method")
# yelp = YelpScoring()
# # print(yelp.run(['romantic, hipster'],['Family', 'bagel'], "85013", None))
# print(yelp.run(['romantic, hipster'],['Family', 'bagel'], "85013", "85003"))

# star = []
# for e in yelp.restaurant_dict:
#   star.append(yelp.restaurant_dict[e]['stars'])
# print(star)
# print(len(yelp.restaurant_dict))
# print(len(star))
# a = []
# countAtt = 0
# countNot = 0
# att = []
# for r in yelp.restaurants:
#   if r['attributes'] is not None:
#     att.append(r['attributes'])
#     countAtt += 1
#     temp = yelp.tokenize_attributes(r['attributes'])
#     for t in temp:
#       if t not in a:
#         a.append(t)
#   else:
#     countNot += 1
# with open('attribute-test.txt', 'w') as f:
#   for e in a:
#     f.write(e + "\n")
# # with open('attribute-rest.txt', 'w') as f:
# #   for at in att:
# #     f.write(json.dumps(at))
# #     f.write("\n")

# print("Method")
# yelp = YelpScoring()
# a = []
# for r in yelp.restaurants:
#   if r['categories'] is not None:
#     temp = yelp.tokenize_categories(r['categories'])
#     for t in temp:
#       if t not in a:
#         a.append(t)
# with open('cat-test.txt', 'w') as f:
#   for e in a:
#     f.write(e + "\n")
# with open('attribute-rest.txt', 'w') as f:
#   for at in att:
#     f.write(json.dumps(at))
#     f.write("\n")

# print(len(a))
# print(countAtt)
# print(countNot)


# res = create_python_dict()
# loc = restaurant_location(res, 'Phoenix', 'AZ')
# f = find_restaurants(['Family', 'Waffle'], ['romantic', 'casual'], res)
# result = combine_location(f, loc)
# k = next(iter(result))
# k2 = list(result.keys())[2]
# print(result[k])
# print(result[k2])

# run(['romantic, hipster'],['Family', 'bagel'], "85013", None)
# run(['romantic, hipster'],['Family', 'bagel'], "85013", "28277")

#print(run(['trendy'],['British', 'coffee', 'street', 'bagel'], "85013", "28277"))
