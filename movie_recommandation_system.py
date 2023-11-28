#!/usr/bin/env python
# coding: utf-8

# # *Importing the Required Libraries*

# In[23]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


# # *Reading the Data from CSV file*

# In[24]:


movies_data = pd.read_csv('movies_data.csv')


# In[25]:


movies_data.head()


# In[26]:


movies_data.tail()


# In[27]:


movies_data.shape


# In[28]:


movies_data.columns


# # *Selecting the desired features for the movie recommandation system*

# In[29]:


selected_features = ['genres','keywords','original_title','overview','tagline','cast','director']


# # *Filling all the missing values with the null string*

# In[30]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# # *Combining all the features*

# In[31]:


combined_feature = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['original_title']+' '+movies_data['overview']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[32]:


combined_feature


# # *Converting the text data into feature vectors*

# In[33]:


vectorizer = TfidfVectorizer()


# In[34]:


feature_vector = vectorizer.fit_transform(combined_feature)


# In[35]:


print(feature_vector)


# # *Using Cosine Similarity*

# In[36]:


# finding the cosine similarity

similarity = cosine_similarity(feature_vector)


# In[37]:


similarity


# In[38]:


similarity.shape


# In[39]:


similarity[0]


# # *Taking name of the movie from the user*

# In[40]:


# movie_name_user = input("Enter the name of your favourite movie : ")


# # *Finding the clossest match of the user's entered movie*

# In[41]:


movie_list = movies_data['original_title'].tolist()


# In[42]:


movie_list


# # *Finding the close match*

# In[43]:


# find_close_match = difflib.get_close_matches(movie_name_user,movie_list)


# In[44]:


# find_close_match


# In[45]:


# if len(find_close_match) == 0:
#     print("Sorry not possible to recommand any of the movie\n")
#     print("You can try again with another movie\n")


# In[46]:


# close_match = find_close_match[0]


# In[47]:


# close_match


# # *Getting the index value of the closed match movie*

# In[48]:


# index_of_movie = movies_data[movies_data['original_title']==close_match]['index'].values[0]


# In[49]:


# index_of_movie


# # *Comparing the similarity score of all the movies of the dataset with the user entered movie*

# In[50]:


# similarity_score = list(enumerate(similarity[index_of_movie]))


# In[51]:


# similarity_score


# In[52]:


# len(similarity_score)


# In[53]:


# sorted_similar_movies = sorted(similarity_score,reverse=True,key=lambda x: x[1])


# In[54]:


# sorted_similar_movies


# # *Suggesting the movies for the user*

# In[55]:


# print("The suggested movie for you are listed below \n")

# i = 1
# for movie in sorted_similar_movies:
#   index = movie[0]
#   movie_title = movies_data[movies_data.index==index]['original_title'].values[0]
#   if i<=30:
#     print(f"{i} -- {movie_title}")
#     i+=1


# # **Movie Recommandation System**

# In[56]:


def recommand_movie(movies_data,movie_list,similarity,movie_name_user):
  find_close_match = difflib.get_close_matches(movie_name_user,movie_list)
  if len(find_close_match)>0:
    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data['original_title']==close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_score,reverse=True,key=lambda x: x[1])
    print("The suggested movie for you are listed below \n")
    i = 1
    for movie in sorted_similar_movies:
      index = movie[0]
      movie_title = movies_data[movies_data.index==index]['original_title'].values[0]
      if i<=30:
        print(f"{i} -- {movie_title}")
        i+=1
  else:
    print("Sorry not possible to recommand any of the movie\n")
    print("You can try again with another movie\n")


# In[57]:


movie_name_user = input("Enter the name of your favourite movie : ")
recommand_movie(movies_data,movie_list,similarity,movie_name_user)

