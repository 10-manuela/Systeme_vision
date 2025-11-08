from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np 
# Deux vecteurs de caractéristiques fictifs 
v1 = np.array([0.3, 0.2, 0.9]) 
v2 = np.array([0.4, 0.25, 0.85]) 
similarity = cosine_similarity([v1], [v2])
print ("Similiralité :", similarity[0] [0])