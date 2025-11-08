from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import numpy as np
# Exemple : réduire 100 vecteurs d’images à 2 dimensions 
X = np.random.rand(100, 128)  # 128 = taille du descripteur 
X_2D = TSNE(n_components=2).fit_transform(X) 
plt.scatter(X_2D[:,0], X_2D[:,1]) 
plt.title("Carte visuelle d’un corpus d’archives") 
plt.show()