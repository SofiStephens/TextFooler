import numpy as np
import sys

'''This script reads word embeddings from a file, normalizes them, computes the cosine similarity matrix, 
and saves the result to a NumPy binary file.'''

# The path to the word embeddings file, provided as a command-line argument
embedding_path = sys.argv[1] # '/data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt'

# Initialize an empty list to store the word embeddings
embeddings = []
# Open the specified file containing word embeddings
with open(embedding_path, 'r') as ifile:
    # Iterate over each line in the file
    for line in ifile:
        # Parse the line, converting the space-separated values to floats and skipping the first value
        embedding = [float(num) for num in line.strip().split()[1:]]
        # Append the embedding to the list
        embeddings.append(embedding)
# Convert the list of embeddings to a NumPy array
embeddings = np.array(embeddings)
# Transpose the embeddings array
print(embeddings.T.shape)
# Compute the L2 normalization of the embeddings along axis 1
norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
# Normalize the embeddings by dividing them by their L2 norm
embeddings = np.asarray(embeddings / norm, "float32")
# Compute the cosine similarity matrix by taking the dot product of the normalized embeddings
product = np.dot(embeddings, embeddings.T)
# Save the cosine similarity matrix to a NumPy binary file
np.save(('cos_sim_counter_fitting.npy'), product)
