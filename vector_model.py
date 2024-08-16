import numpy as np
import math
import re
from collections import Counter
import deep_translator
from translate import Translator



# Assuming the TextProcessor class is defined as follows
def compute_entropy(text):
    """Compute the Shannon entropy of a text."""
    # Calculate frequency of each character
    freq = Counter(text)
    total_chars = len(text)

    # Calculate entropy
    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in freq.values())
    return entropy

def compute_total_bits(A):
    total_bits = 0
    for row in A:
        text_content = row[0]
        if text_content:
            entropy = compute_entropy(text_content)
            total_bits += entropy * len(text_content)  # Total bits = entropy * length of text
    return total_bits

class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.translator = Translator(to_lang='en')

    def word_tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def normalize_text(self, text):
        tokens = self.word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        return stemmed_tokens

    def clean_content(self, content):
        content = str(content)

        char_limit = 300
        translated_chunks = []
        for i in range(0, len(content), char_limit):
            chunk = content[i:i + char_limit]
            try:
                translated_chunk = self.translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            except Exception as e:
                print(f"An error occurred during translation: {e}")
                translated_chunks.append(chunk)
        content = ''.join(translated_chunks)

        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'http[s]?://\S+', '', content)
        content = re.sub(r'(?<=\S)\s{2,}(?=\S)', ' ', content)
        content = re.sub(r'(\s*\n\s*)+', '\n\n', content)
        content = re.sub(r'\s{2,}', '  ', content)
        content = content.strip()

        return content

class PorterStemmer:
    def __init__(self):
        self.suffixes = ['ing', 'ly', 'ed', 'es', 's', 'er', 'ment', 'ness']
        self.rules = [
            ('ies', 'y'),
            ('s', ''),
            ('ed', ''),
            ('ing', ''),
        ]

    def stem(self, word):
        for suffix in self.suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        for suffix, replacement in self.rules:
            if word.endswith(suffix):
                return word[:-len(suffix)] + replacement
        return word

ENGLISH_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

class Vectorizer:
    def __init__(self, vector_dim=50, vocab_size=1000):
        self.vector_dim = vector_dim
        self.word2vec = Word2VecSkipGram(vocab_size=vocab_size, embedding_dim=vector_dim)
        self.text_processor = TextProcessor()

    def preprocess_content(self, content):
        tokens = self.text_processor.normalize_text(content)
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
        return tokens

    def generate_vector(self, content):
        tokens = self.preprocess_content(content)
        word_vectors = []

        for token in tokens:
            # Convert token to index
            word_index = hash(token) % self.word2vec.vocab_size
            word_vectors.append(self.word2vec.word2vec(word_index))

        if not word_vectors:
            return np.zeros(self.vector_dim)

        word_vectors = np.array(word_vectors)

        if word_vectors.shape[0] < self.vector_dim:
            padded_vectors = np.zeros((self.vector_dim, word_vectors.shape[1]))
            padded_vectors[:word_vectors.shape[0], :] = word_vectors
            return np.mean(padded_vectors, axis=0)

        elif word_vectors.shape[0] > self.vector_dim:
            pca = PCA(n_components=self.vector_dim)
            reduced_vectors = pca.fit_transform(word_vectors)
            return np.mean(reduced_vectors, axis=0)

        else:
            return np.mean(word_vectors, axis=0)


class VectorOperations:
    def magnitude(self, vector):
        return np.linalg.norm(vector)

    def cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        return dot_product / (magnitude_v1 * magnitude_v2) if magnitude_v1 * magnitude_v2 != 0 else 0

class Scaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class PCA(Scaler):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def explained_variance_ratio(self, X):
        if self.components is None:
            raise RuntimeError("PCA not yet fitted.")
        covariance_matrix = np.cov(X - self.mean, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        total_variance = np.sum(eigenvalues)
        variance_ratios = eigenvalues[:self.n_components] / total_variance
        return variance_ratios

# Specific Implementations

class VectorCompare(VectorOperations, TextProcessor):
    def concordance(self, document):
        if not isinstance(document, str):
            raise ValueError('Supplied Argument should be of type string')
        words = self.normalize_text(document)
        return Counter(words)

    def vector_magnitude(self, v1, v2):
        if len(v1) != len(v2):
            raise ValueError('Vectors must have the same length')
        return self.magnitude(np.array(v1) - np.array(v2))

    def vector_normalize(self, v1):
        return v1 / np.linalg.norm(v1)

    def relation(self, concordance1, concordance2):
        if not isinstance(concordance1, dict) or not isinstance(concordance2, dict):
            raise ValueError('Supplied Arguments should be of type dict')
        topvalue = sum(concordance1.get(word, 0) * concordance2.get(word, 0) for word in concordance1)
        magnitude_product = self.magnitude(concordance1) * self.magnitude(concordance2)
        return topvalue / magnitude_product if magnitude_product != 0 else 0

    def rank_with_pagerank(self, documents, pagerank_scores):
        return sorted(documents, key=lambda doc: pagerank_scores.get(doc, 0), reverse=True)

class InvertedIndex(VectorCompare):
    def __init__(self):
        super().__init__()
        self.index = {}

    def add_document(self, doc_id, text):
        terms = self.normalize_text(text)
        for term in terms:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(doc_id)

    def search(self, query):
        terms = self.normalize_text(query)
        doc_ids = [self.index.get(term, set()) for term in terms]
        if not doc_ids:
            return set()
        return set.intersection(*doc_ids)

    def search_with_pagerank(self, query, pagerank_scores):
        doc_ids = self.search(query)
        return sorted(doc_ids, key=lambda doc: pagerank_scores.get(doc, 0), reverse=True)

def parallel_search(queries, index):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda query: index.search(query), queries)
    return list(results)

class PageRank:
    def __init__(self, graph):
        self.graph = graph

    def calculate_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        pagerank = {node: 1.0 / len(self.graph.nodes) for node in self.graph.nodes}
        for _ in range(max_iter):
            new_pagerank = {}
            for node in self.graph.nodes:
                rank_sum = sum(
                    pagerank[neighbor] / len(self.graph.neighbors(neighbor))
                    for neighbor in self.graph.predecessors(node)
                )
                new_pagerank[node] = (1 - alpha) / len(self.graph.nodes) + alpha * rank_sum
            diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in pagerank)
            if diff < tol:
                break
            pagerank = new_pagerank
        return pagerank

class KMeans:
    def __init__(self, k=5, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.wcss = None

    def initialize_centroids(self, X):
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]

    def assign_clusters(self, X):
        clusters = []
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
            cluster = np.argmin(distances)
            clusters.append(cluster)
        return np.array(clusters)

    def recalculate_centroids(self, X, clusters):
        new_centroids = []
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                new_centroid = X[np.random.choice(len(X))]
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def calculate_wcss(self, X, clusters):
        wcss = 0
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - self.centroids[i], axis=1)
                wcss += np.sum(distances**2)
        return wcss

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        clusters = None
        for i in range(self.max_iterations):
            clusters = self.assign_clusters(X)
            new_centroids = self.recalculate_centroids(X, clusters)
            diff = np.linalg.norm(self.centroids - new_centroids)

            if diff < self.tolerance:
                print(f"Converged after {i+1} iterations.")
                break

            self.centroids = new_centroids

        self.wcss = self.calculate_wcss(X, clusters)
        return clusters

    def predict(self, X):
        return self.assign_clusters(X)

class Word2VecSkipGram:
    def __init__(self, vocab_size=10000, embedding_dim=25, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        self.W1, self.W2 = self.initialize_weights(vocab_size, embedding_dim)

    def initialize_weights(self, vocab_size, embedding_dim):
        limit = np.sqrt(6 / (vocab_size + embedding_dim))
        W1 = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        W2 = np.random.uniform(-limit, limit, (embedding_dim, vocab_size))
        return W1, W2

    def softmax(self, x):
        max_x = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max_x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X):
        self.h = np.dot(X, self.W1)
        self.u = np.dot(self.h, self.W2)
        self.y_pred = self.softmax(self.u)
        return self.y_pred

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def backward(self, X, y_true, y_pred):
        self.dL_du = y_pred - y_true
        self.dL_dW2 = np.dot(self.h.T, self.dL_du)
        self.dL_dh = np.dot(self.dL_du, self.W2.T)
        self.dL_dW1 = np.dot(X.T, self.dL_dh)

        self.W1 -= self.learning_rate * self.dL_dW1
        self.W2 -= self.learning_rate * self.dL_dW2

    def train(self, X, y_true, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_true, y_pred)
            self.backward(X, y_true, y_pred)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    def word2vec(self, word_index):
        if word_index < self.vocab_size:
            return self.W1[word_index]
        else:
            raise ValueError("Word index out of range.")
