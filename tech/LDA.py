from sklearn.decomposition import LatentDirichletAllocation
import pickle


class LDA:
    def __init__(self, k=None, max_iterations=100, file_name=None):
        self.name = 'LDA'
        if file_name is not None:
            self.model = pickle.load(open(file_name + '.pk', 'rb'))
        else:
            self.k = k
            self.model = LatentDirichletAllocation(n_components=k, max_iter=max_iterations)

    def compute_semantics(self, data):
        return self.model.fit(data)

    def transform_data(self, data):
        return self.model.transform(data)

    def save_model(self, file_name):
        pickle.dump(self.model, open(file_name + '.pk', 'wb'))
