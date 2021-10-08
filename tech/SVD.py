from imageLoader import load_images_from_folder
import numpy as np

class SVD:

    def __init__(self, k):
        self.name = "svd"
        self.k = k

    def compute_semantics(self, data):
        data_t = data.transpose()
        l = np.matmul(data,data_t)
        r = np.matmul(data_t,data)
        l_eig = np.linalg.eig(l)
        r_eig = np.linalg.eig(r)
        l_eig_values = np.around(l_eig[0].real,2)
        r_eig_values = np.around(r_eig[0].real,2)
        sorted_l_indices = np.flip(np.argsort(l_eig_values))
        sorted_r_indices = np.flip(np.argsort(r_eig_values))
        # print(l_eig_values)
        # # print(r_eig[0])
        # print(r_eig_values)
        # l_eig_values = np.flip(np.sort(np.around(l_eig[0],3)))
        # r_eig_values = np.flip(np.sort(np.around(r_eig[0],3)))
        eig_mat = []
        l_eig_vectors = []
        r_eig_vectors = []
        for i in range(min(len(sorted_l_indices),len(sorted_r_indices))):
            if(l_eig_values[sorted_l_indices[i]] == r_eig_values[sorted_r_indices[i]]):
                eig_mat.append(l_eig_values[sorted_l_indices[i]])
                l_eig_vectors.append(l_eig[1][:,sorted_l_indices[i]].real)
                r_eig_vectors.append(r_eig[1][:,sorted_r_indices[i]].real.tolist())
                if(len(eig_mat)==self.k):
                    break
            else:
                break
        eig_mat = np.sqrt(np.diag(np.array(eig_mat)))
        print(np.array(l_eig_vectors).shape)
        # print(sorted_r_indices)
        print(np.array(r_eig[1]).shape)
        return [np.array(l_eig_vectors).transpose().tolist(),eig_mat.tolist(),r_eig_vectors]

