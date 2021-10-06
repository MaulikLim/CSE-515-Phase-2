from imageLoader import load_images_from_folder
import numpy as np

class SVD:

    def __init__(self, model, k):
        self.name = "svd"
        self.model = model
        self.k = k

    def compute_semantics_type(self, images, labels):
        data = self.model.compute_features_for_images(images)
        data_t = data.transpose()
        lsf = np.matmul(data,data_t)
        rsf = np.matmul(data_t,data)
        lsf_eig = np.linalg.eig(lsf)
        rsf_eig = np.linalg.eig(rsf)
        lsf_eig_values = np.flip(np.sort(np.around(lsf_eig[0],3)))
        rsf_eig_values = np.flip(np.sort(np.around(rsf_eig[0],3)))
        eig_mat = []
        for i in range(min(len(lsf_eig_values),len(lsf_eig_values))):
            if(lsf_eig_values[i]!=0 and lsf_eig_values[i] == rsf_eig_values[i]):
                eig_mat.append(lsf_eig_values[i])
        self.k = min(self.k,len(eig_mat))
        lsf_eig_vectors = []
        rsf_eig_vectors = []
        for val in eig_mat:
            for i in range(len(lsf_eig[0])):
                if(val==lsf_eig[0][i]):
                    lsf_eig_vectors.append(lsf_eig[1][:,i])
                    break
            for i in range(len(rsf_eig[0])):
                if(val==rsf_eig[0][i]):
                    rsf_eig_vectors.append(rsf_eig[1][:,i])
                    break
            if(len(lsf_eig_vectors)==self.k):
                break
        for i in range(self.k,len(eig_mat)):
            eig_mat[i]=0
        eig_mat = np.sqrt(np.diag(np.array(eig_mat)))
        return [np.array(lsf_eig_vectors),eig_mat,np.array(rsf_eig_vectors)]

