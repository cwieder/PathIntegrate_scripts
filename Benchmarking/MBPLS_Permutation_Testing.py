import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mbpls.mbpls import MBPLS
import statsmodels.api as sm
import sys
import sspa
from sklearn.cross_decomposition import PLSRegression

import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
base = importr('base')
# import mixOmics
mixomics = importr('mixOmics')


def VIP_multiBlock(x_weights, x_superscores, x_loadings, y_loadings):
    # stack the weights from all blocks 
    weights = np.vstack(x_weights)
    # calculate product of sum of squares of superscores and y loadings
    sumsquares = np.sum(x_superscores**2, axis=0) * np.sum(y_loadings**2, axis=0)
    # p = number of variables - stack the loadings from all blocks
    p = np.vstack(x_loadings).shape[0]
    
    # VIP is a weighted sum of squares of PLS weights 
    vip_scores = np.sqrt(p * np.sum(sumsquares*(weights**2), axis=1) / np.sum(sumsquares))
    return vip_scores

def VIP_PLS(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips

class PermutationTest:
    def __init__(self, model:str, omics_data:list, pathways, response:list, n_comp:int, n_iter:int, vartype:str, savenull:str):
        self.model = model
        self.omics_data = omics_data.copy()
        self.pathway_df = pathways
        self.pathway_score = None
        self.response = response
        self.n_comp = n_comp
        self.n_iter = n_iter
        self.null_dist = []
        self.p_vals = None
        self.vartype = vartype
        self.savenull = savenull
        self.index = None

        if model == 'PLS':
            self.simulate_null_PLS()
        if model == 'MBPLS':
            self.simulate_null_MBPLS()
        if model == 'DIABLO sparse':
            self.simulate_null_DIABLO()
        if model == 'DIABLO_loading':
            self.simulate_null_DIABLO_loading()

    def simulate_null_MBPLS(self):
        rng = np.random.default_rng()
        response_perm = rng.permutation(self.response)
        self.pathway_score = [sspa.sspa_kpca(i, self.pathway_df) for i in self.omics_data]
        # self.index = sum([i.columns.tolist() for i in self.pathway_score], [])

        m_null = MBPLS(n_components=self.n_comp)
        m_null.fit(self.pathway_score, response_perm)
        if self.vartype == 'beta':
            null = m_null.beta_.flatten()
            self.null_dist = [np.round(i, 5) for i in null]
        else:
            m_null_VIP = VIP_multiBlock(m_null.W_, m_null.Ts_, m_null.P_, m_null.V_)
            null = m_null_VIP
            self.null_dist = [np.round(i, 5) for i in null]

        if self.savenull:
            # null_df = pd.DataFrame(self.null_dist)
            # null_df.to_csv(self.savenull+str(self.vartype)+"_null_"+str(self.n_iter)+".txt", sep='\t', index=False)
            # with open("Met_Prot_KPCA_null/COPDgene_M_P_kPCA_"+str(self.vartype)+"_null_"+str(self.n_iter)+".txt", "w") as myfile:
            with open(self.savenull+str(self.vartype)+"_null_"+str(self.n_iter)+".txt", "w") as myfile:
                for i in self.null_dist:
                    myfile.write(f"{i}\n")

        return None
    
    def simulate_null_PLS(self):
        rng = np.random.default_rng()
        response_perm = rng.permutation(self.response)
        # concat omics
        concat_data = self.omics_data[0].merge(self.omics_data[1], how='inner', right_index=True, left_index=True)

        self.pathway_score = sspa.sspa_kpca(concat_data, self.pathway_df)

        m_null = PLSRegression(n_components=self.n_comp)
        m_null.fit(self.pathway_score, response_perm)
        if self.vartype == 'beta':
            null = m_null.coef_.flatten()
            self.null_dist = [np.round(i, 5) for i in null]
        else:
            m_null_VIP = VIP_PLS(self.pathway_score, response_perm, m_null)
            null = m_null_VIP
            self.null_dist = [np.round(i, 5) for i in null]

        if self.savenull:
            # null_df = pd.DataFrame(self.null_dist)
            # null_df.to_csv(self.savenull+str(self.vartype)+"_null_"+str(self.n_iter)+".txt", sep='\t', index=False)
            # with open("Met_Prot_KPCA_null/COPDgene_M_P_kPCA_"+str(self.vartype)+"_null_"+str(self.n_iter)+".txt", "w") as myfile:
            with open(self.savenull+str(self.vartype)+"_null_"+str(self.n_iter)+".txt", "w") as myfile:
                for i in self.null_dist:
                    myfile.write(f"{i}\n")

        return None
    
    def simulate_null_DIABLO(self):
        n_boot=25
        rng = np.random.default_rng()
        response_perm = rng.permutation(self.response)
        self.pathway_score = [sspa.sspa_kpca(i, self.pathway_df) for i in self.omics_data]
        self.pathway_score[0].index = self.pathway_score[1].index

        design_mat = pd.DataFrame(np.full((2, 2), 0), index=['Metabolomics_kpca', 'Proteomics_kpca'], columns=['Metabolomics_kpca', 'Proteomics_kpca'])
        np.fill_diagonal(design_mat.values, 0)

        with (ro.default_converter + pandas2ri.converter).context():
            r_metab_kpca = ro.conversion.get_conversion().py2rpy(self.pathway_score[0])
            r_prot_kpca = ro.conversion.get_conversion().py2rpy(self.pathway_score[1])
            r_design_mat = ro.conversion.get_conversion().py2rpy(design_mat)
            r_labels_kpca = ro.IntVector(response_perm)

        X_matrices = ro.ListVector([('Metabolomics_kpca', r_metab_kpca), ('Proteomics_kpca', r_prot_kpca)])
        test_keepX = ro.ListVector([('Metabolomics_kpca', ro.IntVector([100, 50, 25, 5]) ), ('Proteomics_kpca', ro.IntVector([100, 50, 25, 5]))])
        tune_features = mixomics.tune_block_splsda(X_matrices, r_labels_kpca, ncomp = 1, 
                                test_keepX = test_keepX, design = r_design_mat,
                                validation = 'Mfold', folds = 5, nrepeat = 1)
        n_features_met = tune_features.rx2('choice.keepX')[0][0]
        n_features_prot = tune_features.rx2('choice.keepX')[1][0]
        
        loadings_full = []
        for i in range(0, n_boot):
            # bootstrap 100 times 50% of the sample population
            rng = np.random.default_rng()
            response_perm = rng.permutation(self.response)[0:60]
            met_sample = self.omics_data[0].sample(n=60, replace=True)
            prot_sample = self.omics_data[1].loc[met_sample.index, :]
            
            # rename samples to avoid duplicated indexes
            met_sample = met_sample.reset_index(drop=True)
            prot_sample = prot_sample.reset_index(drop=True)
        
            kpca_scores_met = sspa.sspa_kpca(met_sample, self.pathway_df)
            kpca_scores_prot = sspa.sspa_kpca(prot_sample, self.pathway_df)
            kpca_scores_met['Group'] = response_perm
            kpca_scores_prot['Group'] = response_perm
            pathcols =  kpca_scores_met.iloc[:, :-1].columns.tolist() + kpca_scores_prot.iloc[:, :-1].columns.tolist()

            #create design matrix NULL
            design_mat = pd.DataFrame(np.full((2, 2), 0), index=['Metabolomics_kpca', 'Proteomics_kpca'], columns=['Metabolomics_kpca', 'Proteomics_kpca'])
            np.fill_diagonal(design_mat.values, 0)
        
            with (ro.default_converter + pandas2ri.converter).context():
                r_metab_kpca = ro.conversion.get_conversion().py2rpy(kpca_scores_met.iloc[:, :-1])
                r_prot_kpca = ro.conversion.get_conversion().py2rpy(kpca_scores_prot.iloc[:, :-1])
                r_design_mat = ro.conversion.get_conversion().py2rpy(design_mat)
                r_labels_kpca = ro.IntVector(kpca_scores_met['Group'].tolist())
            X_matrices = ro.ListVector([('Metabolomics_kpca', r_metab_kpca), ('Proteomics_kpca', r_prot_kpca)])
            keep_features = ro.ListVector([('Metabolomics_kpca', ro.IntVector([n_features_met]) ), ('Proteomics_kpca', ro.IntVector([n_features_prot]))])
            diablo_res_sparse = mixomics.block_splsda(X=X_matrices, Y=r_labels_kpca, ncomp=1, design=r_design_mat, keepX = keep_features)

            # get loadings 
            met_loadings_sparse = pd.DataFrame(diablo_res_sparse.rx2("loadings").rx2('Metabolomics_kpca'), index=kpca_scores_met.iloc[:, :-1].columns)
            prot_loadings_sparse = pd.DataFrame(diablo_res_sparse.rx2("loadings").rx2('Proteomics_kpca'), index=kpca_scores_prot.iloc[:, :-1].columns)

            # stack loadings
            loadings_all = pd.concat([met_loadings_sparse, prot_loadings_sparse], axis=0)
            loadings_full.append(loadings_all[0].to_numpy())
        
        # count how many non zero coefficients per column (across bootstrap)
        loadings_full_arr = np.array(loadings_full)
        non_zero_counts = np.count_nonzero(loadings_full_arr, axis=0)
        
        # divide to get propotion of times each var was significant 
        non_zero_counts_proportion = non_zero_counts / n_boot
        self.null_dist = non_zero_counts_proportion
        
        if self.savenull:
            with open(self.savenull+"DIABLO_null_"+str(self.n_iter)+".txt", "w") as myfile:
                for i in self.null_dist:
                    myfile.write(f"{i}\n")

        return None
    
    def simulate_null_DIABLO_loading(self):
        print('hoohah')
        rng = np.random.default_rng()
        response_perm = rng.permutation(self.response)
        self.pathway_score = [sspa.sspa_kpca(i, self.pathway_df) for i in self.omics_data]

        self.pathway_score[0].index = self.pathway_score[1].index

        #create design matrix NULL
        design_mat = pd.DataFrame(np.full((2, 2), 0), index=['Metabolomics_kpca', 'Proteomics_kpca'], columns=['Metabolomics_kpca', 'Proteomics_kpca'])
        np.fill_diagonal(design_mat.values, 0)
    
        with (ro.default_converter + pandas2ri.converter).context():
            r_metab_kpca = ro.conversion.get_conversion().py2rpy(self.pathway_score[0])
            r_prot_kpca = ro.conversion.get_conversion().py2rpy(self.pathway_score[1])
            r_design_mat = ro.conversion.get_conversion().py2rpy(design_mat)
            r_labels_kpca = ro.IntVector(response_perm)
        # print(self.pathway_score[0].index == self.pathway_score[1].index)

        X_matrices = ro.ListVector([('Metabolomics_kpca', r_metab_kpca), ('Proteomics_kpca', r_prot_kpca)])
        diablo_res = mixomics.block_plsda(X=X_matrices, Y=r_labels_kpca, ncomp=1, design=r_design_mat)

        # get loadings 
        met_loadings = pd.DataFrame(diablo_res.rx2("loadings").rx2('Metabolomics_kpca'), index=self.pathway_score[0].columns)
        prot_loadings = pd.DataFrame(diablo_res.rx2("loadings").rx2('Proteomics_kpca'), index=self.pathway_score[1].columns)
        # stack loadings
        loadings_all = pd.concat([met_loadings, prot_loadings], axis=0)
        self.null_dist = [np.round(i, 5) for i in loadings_all[0].to_numpy()]

        if self.savenull:
            with open(self.savenull+"DIABLO_null_"+str(self.n_iter)+".txt", "w") as myfile:
                for i in self.null_dist:
                    myfile.write(f"{i}\n")

        return None

    # def calculate_pval(self):
    #     # two sided t-test - use absolute values
    #     null_dist = [abs(i) for i in self.null_dist]
    #     # calculate sum of null means greater/less than observed betas
    #     self.p_val = np.sum(null_dist >= abs(self.observed_var))/self.n_iter

    #     return self.p_val


# kpca_scores_met = pd.read_csv('COPDgene/KPCA_Met_Scores.csv', index_col=0)
# kpca_scores_prot = pd.read_csv('COPDgene/KPCA_Prot_Scores.csv', index_col=0)

# m1 = MBPLS(n_components=4).fit([kpca_scores_met.iloc[:, :-1], kpca_scores_prot.iloc[:, :-1]], kpca_scores_met['Group'].values)
# m1_VIP = VIP_multiBlock(m1.W_, m1.Ts_, m1.P_, m1.V_)
# nullm = PermutationTestMBPLS(
#     omics_data=[kpca_scores_met.iloc[:, :-1], kpca_scores_prot.iloc[:, :-1]], 
#     response=kpca_scores_met['Group'], 
#     n_comp=3, n_iter=200, var=m1_VIP[1597], var_idx=1597, vartype='vip', plot=True)

# nullm.calculate_pval()
# print(nullm.null_dist)