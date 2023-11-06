import pandas as pd
import numpy as np
import sspa
from simulate_pathway_signals import SimulateData
from mbpls.mbpls import MBPLS
import statsmodels.api as sm
import sys
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt
import seaborn as sns

base = importr('base')
# import mixOmics
mixomics = importr('mixOmics')

def DIABLO_fit(metab, prot, y, design):
    # convert labes to list vector
    r_labels = ro.IntVector(y)
    design_mat = pd.DataFrame(np.full((2, 2), design), index=['Metabolomics', 'Proteomics'], columns=['Metabolomics', 'Proteomics'])
    np.fill_diagonal(design_mat.values, 0)
    # convert dataframes to r dataframes
    with (ro.default_converter + pandas2ri.converter).context():
        r_metab = ro.conversion.get_conversion().py2rpy(metab)
        r_prot = ro.conversion.get_conversion().py2rpy(prot)
        r_design_mat = ro.conversion.get_conversion().py2rpy(design_mat)

    X_matrices = ro.ListVector([('Metabolomics', r_metab), ('Proteomics', r_prot)])
    diablo_res = mixomics.block_plsda(X=X_matrices, Y=r_labels, ncomp=1, design=r_design_mat)
    return diablo_res

def DIABLO_predict(diablo_res, metab, prot):

    # convert dataframes to r dataframes
    with (ro.default_converter + pandas2ri.converter).context():
        r_metab = ro.conversion.get_conversion().py2rpy(metab)
        r_prot = ro.conversion.get_conversion().py2rpy(prot)
    # convert labes to list vector
    X_matrices = ro.ListVector([('Metabolomics', r_metab), ('Proteomics', r_prot)])
    diablo_pred = mixomics.predict_block_pls(diablo_res, newdata = X_matrices)
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(diablo_pred)
    y_pred = list(pd_from_r_df['WeightedVote']['centroids.dist'])
    y_pred = np.array([np.nan if int(i) not in [1, 0] else int(i) for i in y_pred])
    return y_pred


def sspa_kpca(train, test, pathways, min_entity=2):

    """
    Kernel PCA method for single sample pathway analysis

    Args:
        mat (pd.DataFrame): pandas DataFrame omics data matrix consisting of m rows (samples) and n columns (entities).
        Do not include metadata columns
        pathways (pd.DataFrame): Dictionary of pathway identifiers (keys) and corresponding list of pathway entities (values).
        Entity identifiers must match those in the matrix columns
        min_entity (int): minimum number of metabolites mapping to pathways for ssPA to be performed

    Returns:
        pandas DataFrame of pathway scores derived using the kPCA method. Columns represent pathways and rows represent samples.
    """

    pathway_matrices_train = []
    pathway_matrices_test = []
    pathway_ids = []
    for pathway, compounds in pathways.items():
        single_pathway_matrix = train.drop(train.columns.difference(compounds), axis=1)
        if single_pathway_matrix.shape[1] >= min_entity:
            pathway_matrices_train.append(single_pathway_matrix.values)
            pathway_matrices_test.append(test.drop(test.columns.difference(compounds), axis=1).values)
            pathway_ids.append(pathway)

    scores_train = []
    scores_test = []
    for n, m in enumerate(pathway_matrices_train):
        kpca = KernelPCA(n_components=2, kernel="rbf")
        kpca.fit(m)
        scores_train.append(kpca.transform(m)[:, 0])
        scores_test.append(kpca.transform(pathway_matrices_test[n])[:, 0])

        # scatter plot of first two components of kPCA
        # train and test set
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # sns.scatterplot(x=kpca.transform(pathway_matrices_test[n])[:, 0], y=kpca.transform(pathway_matrices_test[n])[:, 1], ax=ax1, hue=group[test.index])
        # sns.scatterplot(x=kpca.transform(m)[:, 0], y=kpca.transform(m)[:, 1], ax=ax2, hue=group[train.index])
        # plt.show()
    scores_df_train = pd.DataFrame(scores_train, columns=train.index, index=pathway_ids).T
    scores_df_test = pd.DataFrame(scores_test, columns=test.index, index=pathway_ids).T
    return scores_df_train, scores_df_test

prot = pd.read_csv('../COVID/prot_uniprot.csv', index_col=0)
metab = pd.read_csv('../COVID/metab_chebi.csv', index_col=0)
# impute missing values with median
prot_imp = prot.iloc[:, :-1].fillna(prot.iloc[:, :-1].median())
metab_imp = metab.iloc[:, :-1].fillna(metab.iloc[:, :-1].median())

prot_imp['Group'] = [1]*63 + [0]*64
metab_imp['Group'] = [1]*63 + [0]*64

# load multi-omics pathways
mo_paths = pd.read_csv("../Pathway_databases/Reactome_multi_omics_ChEBI_Uniprot.csv", index_col=0, dtype=object)
def filt_pathways(path_df):
    pathdict = sspa.utils.pathwaydf_to_dict(path_df)
    compounds_present = metab.columns.tolist() + prot.columns.tolist()
    pathways_present = {k: v for k, v in pathdict.items() if len([i for i in compounds_present if i in v]) > 2}
    path_coverage = {k: [i for i in v if i in compounds_present] for k, v in pathways_present.items()}
    filt_paths = path_df[path_df.index.isin(pathways_present)]
    return filt_paths

mo_paths = filt_pathways(mo_paths)
mo_paths_dict = sspa.utils.pathwaydf_to_dict(mo_paths)

def get_auc(dsets, effect, enrich_paths, paths_dict):
    # simulate data
    simulated_dset = SimulateData(
    input_data=[dsets[0].iloc[:, :-1], dsets[1].iloc[:, :-1]],
    metadata=[dsets[0]['Group'], dsets[1]['Group']],
    pathways=paths_dict,
    enriched_paths=enrich_paths
    ).enrich_paths_base(effect=effect, effect_type='constant', input_type='log')

    metabsim= simulated_dset[0]
    protsim = simulated_dset[1]
    labels = metabsim['Group'].values

    res_all = []

    # generate stratified kfold splits
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for i, (train_index, test_index) in enumerate(skf.split(metabsim.iloc[:, :-1], metabsim['Group'])):

        # generate test train split
        X_train, X_test = metabsim.iloc[train_index, :-1], metabsim.iloc[test_index, :-1]
        X_train2, X_test2 = protsim.iloc[train_index, :-1], protsim.iloc[test_index, :-1]

        # scale test and train sets using pipeline
        pipe = Pipeline([
        ('scale', StandardScaler())])
        X_train = pd.DataFrame(pipe.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(pipe.transform(X_test), index=X_test.index, columns=X_test.columns)
        X_train2 = pd.DataFrame(pipe.fit_transform(X_train2), index=X_train2.index, columns=X_train2.columns)
        X_test2 = pd.DataFrame(pipe.transform(X_test2), index=X_test2.index, columns=X_test2.columns)

        y = labels
        y_train, y_test = y[train_index], y[test_index]

        # generate sspa scores for DIABLO and MBPLS
        sspa_metab_train, sspa_metab_test = sspa_kpca(X_train, X_test, mo_paths_dict)
        sspa_prot_train, sspa_prot_test = sspa_kpca(X_train2, X_test2, mo_paths_dict)

        # generate sspa scores for SingleView
        concat_omics_train = pd.concat([X_train, X_train2], axis=1)
        concat_omics_test = pd.concat([X_test, X_test2], axis=1)
        sspa_concat_train, sspa_concat_test = sspa_kpca(concat_omics_train, concat_omics_test, mo_paths_dict)

        #fit MBPLS model
        mbpls = MBPLS(n_components=1)
        mbpls.fit([sspa_metab_train.to_numpy(), sspa_prot_train.to_numpy()], y_train)
        y_pred_mbpls = mbpls.predict([sspa_metab_test.to_numpy(), sspa_prot_test.to_numpy()])
        auc_mbpls = roc_auc_score(y_test, y_pred_mbpls)

        # mbpls_molec = MBPLS(n_components=1)
        # mbpls_molec.fit([X_train.to_numpy(), X_train2.to_numpy()], y_train)
        # y_pred_mbpls_molec = mbpls_molec.predict([X_test.to_numpy(), X_test2.to_numpy()])
        # auc_mbpls_molec = roc_auc_score(y_test, y_pred_mbpls_molec)

        # fit single view model (PLS)
        sv = PLSRegression(n_components=1)
        sv.fit(sspa_concat_train, y_train)
        y_pred_sv = sv.predict(sspa_concat_test)
        auc_sv = roc_auc_score(y_test, y_pred_sv)

        #fit DIABLO model
        diablo_res = DIABLO_fit(sspa_metab_train, sspa_prot_train, y_train, 0)
        y_pred_diablo = DIABLO_predict(diablo_res, sspa_metab_test, sspa_prot_test)
        auc_diablo = roc_auc_score(y_test, y_pred_diablo)

        # fit DIABLO full model
        diablo_res_full = DIABLO_fit(sspa_metab_train, sspa_prot_train, y_train, 1)
        y_pred_diablo_full = DIABLO_predict(diablo_res_full, sspa_metab_test, sspa_prot_test)
        auc_diablo_full = roc_auc_score(y_test, y_pred_diablo_full)

        # fit DIABLO null molecule model
        diablo_res_mol = DIABLO_fit(X_train, X_train2, y_train, 0)
        y_pred_diablo_mol = DIABLO_predict(diablo_res_mol, X_test, X_test2)
        auc_diablo_mol = roc_auc_score(y_test, y_pred_diablo_mol)

        res = [auc_mbpls, auc_sv, auc_diablo, auc_diablo_full, auc_diablo_mol]
        res_all.append(res)
    res_df_all = pd.DataFrame(res_all, columns=['MBPLS', 'SV', 'DIABLO_null', 'DIABLO_full', 'DIABLO_null_mol'])
    return res_df_all.mean(axis=0).values

        
pathway_idx = int(sys.argv[1])
# pathway_idx = 45

res_dfs = []

effects = [0, 0.25, 0.5, 1, 2 ,3]
# effects = [1]

for e in effects:
    pathway_enriched = mo_paths.index.tolist()[pathway_idx-1]
    res = get_auc([metab_imp, prot_imp], e, [pathway_enriched], mo_paths_dict)
    res_df = pd.DataFrame(res, index=['Multi-View', 'Single-View', 'DIABLO_null', 'DIABLO_full', 'DIABLO_null_mol']).T
    res_df['Effect'] = e
    # res_df['Pathway_set'] = name
    # res_df['Size'] = len(pset)
    res_df['Target'] = pathway_enriched
    res_dfs.append(res_df)


res_all = pd.concat(res_dfs)
res_all.to_csv('COVID/Sensitivity/COVID_sensitivity_AUC_Reactome_v6_'+str(pathway_idx)+'.csv')

