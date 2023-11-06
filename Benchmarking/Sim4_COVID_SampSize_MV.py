import pandas as pd
import numpy as np
import sspa
from simulate_pathway_signals import SimulateData
from mbpls.mbpls import MBPLS
import statsmodels.api as sm
import sys
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import KernelPCA


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

# load multi-omics pathways
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

pathway_idx = int(sys.argv[1])
# pathway_idx = 1

pathway_enriched = mo_paths.index.tolist()[pathway_idx-1]

sizes = [15, 30, 60]
effects = [0, 0.25, 0.5, 1, 2 ,3]
metrics = {
    'Sample size': [],
    'Effect': [],
    'AUC': [],
    'EV_X': []
    }

for sampsize in sizes:
    for effect in effects:

        met_samp = metab_imp.groupby('Group', group_keys=False).apply(lambda x: x.sample(n=sampsize, replace=False))
        prot_samp = prot_imp[prot_imp.index.isin(met_samp.index)]

        simulated_dset = SimulateData(
            input_data=[met_samp.iloc[:, :-1], prot_samp.iloc[:, :-1]],
            metadata=[met_samp['Group'], prot_samp['Group']],
            pathways=mo_paths_dict,
            enriched_paths=[pathway_enriched]).enrich_paths_base(effect=effect, effect_type='constant', input_type='log')

        metabsim_orig= simulated_dset[0]
        protsim_orig = simulated_dset[1]

        # scale data
        metabsim = (metabsim_orig - metabsim_orig.mean(axis=0)) / metabsim_orig.std(axis=0)
        protsim = (protsim_orig - protsim_orig.mean(axis=0)) / protsim_orig.std(axis=0)

        # kpca_scores_met = sspa.sspa_kpca(metabsim.iloc[:,:-1], mo_paths)
        # kpca_scores_prot = sspa.sspa_kpca(protsim.iloc[:,:-1], mo_paths)

        # get the AUC 
        skf = StratifiedKFold(n_splits=3, shuffle=True)

        # X_train_m, X_test_m, y_train, y_test = train_test_split(metabsim.iloc[:, :-1], metabsim['Group'], test_size=0.25, stratify=True)
        for train_index, test_index in skf.split(metabsim.iloc[:, :-1], metabsim_orig['Group']):
            X_train_m, X_test_m = metabsim.iloc[train_index, :], metabsim.iloc[test_index, :]
            y_train, y_test = metabsim_orig['Group'][train_index], metabsim_orig['Group'][test_index]
            X_train_p, X_test_p = protsim[protsim.index.isin(X_train_m.index)], protsim[protsim.index.isin(X_test_m.index)]

            sspa_train_m, sspa_test_m = sspa_kpca(X_train_m.iloc[:,:-1], X_test_m.iloc[:,:-1], mo_paths)
            sspa_train_p, sspa_test_p = sspa_kpca(X_train_p.iloc[:,:-1], X_test_p.iloc[:,:-1], mo_paths)

            m1_3o = MBPLS(n_components=1)
            m1_3o.fit([sspa_train_m, sspa_train_p], y_train)
            y_pred = m1_3o.predict([sspa_test_m, sspa_test_p], y_test.tolist())
            auc = roc_auc_score(y_test, y_pred)
            print(auc)
            pev = np.sum(m1_3o.explained_var_x_)*100

            metrics['Sample size'].append(sampsize)
            metrics['Effect'].append(effect)
            metrics['AUC'].append(auc)
            metrics['EV_X'].append(pev)

metrics_df = pd.DataFrame(metrics)
# save results to csv
metrics_df.to_csv('COVID/SampSize/COVID_SampSize_AUC_MBPLS_'+pathway_enriched+'.csv')
