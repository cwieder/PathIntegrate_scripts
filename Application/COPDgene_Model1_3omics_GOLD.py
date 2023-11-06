import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split, StratifiedKFold
from mbpls.mbpls import MBPLS
from sklearn.metrics import roc_auc_score
import sspa
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import scipy.stats as st

# Cross-validated AUC for COPDgene multi-omics PathIntegrate Multi-View and molecular level models

# load metadata
md = pd.read_csv('D:\COPDgene\COPDGene_P1P2_SM_NS_25OCT21.txt', sep='\t')

# load omics datasets
prot = pd.read_csv('A:\home\pathway-integration\COPDgene/COPDgene_proteomics_UniProt_LOG2.csv', index_col=0)
prot = prot.drop(['23931W',
 '21318W',
 '24385A',
 '19819A',
 '17417W',
 '24265Q',
 '23761X',
 '23504D',
 '17828R',
 '21748V'], axis=0)

# load omics datasets
metab = pd.read_csv('A:\home\pathway-integration\COPDgene/COPDgene_metabolomics_CHEBI_mapped_log.csv', index_col=0)
trans = pd.read_csv('D:/COPDgene/Processed/COPDgene_transcriptomics_filt_Q1.csv', index_col=0)
md = md[md['cohort'] == 'Smoker']
md = md[md['finalgold_Phase2'] > -1]
md['finalGoldP2_binary'] = md['finalgold_Phase2'].map({0:0, 1:1, 2:1, 3:1, 4:1})

metab['Group'] = metab.index.map(dict(zip(md['sid'], md['finalGoldP2_binary'])))
metab = metab[metab['Group'].isin([0, 1])]
intersect_samples = set(metab.index.tolist()) & set(prot.index.tolist()) & set(trans.index.tolist())
prot_filt = prot.loc[intersect_samples, :]
metab_filt = metab.loc[intersect_samples, :]
trans_filt = trans.loc[intersect_samples, :]

# md['FEV1_FEC_post_binary'] = [0 if i < 0.7 else 1 for i in md['FEV1_FVC_post']]
# metab['Group'] = metab.index.map(dict(zip(md['sid'], md['FEV1_FEC_post_binary'])))
# metab = metab[metab['Group'].isin([0, 1])]
metab_filt['Group'] = metab_filt.index.map(dict(zip(md['sid'], md["finalGoldP2_binary"])))
prot_filt['Group'] = prot_filt.index.map(dict(zip(md['sid'], md["finalGoldP2_binary"])))
trans_filt['Group'] = trans_filt.index.map(dict(zip(md['sid'], md["finalGoldP2_binary"])))

# # load multi-omics pathways
reactome_cpds = sspa.process_gmt('../Pathway_databases/Reactome_Homo_sapiens_pathways_compounds_R81.gmt')
reactome_uniprot = sspa.process_pathways.process_reactome("Homo sapiens", "../Pathway_databases/UniProt2Reactome_All_Levels.txt")
reactome_ensembl = sspa.process_pathways.process_reactome("Homo sapiens", "D:/Pathway_databases/Ensembl2Reactome_All_Levels.txt")
mo_paths_all = pd.read_csv('D:/Pathway_databases\Reactome_multi_omics_ChEBI_Uniprot_Ensembl.csv', index_col=0)
mo_paths_dict = sspa.utils.pathwaydf_to_dict(mo_paths_all)

# only molecules mapping to pathways 
mols_pathway = sum(list(mo_paths_dict.values()), [])
mols_mapped = list(set(metab_filt.columns.tolist() + prot_filt.columns.tolist() + trans_filt.columns.tolist()) & set(mols_pathway))
metab_filt = metab_filt[metab_filt.columns.intersection(mols_mapped + ['Group'])]
prot_filt = prot_filt[prot_filt.columns.intersection(mols_mapped+ ['Group'])]
trans_filt = trans_filt[trans_filt.columns.intersection(mols_mapped+ ['Group'])]


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

    scores_df_train = pd.DataFrame(scores_train, columns=train.index, index=pathway_ids).T
    scores_df_test = pd.DataFrame(scores_test, columns=test.index, index=pathway_ids).T
    return scores_df_train, scores_df_test


def get_metrics(n_repeats, omics:list, n_comp:int):
    # prepare the cross-validation procedure
    # create model
    aucs = []
    for i in range(0, n_repeats):
        cv_aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in skf.split(metab_filt.iloc[:, :-1], metab_filt['Group']):

            # scale test train separately
            # pipe = Pipeline([('scaler', StandardScaler())])
            pipe = make_pipeline(StandardScaler())
            pipe.set_output(transform="pandas")

            X_train_m, X_test_m = pipe.fit_transform(metab_filt.iloc[train_index, :-1]), pipe.transform(metab_filt.iloc[test_index, :-1])
            X_train_p, X_test_p = pipe.fit_transform(prot_filt.iloc[train_index, :-1]), pipe.transform(prot_filt.iloc[test_index, :-1])
            X_train_t, X_test_t = pipe.fit_transform(trans_filt.iloc[train_index, :-1]), pipe.transform(trans_filt.iloc[test_index, :-1])
            print('Scaling complete, starting sspa')
            kpca_scores_met, kpca_scores_met_test = sspa_kpca(X_train_m, X_test_m, reactome_cpds)
            # kpca_scores_met, kpca_scores_met_test = None, None
            kpca_scores_prot, kpca_scores_prot_test = sspa_kpca(X_train_p, X_test_p, reactome_uniprot)
            # kpca_scores_prot, kpca_scores_prot_test = None, None
            kpca_scores_trans, kpca_scores_trans_test = sspa_kpca(X_train_t, X_test_t, reactome_ensembl)
            # kpca_scores_trans, kpca_scores_trans_test = None, None
            print('sspa generated')

            y_train_m, y_test_m = metab_filt.iloc[train_index, -1], metab_filt.iloc[test_index, -1]

            omics_train = {'M':kpca_scores_met, 'P': kpca_scores_prot, 'T': kpca_scores_trans}
            omics_test = {'M':kpca_scores_met_test, 'P': kpca_scores_prot_test, 'T': kpca_scores_trans_test}
            m1_3o = MBPLS(n_components=n_comp)
            m1_3o.fit([omics_train[i] for i in omics], y_train_m)
            y_pred = m1_3o.predict([omics_test[i] for i in omics])
            print(roc_auc_score(y_test_m, y_pred))
            aucs.append(roc_auc_score(y_test_m, y_pred))
        # aucs.append(np.mean(cv_aucs))
    print(aucs)
    print(np.mean(aucs))
    print(np.std(aucs))
    print('95% CI:')
    ci95 = st.t.interval(confidence=0.95, df=len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs)) 
    print(ci95)
    return(aucs)


def get_metrics_molecular_level(n_repeats, omics:list, n_comp:int):
    # prepare the cross-validation procedure
    # create model
    aucs = []
    for i in range(0, n_repeats):
        cv_aucs = []
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        for train_index, test_index in skf.split(metab_filt.iloc[:, :-1], metab_filt['Group']):
            X_train_m, X_test_m = metab_filt.iloc[train_index, :-1], metab_filt.iloc[test_index, :-1]
            y_train_m, y_test_m = metab_filt.iloc[train_index, -1], metab_filt.iloc[test_index, -1]
            
            X_train_p, X_test_p = prot_filt.iloc[train_index, :-1], prot_filt.iloc[test_index, :-1]
            # y_train_p, y_test_p = kpca_scores_prot.iloc[train_index, -1], kpca_scores_prot.iloc[test_index, -1]

            X_train_t, X_test_t = trans_filt.iloc[train_index, :-1], trans_filt.iloc[test_index, :-1]
            # y_train_t, y_test_t = kpca_scores_trans.iloc[train_index, -1], kpca_scores_trans.iloc[test_index, -1]
            
            omics_train = {'M':X_train_m, 'P': X_train_p, 'T': X_train_t}
            omics_test = {'M':X_test_m, 'P': X_test_p, 'T': X_test_t}
            m1_3o = MBPLS(n_components=n_comp)
            m1_3o.fit([omics_train[i] for i in omics], y_train_m)
            y_pred = m1_3o.predict([omics_test[i] for i in omics])
            aucs.append(roc_auc_score(y_test_m, y_pred))
            print(roc_auc_score(y_test_m, y_pred))
        # aucs.append(np.mean(cv_aucs))
    print(aucs)
    print(np.mean(aucs))
    print(np.std(aucs))
    print('95% CI:')
    ci95 = st.t.interval(confidence=0.95, df=len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs)) 
    print(ci95)

get_metrics(5, ['T', 'P', 'M'], n_comp=5)
get_metrics_molecular_level(5, ['T', 'P', 'M'], n_comp=6)
get_metrics_molecular_level(5, ['T', 'P'], n_comp=7)
get_metrics_molecular_level(5, ['T', 'M'], n_comp=6)
get_metrics_molecular_level(5, ['M', 'P'], n_comp=2)
get_metrics_molecular_level(5, ['M'], n_comp=2)
get_metrics_molecular_level(5, ['P'], n_comp=3)
get_metrics_molecular_level(5, ['T'], n_comp=7)