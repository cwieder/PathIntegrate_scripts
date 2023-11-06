import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split, StratifiedKFold
from mbpls.mbpls import MBPLS
from sklearn.metrics import roc_auc_score
import seaborn as sns
import sspa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

mo_paths = pd.read_csv("Pathway_databases/Reactome_multi_omics_ChEBI_Uniprot.csv", index_col=0, dtype=object) 
reactome_ensembl = sspa.process_pathways.process_reactome("Homo sapiens", "D:/Pathway_databases/Ensembl2Reactome_All_Levels.txt")

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
    

def tune_LVs(omics:list, omics_data, plot=True, plotname=None):
    '''
    Nested cross validation approach for tuning the number of latent variables in MB-PLS multi omics pathway models
    '''
    # Outer loop
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    # for each of 3-folds test a series of possible LVs
    outer_aucs = []
    outer_lvs = []

    # for plotting
    inner_cv_series = []
 
    for n, i in enumerate(omics):
        # generate pathway scores
        if i == 'T':
            kpca_scores = sspa.sspa_kpca(omics_data[n].iloc[:, :-1], reactome_ensembl, 2)
            kpca_scores['Group'] = omics_data[n]['Group']
            omics_sspa[i] = kpca_scores
        else:
            kpca_scores = sspa.sspa_kpca(omics_data[n].iloc[:, :-1], mo_paths, 2)
            kpca_scores['Group'] = omics_data[n]['Group']
            omics_sspa[i] = kpca_scores


        generate scores
    omics_sspa = {'M': kpca_scores_met, 'P': kpca_scores_prot, 'T': kpca_scores_trans}
    kpca_scores_met = omics_data[0]
    kpca_scores_prot = omics_data[1]
    kpca_scores_trans = omics_data[2]

    for train_index, test_index in skf.split(kpca_scores_met.iloc[:, :-1], kpca_scores_met['Group']):
        cv_aucs = {}
        X_train_m, X_test_m = kpca_scores_met.iloc[train_index, :-1], kpca_scores_met.iloc[test_index, :-1]
        y_train_m, y_test_m = kpca_scores_met.iloc[train_index, -1], kpca_scores_met.iloc[test_index, -1]
        
        X_train_p, X_test_p = kpca_scores_prot.iloc[train_index, :-1], kpca_scores_prot.iloc[test_index, :-1]
        # y_train_p, y_test_p = kpca_scores_prot.iloc[train_index, -1], kpca_scores_prot.iloc[test_index, -1]

        X_train_t, X_test_t = kpca_scores_trans.iloc[train_index, :-1], kpca_scores_trans.iloc[test_index, :-1]
        # y_train_t, y_test_t = kpca_scores_trans.iloc[train_index, -1], kpca_scores_trans.iloc[test_index, -1]
        
        omics_train = {'M':X_train_m, 'P': X_train_p, 'T': X_train_t}
        omics_test = {'M':X_test_m, 'P': X_test_p, 'T': X_test_t}
        print([omics_train[i].shape for i in omics])

        # Inner loop
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        # On test set of each outer loop fold optimise the LVs
        # for each of 3-folds test a series of possible LVs
        best_estimators = {}
        best_lvs = []
        # report back the best LV from each of the 3 folds

        print('Commencing inner loop evaluation...')
        for train_index, test_index in skf.split(X_train_m, y_train_m):
            cv_aucs = {}
            X_train_m, X_test_m = kpca_scores_met.iloc[train_index, :-1], kpca_scores_met.iloc[test_index, :-1]
            y_train_m, y_test_m = kpca_scores_met.iloc[train_index, -1], kpca_scores_met.iloc[test_index, -1]
            
            X_train_p, X_test_p = kpca_scores_prot.iloc[train_index, :-1], kpca_scores_prot.iloc[test_index, :-1]
            # y_train_p, y_test_p = kpca_scores_prot.iloc[train_index, -1], kpca_scores_prot.iloc[test_index, -1]

            X_train_t, X_test_t = kpca_scores_trans.iloc[train_index, :-1], kpca_scores_trans.iloc[test_index, :-1]
            # y_train_t, y_test_t = kpca_scores_trans.iloc[train_index, -1], kpca_scores_trans.iloc[test_index, -1]
            
            omics_train = {'M':X_train_m, 'P': X_train_p, 'T': X_train_t}
            omics_test = {'M':X_test_m, 'P': X_test_p, 'T': X_test_t}

            for i in range(1, 10):
                # test LVs from 1-15
                m1_3o = MBPLS(n_components=i)
                m1_3o.fit([omics_train[i] for i in omics], y_train_m)
                y_pred = m1_3o.predict([omics_test[i] for i in omics], y_test_m.tolist())
                cv_aucs[i] = roc_auc_score(y_test_m, y_pred)

            inner_cv_series.append(pd.Series(cv_aucs))
            # max performance and best LV per fold
            best_estimators[max(cv_aucs, key=cv_aucs.get)] = max(cv_aucs.values())
            best_lvs.append(max(cv_aucs, key=cv_aucs.get))
        avg_best_lv = round(np.mean(best_lvs))
        outer_lvs.append(avg_best_lv)
        
        # print(best_estimators)
        # print(best_lvs)
        # print(round(np.mean(best_lvs)))

        # Test the optimised LVs on the left out test data
        print('Commencing outer loop evaluation...')
        model_outer = MBPLS(n_components=avg_best_lv)
        model_outer.fit([omics_train[i] for i in omics], y_train_m)
        y_pred = model_outer.predict([omics_test[i] for i in omics], y_test_m.tolist())
        outer_aucs.append(roc_auc_score(y_test_m, y_pred))
    # print(outer_aucs)
    # print(outer_lvs)
    inner_cv_df = pd.concat(inner_cv_series, axis=1)
    # inner_cv_df.to_csv('Modelling_scripts/LV_tuning/MP_inner.csv')

    if plot:
        sns.set_style('ticks')
        plt.figure(figsize=(8, 6))
        g = sns.lineplot(data=inner_cv_df.melt(ignore_index=False).reset_index(), x='index', y='value', markers=True)
        g.set(xlabel='LV', ylabel='AUC')
        g.set_title(str([i for i in omics]) +' optimal LV (inner loop CV results averaged)')
        plt.savefig('LV_tuning/'+'_'.join(omics)+plotname+'_elbo.png', dpi=300)
        plt.show()

    # Return outer LV that yields the highest AUC
    print('Overall best LV:', outer_lvs[outer_aucs.index(max(outer_aucs))])
    return outer_lvs[outer_aucs.index(max(outer_aucs))] 

# load metadata
md = pd.read_csv('D:\COPDgene\COPDGene_P1P2_SM_NS_25OCT21.txt', sep='\t')

# load omics datasets
prot = pd.read_csv('A:\home\pathway-integration\COPDgene/COPDgene_proteomics_UniProt_scaled.csv', index_col=0)
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
metab = pd.read_csv('../COPDgene/COPDgene_metabolomics_CHEBI_mapped.csv', index_col=0)
trans = pd.read_csv('D:/COPDgene/Processed/COPDgene_transcriptomics_filt_Q1_scaled.csv', index_col=0)

# metab['Group'] = metab.index.map(dict(zip(md['sid'], md["COPD"])))
# metab = metab[metab['Group'].isin([0, 1])]
md = md[md['cohort'] == 'Smoker']
md = md[md['finalgold_Phase2'] > -1]
md['finalGoldP2_binary'] = md['finalgold_Phase2'].map({0:0, 1:1, 2:1, 3:1, 4:1})

metab['Group'] = metab.index.map(dict(zip(md['sid'], md['finalGoldP2_binary'])))
metab = metab[metab['Group'].isin([0, 1])]
intersect_samples = set(metab.index.tolist()) & set(prot.index.tolist()) & set(trans.index.tolist())
prot = prot.loc[intersect_samples, :]
metab = metab.loc[intersect_samples, :]
trans = trans.loc[intersect_samples, :]
metab_scaled = pd.DataFrame(StandardScaler().fit_transform(metab), columns=metab.columns, index=metab.index)
prot_scaled = pd.DataFrame(StandardScaler().fit_transform(prot), columns=prot.columns, index=prot.index)
trans_scaled = pd.DataFrame(StandardScaler().fit_transform(trans), columns=trans.columns, index=trans.index)
metab_scaled['Group'] = metab_scaled.index.map(dict(zip(md['sid'], md["finalGoldP2_binary"])))
prot_scaled['Group'] = prot_scaled.index.map(dict(zip(md['sid'], md["finalGoldP2_binary"])))
trans_scaled['Group'] = trans_scaled.index.map(dict(zip(md['sid'], md["finalGoldP2_binary"])))

mo_paths = pd.read_csv("D:/Pathway_databases/Reactome_multi_omics_ChEBI_Uniprot_Ensembl.csv", index_col=0, dtype=object) 
mo_paths_dict = sspa.utils.pathwaydf_to_dict(mo_paths)

# keep only molecules in pathways 
pathway_mols = list(set(sum(list(mo_paths_dict.values()), [])))
metab_filt = metab_scaled[metab_scaled.columns.intersection(pathway_mols+ ['Group'])]
prot_filt = prot_scaled[prot_scaled.columns.intersection(pathway_mols+ ['Group'])]
trans_filt = trans_scaled[trans_scaled.columns.intersection(pathway_mols+ ['Group'])]

tune_LVs(omics=['M', 'P', 'T'], omics_data=[metab_scaled, prot_scaled, trans_scaled], plot=True, plotname='3_omics_molecular_GOLD')