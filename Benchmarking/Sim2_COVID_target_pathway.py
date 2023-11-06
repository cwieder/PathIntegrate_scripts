import pandas as pd
import numpy as np
import sspa
from simulate_pathway_signals import SimulateData
from mbpls.mbpls import MBPLS
import statsmodels.api as sm
import sys
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression
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

prot = pd.read_csv('../COVID/prot_uniprot.csv', index_col=0)
metab = pd.read_csv('../COVID/metab_chebi.csv', index_col=0)
# impute missing values with median
prot_imp = prot.iloc[:, :-1].fillna(prot.iloc[:, :-1].median())
metab_imp = metab.iloc[:, :-1].fillna(metab.iloc[:, :-1].median())

prot_imp['Group'] = [1]*63 + [0]*64
metab_imp['Group'] = [1]*63 + [0]*64

concat_omics = pd.concat([metab_imp.iloc[:, :-1], prot_imp.iloc[:, :-1]], axis=1)

# load multi-omics pathways
mo_paths = pd.read_csv("../Pathway_databases/Reactome_multi_omics_ChEBI_Uniprot.csv", index_col=0, dtype=object)

def filt_pathways(path_df):
    pathdict = sspa.utils.pathwaydf_to_dict(path_df)
    compounds_present = metab.columns.tolist() + prot.columns.tolist()
    pathways_present = {k: v for k, v in pathdict.items() if len([i for i in compounds_present if i in v]) > 2}
    path_coverage = {k: [i for i in v if i in compounds_present] for k, v in pathways_present.items()}
    filt_paths = path_df[path_df.index.isin(pathways_present)]
    return filt_paths, path_coverage

mo_paths, path_coverage = filt_pathways(mo_paths)
mo_paths_dict = sspa.utils.pathwaydf_to_dict(mo_paths)

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

def overlap_coefficient(list1, list2):
    # Szymkiewicz–Simpson coefficient
    intersection = len(list(set(list1).intersection(list(set(list2)))))
    smaller_set = min(len(list1), len(list2))
    return float(intersection) / smaller_set


def id_target_pathway(dsets, effect, enrich_paths, paths_dict):
    simulated_dset = SimulateData(
    input_data=[dsets[0].iloc[:, :-1], dsets[1].iloc[:, :-1]],
    metadata=[dsets[0]['Group'], dsets[1]['Group']],
    pathways=paths_dict,
    enriched_paths=enrich_paths
    ).enrich_paths_base(effect=effect, effect_type='constant', input_type='log')

    metabsim_orig = simulated_dset[0]
    protsim_orig = simulated_dset[1]

    # scale data
    metabsim = (metabsim_orig - metabsim_orig.mean(axis=0)) / metabsim_orig.std(axis=0)
    protsim = (protsim_orig - protsim_orig.mean(axis=0)) / protsim_orig.std(axis=0)
    
    # scores for mv and diablo
    kpca_scores_met = sspa.sspa_kpca(metabsim.iloc[:,:-1], mo_paths)
    kpca_scores_prot = sspa.sspa_kpca(protsim.iloc[:,:-1], mo_paths)
    kpca_scores_met.columns += 'M'
    kpca_scores_prot.columns += 'P'
    kpca_scores_prot['Group'] = protsim['Group']
    labels = pd.factorize(kpca_scores_prot['Group'])[0]

    pathcols = kpca_scores_met.columns.tolist() + kpca_scores_prot.iloc[:, :-1].columns.tolist()

    # scores for sv
    met_prot_data = metabsim.merge(protsim, how='inner', right_index=True, left_index=True)
    kpca_scores = sspa.sspa_kpca(met_prot_data, mo_paths)
    kpca_scores['Group'] = metabsim['Group']
    pathcols_sv = kpca_scores.iloc[:, :-1].columns.tolist()
    labels_sv = pd.factorize(kpca_scores['Group'])[0]

    # SV ---------------------------------------------------------------------------
    m0 = PLSRegression(1)
    m0.fit(kpca_scores.iloc[:, :-1].to_numpy(), labels_sv)
    m0_VIP = VIP_PLS(kpca_scores.iloc[:, :-1].to_numpy(), labels_sv, m0)

    res_sv= pd.DataFrame(m0_VIP, columns=['VIP'], index=pathcols_sv)
    res_sv['Model'] = 'Single-View'
    res_sv['Effect'] = effect
    res_sv['Target'] = [True if i in enrich_paths else False for i in pathcols_sv]

    # MV ---------------------------------------------------------------------------
    m1 = MBPLS(n_components=1)
    m1.fit([kpca_scores_met, kpca_scores_prot.iloc[:, :-1]], labels)
    m1_VIP = VIP_multiBlock(m1.W_, m1.Ts_, m1.P_, m1.V_)
    
    res_m1 = pd.DataFrame(m1_VIP, columns=['VIP'], index=pathcols)
    res_m1['Model'] = 'Multi-View'
    res_m1['Effect'] = effect
    res_m1['Target'] = [True if i[:-1] in enrich_paths else False for i in pathcols]

    # DIABLO loadings --------------------------------------------------------------
    
    #create design matrix NULL
    design_mat = pd.DataFrame(np.full((2, 2), 0), index=['Metabolomics_kpca', 'Proteomics_kpca'], columns=['Metabolomics_kpca', 'Proteomics_kpca'])
    np.fill_diagonal(design_mat.values, 0)

    with (ro.default_converter + pandas2ri.converter).context():
        r_metab_kpca = ro.conversion.get_conversion().py2rpy(kpca_scores_met)
        r_prot_kpca = ro.conversion.get_conversion().py2rpy(kpca_scores_prot.iloc[:, :-1])
        r_design_mat = ro.conversion.get_conversion().py2rpy(design_mat)
        r_labels_kpca = ro.IntVector(labels)
    X_matrices = ro.ListVector([('Metabolomics_kpca', r_metab_kpca), ('Proteomics_kpca', r_prot_kpca)])
    diablo_res = mixomics.block_plsda(X=X_matrices, Y=r_labels_kpca, ncomp=1, design=r_design_mat)

    # get loadings 
    met_loadings= pd.DataFrame(diablo_res.rx2("loadings").rx2('Metabolomics_kpca'), index=kpca_scores_met.columns)
    prot_loadings= pd.DataFrame(diablo_res.rx2("loadings").rx2('Proteomics_kpca'), index=kpca_scores_prot.iloc[:, :-1].columns)

    # stack loadings
    loadings_all = pd.concat([met_loadings, prot_loadings], axis=0)
    
    res_df_loadings = pd.DataFrame(loadings_all.values, columns=['Loading'], index=pathcols)
    res_df_loadings['Model'] = 'DIABLO_loadings'
    res_df_loadings['Effect'] = e
    res_df_loadings['Target'] = [True if i[:-1] in enrich_paths else False for i in pathcols]

    # compute overlap between mols in target and each other pathway
    target_mols = paths_dict[enrich_paths[0]]
    overlaps = [overlap_coefficient(target_mols, path_coverage[i[:-1]]) for i in pathcols]
    res_m1['OC'] = overlaps
    res_df_loadings['OC'] = overlaps

    overlaps_sv = [overlap_coefficient(target_mols, path_coverage[i]) for i in pathcols_sv]
    res_sv['OC'] = overlaps_sv

    # sparse DIABLO -------------------------------------------------------
    # use same design and matrices as DIABLO above

    test_keepX = ro.ListVector([('Metabolomics_kpca', ro.IntVector([200, 100, 50, 25, 5]) ), ('Proteomics_kpca', ro.IntVector([200, 100, 50, 25, 5]))])
    tune_features = mixomics.tune_block_splsda(X_matrices, r_labels_kpca, ncomp = 1, 
                              test_keepX = test_keepX, design = r_design_mat,
                              validation = 'Mfold', folds = 5, nrepeat = 1)
    n_features_met = tune_features.rx2('choice.keepX')[0][0]
    n_features_prot = tune_features.rx2('choice.keepX')[1][0]
    
    loadings_full = []
    for i in range(0, 25):
        # bootstrap 100 times 50% of the sample population
        # balanced classes
        met_sample = metab_imp[metab_imp['Group'] == 0].sample(n=30, replace=True).append(metab_imp[metab_imp['Group'] == 1].sample(n=30, replace=True))
        prot_sample = prot_imp.loc[met_sample.index, :]
        # rename samples to avoid duplicated indexes
        met_sample = met_sample.reset_index(drop=True)
        prot_sample = prot_sample.reset_index(drop=True)
        
        simulated_dset = SimulateData(
        input_data=[met_sample.iloc[:, :-1], prot_sample.iloc[:, :-1]],
        metadata=[met_sample['Group'], prot_sample['Group']],
        pathways=paths_dict,
        enriched_paths=enrich_paths).enrich_paths_base(effect=effect, effect_type='constant', input_type='log')

        metabsim_orig = simulated_dset[0]
        protsim_orig = simulated_dset[1]

        # scale data
        metabsim = (metabsim_orig - metabsim_orig.mean(axis=0)) / metabsim_orig.std(axis=0)
        protsim = (protsim_orig - protsim_orig.mean(axis=0)) / protsim_orig.std(axis=0)
        
        # scores for mv and diablo
        kpca_scores_met = sspa.sspa_kpca(metabsim.iloc[:,:-1], mo_paths)
        kpca_scores_prot = sspa.sspa_kpca(protsim.iloc[:,:-1], mo_paths)
        kpca_scores_met.columns += 'M'
        kpca_scores_prot.columns += 'P'
        kpca_scores_prot['Group'] = protsim_orig['Group']
        labels = pd.factorize(kpca_scores_prot['Group'])[0]

        pathcols = kpca_scores_met.columns.tolist() + kpca_scores_prot.iloc[:, :-1].columns.tolist()

        #create design matrix NULL
        design_mat = pd.DataFrame(np.full((2, 2), 0), index=['Metabolomics_kpca', 'Proteomics_kpca'], columns=['Metabolomics_kpca', 'Proteomics_kpca'])
        np.fill_diagonal(design_mat.values, 0)
    
        with (ro.default_converter + pandas2ri.converter).context():
            r_metab_kpca = ro.conversion.get_conversion().py2rpy(kpca_scores_met)
            r_prot_kpca = ro.conversion.get_conversion().py2rpy(kpca_scores_prot.iloc[:, :-1])
            r_design_mat = ro.conversion.get_conversion().py2rpy(design_mat)
            r_labels_kpca_sparse = ro.IntVector(kpca_scores_prot['Group'].tolist())
        X_matrices = ro.ListVector([('Metabolomics_kpca', r_metab_kpca), ('Proteomics_kpca', r_prot_kpca)])
        keep_features = ro.ListVector([('Metabolomics_kpca', ro.IntVector([n_features_met]) ), ('Proteomics_kpca', ro.IntVector([n_features_prot]))])
        diablo_res_sparse = mixomics.block_splsda(X=X_matrices, Y=r_labels_kpca_sparse, ncomp=1, design=r_design_mat, keepX = keep_features)

        # get loadings 
        met_loadings_sparse = pd.DataFrame(diablo_res_sparse.rx2("loadings").rx2('Metabolomics_kpca'), index=kpca_scores_met.columns)
        prot_loadings_sparse = pd.DataFrame(diablo_res_sparse.rx2("loadings").rx2('Proteomics_kpca'), index=kpca_scores_prot.iloc[:, :-1].columns)

        # stack loadings
        loadings_all = pd.concat([met_loadings_sparse, prot_loadings_sparse], axis=0)
        loadings_full.append(loadings_all[0].to_numpy())
    
    # count how many non zero coefficients per column (across bootstrap)
    loadings_full_arr = np.array(loadings_full)
    non_zero_counts = np.count_nonzero(loadings_full_arr, axis=0)
    
    # divide to get propotion of times each var was significant 
    non_zero_counts_proportion = non_zero_counts / 25
    
    res_df_sparse = pd.DataFrame(non_zero_counts_proportion, columns=['Non-zero proportion'], index=pathcols)
    res_df_sparse['Effect'] = e
    res_df_sparse['Target'] = [True if i[:-1] in enrich_paths else False for i in pathcols]
    res_df_sparse['OC'] = overlaps

    return res_m1, res_sv, res_df_loadings, res_df_sparse

pathway_idx = int(sys.argv[1])
pathway_enriched = mo_paths.index.tolist()[pathway_idx-1]
res_dfs_mv = []
res_dfs_sv = []
res_dfs_diablo_loadings = []
res_dfs_diablo_sparse = []

# for e in [0]:
for e in [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3]:

    r_m1, r_sv, diablo_loadings, diablo_sparse = id_target_pathway([metab_imp, prot_imp], e, [pathway_enriched], mo_paths_dict)
    res_dfs_mv.append(r_m1)
    res_dfs_sv.append(r_sv)
    res_dfs_diablo_loadings.append(diablo_loadings)
    res_dfs_diablo_sparse.append(diablo_sparse)

effects_df_mv = pd.concat(res_dfs_mv, axis=0)
effects_df_sv = pd.concat(res_dfs_sv, axis=0)
effects_df_diablo_loading = pd.concat(res_dfs_diablo_loadings, axis=0)
effects_df_diablo_sparse = pd.concat(res_dfs_diablo_sparse, axis=0)
# save results to csv

effects_df_mv.to_csv('COVID/Target/target_MBPLS_Reactome_v2_'+pathway_enriched+'.csv')
effects_df_sv.to_csv('COVID/Target/target_SV_Reactome_v2_'+pathway_enriched+'.csv')
effects_df_diablo_loading.to_csv('COVID/Target/target_DIABLO_loadings_Reactome_v2_'+pathway_enriched+'.csv')
effects_df_diablo_sparse.to_csv('COVID/Target/target_DIABLO_sparse_Reactome_v2_'+pathway_enriched+'.csv')