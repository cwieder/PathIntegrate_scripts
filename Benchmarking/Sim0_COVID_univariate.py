import pandas as pd
import numpy as np
import sspa
from simulate_pathway_signals import SimulateData
from mbpls.mbpls import MBPLS
import statsmodels.api as sm
import sys
from scipy.stats import mannwhitneyu, combine_pvalues
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import KernelPCA

# load multi-omics pathways
prot = pd.read_csv('../COVID/prot_uniprot.csv', index_col=0)
metab = pd.read_csv('../COVID/metab_chebi.csv', index_col=0)
# impute missing values with median
prot_imp = prot.iloc[:, :-1].fillna(prot.iloc[:, :-1].median())
metab_imp = metab.iloc[:, :-1].fillna(metab.iloc[:, :-1].median())

prot_imp['Group'] = [1]*63 + [0]*64
metab_imp['Group'] = [1]*63 + [0]*64
concat_omics = pd.concat([prot.iloc[:, :-1], metab.iloc[:, :-1]], axis=1)

# load multi-omics pathways
mo_paths = pd.read_csv("../Pathway_databases/Reactome_multi_omics_ChEBI_Uniprot.csv", index_col=0, dtype=object)
def filt_pathways(path_df):
    pathdict = sspa.utils.pathwaydf_to_dict(path_df)
    compounds_present = metab.columns.tolist() + prot.columns.tolist()
    pathways_present = {k: v for k, v in pathdict.items() if len([i for i in compounds_present if i in v]) > 2}
    path_coverage = {k: len([i for i in v if i in compounds_present]) for k, v in pathways_present.items()}
    filt_paths = path_df[path_df.index.isin(pathways_present)]
    return filt_paths, path_coverage

mo_paths, cvrg = filt_pathways(mo_paths)
mo_paths_dict = sspa.utils.pathwaydf_to_dict(mo_paths)

def sspa_kpca(mat, pathways, target, min_entity=2):

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


    single_pathway_matrix = mat.drop(mat.columns.difference(pathways[target]), axis=1)

    kpca = KernelPCA(n_components=2, kernel="rbf")
    new_data = kpca.fit_transform(single_pathway_matrix)
    scores_df = pd.DataFrame(new_data[:, 0], index=mat.index, columns=[target])
    return scores_df


def get_model1_rankings(enrich_paths:list, effect=1):
    simulated_dset = SimulateData(
    input_data=[metab.iloc[:, :-1], prot.iloc[:, :-1]],
    metadata=[metab['Group'], prot['Group']],
    pathways=mo_paths_dict,
    enriched_paths=enrich_paths).enrich_paths_base(effect=effect, effect_type='constant', input_type='log')

    metabsim_orig = simulated_dset[0]
    protsim_orig = simulated_dset[1]

    # scale the data
    metabsim = (metabsim_orig - metabsim_orig.mean(axis=0)) / metabsim_orig.std(axis=0)
    protsim = (protsim_orig - protsim_orig.mean(axis=0)) / protsim_orig.std(axis=0)

    met_prot_data = metabsim.iloc[:, :-1].merge(protsim.iloc[:, :-1], how='inner', right_index=True, left_index=True)
    target_molecules = met_prot_data.loc[:, met_prot_data.columns.isin(mo_paths_dict[enrich_paths[0]])]

    kpca_scores = sspa_kpca(met_prot_data, mo_paths_dict, enrich_paths[0], min_entity=2)
    target_pathway = kpca_scores[enrich_paths[0]]

    u, pval_pathway = mannwhitneyu(target_pathway[protsim_orig['Group']==1], target_pathway[protsim_orig['Group']==0])
    u, pvals_molecule = mannwhitneyu(target_molecules[protsim_orig['Group']==1], target_molecules[protsim_orig['Group']==0])

    # correct using bonferroni (the number of molecules in the pathway across all pathways being tested)
    # then dont need to correct p-vals at later stage
    p_vals_molecule_adj = [p * 18155 for p in pvals_molecule]
    pval_pathway_adj = pval_pathway * 1290
    pval_pathway_adj = min(pval_pathway_adj, 1)

    len_pathway = cvrg[enrich_paths[0]]

    # if at least 50% molecules in pathway have sig p, consider it significant and perform fisher's method
    if len([i for i in p_vals_molecule_adj if i < 0.05]) >= len_pathway/2:
        # combine the pvals molecule using the Fisher method
        pval_molecule = combine_pvalues(p_vals_molecule_adj, method='fisher')[1]
    else:
        pval_molecule = 1

    return pd.DataFrame([pval_molecule, pval_pathway_adj],
                         index=['Molecule-Bonf-combined', 'Pathway'])

# pathway_idx = int(sys.argv[1])
# pathway_idx = 3

# pathway_enriched = mo_paths.index.tolist()[pathway_idx-1]

for pathway_enriched in mo_paths.index.tolist():
    res_dfs = []
    for i in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.25, 0.5]:
    # for i in [0.05 ]:
        res_df = get_model1_rankings(enrich_paths=[pathway_enriched], effect=i)
        res_df['Effect'] = i
        res_df['Target'] = pathway_enriched
        res_dfs.append(res_df)
    effects_df = pd.concat(res_dfs)
    # print(effects_df)

    # save results to csv
    effects_df.to_csv('COVID/Univar/COVID_univar'+pathway_enriched+'.csv')
