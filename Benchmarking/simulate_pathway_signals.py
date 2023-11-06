import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sspa

class SimulateData:
    """
    Class for creation of semi-synthetic multi-omics data with known pathway signals
    """    
    def __init__(self, input_data, metadata, pathways, enriched_paths):
        """_summary_

        Args:
            input_data (_list_): _description_
            metadata (_series_): _description_
            pathways (_dict_): _description_
            enriched_paths (_list_): _description_
        """ 
        self.input_data = input_data
        self.input_data_filt = []
        self.metadata = metadata
        self.metadata_filt = []
        self.metadata_perm = []
        self.pathways = pathways
        self.enriched_paths = enriched_paths
        self.enriched_mols = []
        self.sim_data = []

    def sample_permutation(self):
        """
        Permute sample class labels to wipe out original omics signal
        """
        
        # only filter if multi omics data supplied
        if len(self.input_data) > 1:
            # get sample intersection between n omics dataframes
            intersect_samples = list(set.intersection(*[set(df.index.tolist()) for df in self.input_data]))

            # filter each dataframe to contain same samples
            self.input_data_filt = [df.loc[intersect_samples, :] for df in self.input_data]

            n_samples = len(intersect_samples)
            # n_features = [df.shape[1] for df in self.input_data]
            # n_cases = n_samples / 2
            # n_controls = n_samples / 2

            # filter metadata
            self.metadata_filt = [i[i.index.isin(intersect_samples)] for i in self.metadata]
        else:
            self.input_data_filt = [df for df in self.input_data]
            self.metadata_filt = [i for i in self.metadata]

        md_binary = pd.factorize(self.metadata_filt[0].tolist())[0]

        # permute sample metadata
        rng = np.random.default_rng()
        self.metadata_perm = rng.permutation(md_binary)


    def enrich_paths_base(self, effect=1, effect_type='var', input_type='log'):
        """
        Enrich specified pathways in n omics datasets 
        """

        # fill in the data with permuted samples
        self.sample_permutation()

        # Get metabolites and proteins to be enriched from each pathway 
        enriched_mols = list(sum([self.pathways[i] for i in self.enriched_paths], []))
        self.enriched_mols = enriched_mols
        enriched_proteins = [i for i in enriched_mols if i.startswith("P|O|Q")]
        enriched_metabs = np.setdiff1d(enriched_mols, enriched_proteins)

        # add signal just to one group in each omics 
        indices_0 = np.argwhere(self.metadata_perm == 0).ravel()  # control
        indices_1 = np.argwhere(self.metadata_perm == 1).ravel() 

        for df in self.input_data_filt:
            df_enriched = df.copy()


            if input_type == 'zscore':
                if effect_type == 'constant':
                    df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] = df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] * (1+effect)
                # add constant based on variance of the feature 
                if effect_type == 'var':
                    # get standard deviations
                    sd = df_enriched.iloc[:, df_enriched.columns.isin(enriched_mols)].std()
                    # normalise effect size by SD
                    alpha  = effect / sd
                    df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] = df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] * (1+alpha)

            if input_type == 'log':
                if effect_type == 'constant':
                    df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] = df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] + effect
                # add constant based on variance of the feature 
                if effect_type == 'var':
                    # get standard deviations
                    sd = df_enriched.iloc[:, df_enriched.columns.isin(enriched_mols)].std()
                    # normalise effect size by SD
                    alpha  = effect / sd
                    df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] = df_enriched.iloc[indices_1, df_enriched.columns.isin(enriched_mols)] + alpha

            # Add group labels
            df_enriched["Group"] = self.metadata_perm
            self.sim_data.append(df_enriched)
        
        return self.sim_data

