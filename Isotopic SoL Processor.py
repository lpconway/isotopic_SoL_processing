from datetime import datetime
import json
import numpy as np
from os import path
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

def site_PSM_counts(peptide, PSM_table):
    loc_counts = PSM_table[PSM_table['Annotated Sequence'] == peptide].groupby(['Label Loc'], as_index=False)['ID'].nunique()
    loc_counts['Label Loc'] = loc_counts['Label Loc'].astype(int)
    loc_counts = loc_counts.sort_values('Label Loc')
    sites = ('; ').join(loc_counts['Label Loc'].astype(str))
    loc_counts = ('; ').join(loc_counts['ID'].astype(str))
    return(pd.Series((loc_counts, sites), index=['Loc PSM Counts', 'Locs']))

def label_type(mod, light_mod, heavy_mod):
    if type(mod) != str:
        return('NA')
    if str(heavy_mod) in mod:
        return('H')
    elif str(light_mod) in mod:
        return('L')
    else:
        return('NA')
    
def get_mean_RTs(length, df):
    return(np.mean(df[df['Peptide Length'] == length]['RT [min]']))

def process_PSMs_isotope(psm_data):
    # If Master Protein Accessions blank, fill with first entry from Protein Accessions, lose the hypenation
    psm_data['Master Protein Accessions'] = psm_data['Master Protein Accessions'].fillna((psm_data['Protein Accessions'].str.extract('(\w+)[;-]?')[0]))
    # Keep only first accession in Master Protein Accessions
    psm_data['Master Protein Accessions'] = psm_data['Master Protein Accessions'].str.extract('(\w+)[;-]?')
    # Strip out flanking peptides, if present
    if np.sum(psm_data['Annotated Sequence'].str.contains('.')) > 1:
        psm_data['Annotated Sequence'] = psm_data['Annotated Sequence'].str.extract('\].(\w+).\[')
    # Make sequences uppercase
    psm_data['Annotated Sequence'] = psm_data['Annotated Sequence'].str.upper()
    
    # Identify heavy/light mod names
    mods = set(psm_data['Modifications'].str.extractall('\((.+?)\)')[0])
    heavy_mod = [x for x in mods if 'heavy' in x.lower()]
    if heavy_mod != []:
        heavy_mod = heavy_mod[0]
    else:
        heavy_mod = False
    light_mod = [x for x in mods if 'light' in x.lower()]
    if light_mod != []:
        light_mod = light_mod[0]
    else:
        light_mod = False
    
    print('Light mod: ' + light_mod)
    print('Heavy mod: ' + heavy_mod)
    
    # Extract label site from heavy/light modifications
    #mod_pattern = '(\w\d+)\([('+str(heavy_mod)+')('+str(light_mod)+')]'
    mod_pattern = '(\w\d+)\\((?:'+str(heavy_mod)+'|'+str(light_mod)+')'
    psm_data['Label Site'] = psm_data['Modifications'].str.extract(mod_pattern)
    psm_data['Label AA'] = psm_data['Label Site'].str.extract('(\w)')
    psm_data['Label Loc'] = psm_data['Label Site'].str.extract('(\d+)')

    # Create unique ID
    psm_data['Modifications'] = psm_data['Modifications'].fillna('')
    psm_data['Scan ID'] = psm_data['First Scan'].astype(str) + '_' + psm_data['File ID']
    psm_data['Scan ID Pep'] = psm_data['Scan ID'] + '_' + psm_data['Annotated Sequence']
    psm_data['Unique ID'] = psm_data['Scan ID Pep'] + '_' + psm_data['Modifications']

    # Calculate peptide length
    psm_data['Peptide Length'] = psm_data['Annotated Sequence'].str.len()

    # Mark labeled peptides
    psm_data['Labeled'] = psm_data['Modifications'].str.contains(str(heavy_mod)) | psm_data['Modifications'].str.contains(str(light_mod))
    psm_data['Labeled'] = psm_data['Labeled'].astype(int)
    
    # Make dataframes for labeled / unlabeled psms
    labeled_psms = psm_data[psm_data['Labeled'] == True]
    unlabeled_psms = psm_data[psm_data['Labeled'] == False]
    light_psms = labeled_psms[labeled_psms['Modifications'].str.contains(str(light_mod))]
    heavy_psms = labeled_psms[labeled_psms['Modifications'].str.contains(str(heavy_mod))]

    psm_data['Label Type'] = psm_data['Modifications'].apply(label_type, args=(light_mod, heavy_mod))

    labeled_psms['Paired'] = labeled_psms['Annotated Sequence'].isin(light_psms['Annotated Sequence']) & labeled_psms['Annotated Sequence'].isin(heavy_psms['Annotated Sequence'])
    paired_index = labeled_psms[['Paired', 'Unique ID']].set_index('Unique ID').to_dict()['Paired']
    psm_data['Paired'] = psm_data['Unique ID'].map(paired_index)

    scan_counts = psm_data['Scan ID'].value_counts(dropna=False)
    psm_data['NumPSMs'] = psm_data['Scan ID'].map(scan_counts)
    # Scale NumPSMs
    psm_data['NumPSMs_scaled'] = (psm_data['NumPSMs'] - np.mean(psm_data['NumPSMs'])) / np.std(psm_data['NumPSMs'])
    
    # Calculate fraction of PSMs assigned to this peptide over all assignments for this scan
    psm_data['Agree_PSMs'] = psm_data[['Scan ID', 'Scan ID Pep']].apply(lambda x : len(psm_data[psm_data['Scan ID Pep'] == x[1]]) / len(psm_data[psm_data['Scan ID'] == x[0]]), axis=1)

    # Combine DeltaScores and scale 
    psm_data['ScoreDiff'] = psm_data['DeltaScore'].fillna(psm_data['DeltaCn'])
    psm_data['ScoreDiff_scaled'] = (psm_data['ScoreDiff'] - np.mean(psm_data['ScoreDiff'])) / np.std(psm_data['ScoreDiff'])

    # Calculate mean RTs for labeled and unlabeled peptides for each peptide length
    RT_df = pd.DataFrame(index = list(set(psm_data['Peptide Length'])))
    RT_df['unlabeled_mean_RT'] = RT_df.index.map(lambda x : get_mean_RTs(x, unlabeled_psms))
    RT_df['labeled_mean_RT']  = RT_df.index.map(lambda x : get_mean_RTs(x, labeled_psms[labeled_psms['Paired']]))

    # Calculate difference between experimental RT and mean RT for unlabeled petide of that length
    psm_data = psm_data.join(RT_df, on='Peptide Length')
    psm_data['RT_diff_unlabeled'] = psm_data['RT [min]'] - psm_data['unlabeled_mean_RT']
    psm_data['RT_diff_unlabeled_scaled'] = (psm_data['RT_diff_unlabeled'] - np.mean(psm_data['RT_diff_unlabeled'])) / np.std(psm_data['RT_diff_unlabeled'])

    # Generate Protein/Peptide identifiers
    psm_data['ProtID_Pep'] = psm_data['Master Protein Accessions'] + '_' + psm_data['Annotated Sequence']
    
    # Count PSMs per protein, labeled PSMs per protein, etc
    protein_psms = psm_data['Master Protein Accessions'].value_counts()
    protein_labeled = psm_data[['Master Protein Accessions', 'Labeled', 'Paired']].groupby('Master Protein Accessions').sum()
    protein_labeled.columns = ['Protein Labeled PSMs', 'Protein Paired PSMs']
    protein_peptides = psm_data[['Master Protein Accessions', 'Annotated Sequence']].groupby('Master Protein Accessions').nunique()['Annotated Sequence']
    protein_labeled_unique = labeled_psms[['Master Protein Accessions', 'Annotated Sequence']].groupby('Master Protein Accessions').nunique()['Annotated Sequence']
    protein_paired_unique = labeled_psms[labeled_psms['Paired']][['Master Protein Accessions', 'Annotated Sequence']].groupby('Master Protein Accessions').nunique()['Annotated Sequence']
    psm_data['Protein PSMs'] = psm_data['Master Protein Accessions'].map(protein_psms)
    psm_data = psm_data.join(protein_labeled, on='Master Protein Accessions')
    psm_data['Protein Unique Peptides'] = psm_data['Master Protein Accessions'].map(protein_peptides)
    psm_data['Protein Unique Labeled Peptides'] = psm_data['Master Protein Accessions'].map(protein_labeled_unique)
    psm_data['Protein Unique Paired Peptides'] = psm_data['Master Protein Accessions'].map(protein_paired_unique)

    #Split unlabeled psms in half (by every other row) to make unique negative sets for models
    train_neg  = unlabeled_psms.iloc[range(0, len(unlabeled_psms), 2)]
    test_neg  = unlabeled_psms.iloc[range(1, len(unlabeled_psms), 2)]

    train_set = pd.concat([psm_data[psm_data['Paired'] == True], psm_data[psm_data['Unique ID'].isin(train_neg['Unique ID'])]])
    test_set = pd.concat([psm_data[(psm_data['Paired'] != True) & (psm_data['Labeled'] == True)], psm_data[psm_data['Unique ID'].isin(test_neg['Unique ID'])]])

    formula = 'Labeled ~ NumPSMs_scaled + ScoreDiff_scaled + Agree_PSMs + RT_diff_unlabeled_scaled'
    model = smf.glm(formula = formula, data=train_set, family=sm.families.Binomial())
    result = model.fit()
    print(result.summary())

    train_pred = result.predict(train_set)
    test_pred = result.predict(test_set)

    train_out = train_set.copy()
    train_out['GLM Prob'] = train_pred
    train_out['Dataset'] ='Train'

    test_out = test_set.copy()
    test_out['GLM Prob'] = test_pred
    test_out['Dataset'] ='Test'

    processed_PSMs = pd.concat([train_out, test_out])
    processed_PSMs['GLM Pred'] = processed_PSMs['GLM Prob'] > 0.5
    processed_PSMs['Confidence'] = ['High' if x >= 0.85 else 'Medium' if x >= 0.5 else 'Low' for x in processed_PSMs['GLM Prob']]

    processed_PSMs_sorted = processed_PSMs.sort_values(['Scan ID', 'NumPSMs', 'XCorr'])
    unique_scan_df = processed_PSMs_sorted.drop_duplicates('Scan ID')
    pep_df = unique_scan_df[['Annotated Sequence', 'Master Protein Accessions']].drop_duplicates()
    pep_df['Unique Scans'] = pep_df['Annotated Sequence'].apply(lambda x : len(unique_scan_df[unique_scan_df['Annotated Sequence'] == x]))
    pep_df['PSMs'] = pep_df['Annotated Sequence'].apply(lambda x : len(processed_PSMs_sorted[processed_PSMs_sorted['Annotated Sequence'] == x]))
    pep_df = pep_df.join(processed_PSMs[['Annotated Sequence', 'GLM Prob', 'XCorr', 'ScoreDiff', 'NumPSMs', 'RT_diff_unlabeled']].groupby('Annotated Sequence').mean(), on='Annotated Sequence')
    GLM_max = processed_PSMs[['Annotated Sequence', 'GLM Prob']].groupby('Annotated Sequence').max()
    GLM_max.columns = ['GLM Max']
    pep_df = pep_df.join(GLM_max, on='Annotated Sequence')
    processed_PSMs_sorted['ID'] = processed_PSMs_sorted.index
    pep_df[['Loc PSM Counts', 'Locs']] = pep_df['Annotated Sequence'].apply(lambda x : site_PSM_counts(x, processed_PSMs_sorted))
    return(pep_df)

def main(argv):
   if len(argv) != 1:
      raise ValueError("Requires one argument for node_args.json path")
   print('PSM path: ' + argv[0])
   json_path = argv[0]
   f = open(json_path)
   json_data = json.load(f)
   
   psm_file = 'TargetPeptideSpectrumMatch.txt'
   psm_file = json_data['Tables'][0]['DataFile']
   result_path = json_data['ResultFilePath']
   output_path, expt_name = path.split(result_path)
   expt_name = path.splitext(expt_name)[0]
   output_path = json_data['ExpectedResponsePath']
   psm_data = pd.read_csv(psm_file, sep='\t')
   processed_PSMs = process_PSMs_isotope(psm_data)
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   processed_PSMs.to_csv(output_path+'\\'+expt_name+'_Processed_SOL_'+timestamp+'.tsv', sep='\t') 

if __name__ == "__main__":
   main(sys.argv[1:])
