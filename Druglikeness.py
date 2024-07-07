#!/usr/bin/env python
# coding: utf-8

# # Functions to calculate druglikeness properties
# ## Dr. Ricardo Romero
# ### Natural Sciences Department, UAM-C

# In[ ]:


# Load libraries
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski, Crippen, FilterCatalog, rdMolDescriptors


# In[ ]:


# Calculate druglikeness rules violations
def calculate_rule_violations(smiles):
    violations = []
    
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        
        # Lipinski's Rule of Five
        lipinski_violations = Lipinski.NumHAcceptors(mol) > 10 or Lipinski.NumHDonors(mol) > 5 or \
                             Descriptors.MolLogP(mol) > 5 or Descriptors.MolWt(mol) > 500
        violations.append(int(lipinski_violations))
        
        # Ghose Filter
        ghose_violations = Crippen.MolMR(mol) > 130 or Crippen.MolMR(mol) < 40 or \
                           Descriptors.MolLogP(mol) > 5.6 or Descriptors.MolLogP(mol) < -0.4 or \
                           Descriptors.MolWt(mol) > 480 or Descriptors.MolWt(mol) < 160 or \
                           mol.GetNumAtoms() > 70 or  mol.GetNumAtoms() < 20
        violations.append(int(ghose_violations))
        
        # Veber's Rule
        veber_violations = Descriptors.NumRotatableBonds(mol) > 10 or rdMolDescriptors.CalcTPSA(mol) > 140 
        violations.append(int(veber_violations))
        
        # Egan's Rule
        egan_violations = Descriptors.MolLogP(mol) > 5.88 or rdMolDescriptors.CalcTPSA(mol) > 131.6 
        violations.append(int(egan_violations))
        
        # Muegge's Rule
        muegge_violations = Descriptors.MolLogP(mol) < -0.4 or Descriptors.MolLogP(mol) > 5.6 or \
                            Descriptors.MolWt(mol) < 200 or Descriptors.MolWt(mol) > 600 or \
                            Descriptors.NumHDonors(mol) > 5 or Descriptors.NumHAcceptors(mol) > 10
        violations.append(int(muegge_violations))
        
    return violations


# In[ ]:


# Evaluate leadlikeness properties
def check_leadlikeness(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES"

    # Calculate properties
    mw = Descriptors.MolWt(mol)
    xlogp = Descriptors.MolLogP(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    # Check leadlikeness criteria
    if 250 <= mw <= 350 and xlogp <= 3.5 and num_rotatable_bonds <= 7:
        return True, mw, xlogp, num_rotatable_bonds
    else:
        return False, mw, xlogp, num_rotatable_bonds
    
def evaluate_smiles(smiles_list):
    results = []
    for smiles in smiles_list:
        leadlike, mw, xlogp, num_rotatable_bonds = check_leadlikeness(smiles)
        results.append({
            'smiles': smiles,
            'Leadlike': leadlike,
            'MW': mw,
            'XLOGP': xlogp,
            'Rotatable Bonds': num_rotatable_bonds
        })
    return pd.DataFrame(results)

# Main function to read input and save output
def main(input_csv, output_csv):
    # Read the input CSV file
    df_input = pd.read_csv(input_csv)

    # Ensure the input CSV has a 'SMILES' column
    if 'smiles' not in df_input.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    # Evaluate the SMILES strings
    df_results = evaluate_smiles(df_input['smiles'].tolist())

    # Merge the results with the original data
    df_output = pd.concat([df_input, df_results.drop(columns=['smiles'])], axis=1)

    # Save the results to a new CSV file
    df_output.to_csv(output_csv, index=False)

    # Print the results
    print(df_output)
    
# Example usage
if __name__ == "__main__":
    input_csv = 'data.csv'  # Path to the input CSV file
    output_csv = 'data_leadlikeness_results.csv'  # Path to the output CSV file
    main(input_csv, output_csv)


# In[ ]:


# Calculate structure alerts and BBB penetration
# Load the SMILES data from a CSV file
def load_smiles_data(file_path):
    df = pd.read_csv(file_path)
    smiles_list = df['smiles'].tolist()
    return df, smiles_list

# Calculate Brenk alerts
def calculate_brenk_alerts(molecule):
    brenk_catalog = FilterCatalog.FilterCatalogParams(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(brenk_catalog)
    return catalog.HasMatch(molecule)

# Calculate PAINS alerts
def calculate_pains_alerts(molecule):
    pains_catalog = FilterCatalog.FilterCatalogParams(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog = FilterCatalog.FilterCatalog(pains_catalog)
    return catalog.HasMatch(molecule)

# Calculate BBB penetration
def calculate_bbb_penetration(molecule):
    logP = Descriptors.MolLogP(molecule)
    psa = rdMolDescriptors.CalcTPSA(molecule)
    mw = Descriptors.MolWt(molecule)
    return logP > 2 and psa < 90 and mw < 450

# Analyze molecules
def analyze_molecules(smiles_list):
    results = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            brenk_alerts = calculate_brenk_alerts(molecule)
            pains_alerts = calculate_pains_alerts(molecule)
            bbb_penetration = calculate_bbb_penetration(molecule)
            results.append({
                'smiles': smiles,
                'brenk_alerts': brenk_alerts,
                'pains_alerts': pains_alerts,
                'bbb_penetration': bbb_penetration
            })
        else:
            results.append({
                'smiles': smiles,
                'brenk_alerts': None,
                'pains_alerts': None,
                'bbb_penetration': None
            })
    return results

# Main function
def main(file_path):
    df, smiles_list = load_smiles_data(file_path)
    results = analyze_molecules(smiles_list)
    results_df = pd.DataFrame(results)
    
    # Merge the results with the original dataframe
    df = pd.merge(df, results_df, on='smiles')
    
    # Save the merged dataframe to a new CSV file
    df.to_csv('data_alerts.csv', index=False)
    print(df)
    
# Example usage
if __name__ == "__main__":
    file_path = 'data.csv'  # Path to data CSV file path
    main(file_path)

