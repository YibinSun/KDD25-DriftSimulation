from sdv.single_table import CTGANSynthesizer
import pandas as pd
import numpy as np
import arff
import drifting_functions
import os
from utilities import read_arff_to_dataframe, generate_conditional_synthetic_data, simulate_shifted_distribution, train_ctgan_models
from correlation import check_feature_correlation

def save_dataset(df, filename, format):
    """Save the dataset as CSV or ARFF."""
    if not os.path.exists('./new_generated_datasets/'):
        os.makedirs('./new_generated_datasets/')

    if format.lower() == "csv":
        df.to_csv(f"./new_generated_datasets/{filename}.csv", index=False)
        print(f"Dataset saved as {filename}.csv")
    elif format.lower() == "arff":
        try:
            # from scipy.io import arff
            arff_data = {
                "description": "Simulated dataset with concept drift",
                "relation": "drift_simulation",
                "data": df.to_records(index=False),
                "attributes": [(col, "NUMERIC") if df[col].dtype in [float, int] else (col, "STRING") for col in
                               df.columns],
            }
            with open(f"./new_generated_datasets/{filename}.arff", "w") as f:
                arff.dump(arff_data, f)
            print(f"Dataset saved as {filename}.arff")
        except ImportError:
            print("Error: Please install scipy to save in ARFF format.")
    else:
        print("Invalid format. Please choose either 'csv' or 'arff'.")



# NZEP Abrupt
# alb = CTGANSynthesizer.load('./trained_models/ctgan_alb_avg_4hr.pkl').sample(50000)
# wil = CTGANSynthesizer.load('./trained_models/ctgan_wil_avg_24hr.pkl').sample(50000)
# ham = CTGANSynthesizer.load('./trained_models/ctgan_ham_avg_6hr.pkl').sample(50000)
# sdn = CTGANSynthesizer.load('./trained_models/ctgan_sdn_avg_30min.pkl').sample(50000)
#
# nzep_abrupt_1 = drifting_functions.simulate_abrupt_drift(num_drifts=1, concept1=alb, concept2=wil, shuffle=False)
# nzep_abrupt_2 = drifting_functions.simulate_abrupt_drift(num_drifts=1, concept1=ham, concept2=sdn, shuffle=False)
# nzep_abrupt = drifting_functions.simulate_abrupt_drift(num_drifts=1, concept1=nzep_abrupt_1, concept2=nzep_abrupt_2, shuffle=False)
#
# save_dataset(nzep_abrupt, 'nzep_3abrupts', 'csv')
# save_dataset(nzep_abrupt, 'nzep_3abrupts', 'arff')
#
#
# # NZEP Gradual
#
# nzep_gradual_1 = drifting_functions.simulate_gradual_drift(num_drifts=1, concept1=alb, concept2=wil, shuffle=False, drift_lengths=10000)
# nzep_gradual_2 = drifting_functions.simulate_gradual_drift(num_drifts=1, concept1=ham, concept2=sdn, shuffle=False, drift_lengths=10000)
# nzep_gradual = drifting_functions.simulate_gradual_drift(num_drifts=1, concept1=nzep_gradual_1, concept2=nzep_gradual_2, shuffle=False, drift_lengths=10000)
#
# nzep_gradual.to_csv('./new_generated_datasets/nzep_3graduals.csv', index=False)
#
# save_dataset(nzep_gradual, 'nzep_3graduals', 'csv')
# save_dataset(nzep_gradual, 'nzep_3graduals', 'arff')
#
# # NZEP Incremental
# alb_incremental = drifting_functions.simulate_incremental_drift(
#     CTGANSynthesizer.load('./trained_models/ctgan_alb_avg_4hr.pkl').sample(100000), 2, 20000, method='pearson'
# )
# save_dataset(alb_incremental, 'alb_incremental', 'csv')
# save_dataset(alb_incremental, 'alb_incremental', 'arff')
#
# wil_incremental = drifting_functions.simulate_incremental_drift(
#     CTGANSynthesizer.load('./trained_models/ctgan_wil_avg_24hr.pkl').sample(100000), 2, 20000, method='pearson'
# )
# save_dataset(wil_incremental, 'wil_incremental', 'csv')
# save_dataset(wil_incremental, 'wil_incremental', 'arff')
#
# ham_incremental = drifting_functions.simulate_incremental_drift(
#     CTGANSynthesizer.load('./trained_models/ctgan_ham_avg_6hr.pkl').sample(100000), 2, 20000, method='pearson'
# )
# save_dataset(ham_incremental, 'ham_incremental', 'csv')
# save_dataset(ham_incremental, 'ham_incremental', 'arff')
#
# sdn_incremental = drifting_functions.simulate_incremental_drift(
#     CTGANSynthesizer.load('./trained_models/ctgan_sdn_avg_30min.pkl').sample(100000), 2, 20000, method='pearson'
# )
# save_dataset(sdn_incremental, 'sdn_incremental', 'csv')
# save_dataset(sdn_incremental, 'sdn_incremental', 'arff')

##############################################################################################################
# Abalone
drifting_feature = check_feature_correlation('./original_data/abalone.arff', method='pearson')
abalone_origin = read_arff_to_dataframe('./original_data/abalone.arff')
abalone_model = CTGANSynthesizer.load('./trained_models/ctgan_abalone.pkl')

models = train_ctgan_models(abalone_origin, drifting_feature, 4)

abalone_concept_1 = models[0].sample(50000)
abalone_concept_2 = models[1].sample(50000)
abalone_concept_3 = models[2].sample(50000)
abalone_concept_4 = models[3].sample(50000)

# abalone_concept_1 = generate_conditional_synthetic_data(
#     simulate_shifted_distribution(drifting_data, (feature_max - feature_mean) / 4),
#     abalone_model,
#     drifting_feature
# )
#
# abalone_concept_2 = generate_conditional_synthetic_data(
#     simulate_shifted_distribution(drifting_data, 3 * (feature_max - feature_mean) / 4),
#     abalone_model,
#     drifting_feature
# )
#
# abalone_concept_3 = generate_conditional_synthetic_data(
#     simulate_shifted_distribution(drifting_data, (feature_min - feature_mean) / 4),
#     abalone_model,
#     drifting_feature
# )
#
# abalone_concept_4 = generate_conditional_synthetic_data(
#     simulate_shifted_distribution(drifting_data, 3 * (feature_min - feature_mean) / 4),
#     abalone_model,
#     drifting_feature
# )

abalone_abrupt_1 = drifting_functions.simulate_abrupt_drift(1, abalone_concept_1, abalone_concept_3, shuffle=False)
abalone_abrupt_2 = drifting_functions.simulate_abrupt_drift(1, abalone_concept_4, abalone_concept_2, shuffle=False)
abalone_abrupt = drifting_functions.simulate_abrupt_drift(1, abalone_abrupt_1, abalone_abrupt_2, shuffle=False)

save_dataset(abalone_abrupt, 'abalone_3abrupts', 'csv')
save_dataset(abalone_abrupt, 'abalone_3abrupts', 'arff')

abalone_gradual_1 = drifting_functions.simulate_gradual_drift(1, abalone_concept_1, abalone_concept_3, shuffle=False, drift_lengths=10000)
abalone_gradual_2 = drifting_functions.simulate_gradual_drift(1, abalone_concept_4, abalone_concept_2, shuffle=False, drift_lengths=10000)
abalone_gradual = drifting_functions.simulate_gradual_drift(1, abalone_gradual_1, abalone_gradual_2, shuffle=False, drift_lengths=10000)

save_dataset(abalone_gradual, 'abalone_3graduals', 'csv')
save_dataset(abalone_gradual, 'abalone_3graduals', 'arff')


abalone_incremental = drifting_functions.simulate_incremental_drift(
    abalone_model.sample(100000), 2, 20000, method='pearson'
)
save_dataset(abalone_incremental, 'abalone_incremental', 'csv')
save_dataset(abalone_incremental, 'abalone_incremental', 'arff')



# Bike
drifting_feature = check_feature_correlation('./original_data/bike.arff', method='pearson')
bike_origin = read_arff_to_dataframe('./original_data/bike.arff')
bike_model = CTGANSynthesizer.load('./trained_models/ctgan_bike.pkl')
models = train_ctgan_models(bike_origin, drifting_feature, 4)

bike_concept_1 = models[0].sample(50000)
bike_concept_2 = models[1].sample(50000)
bike_concept_3 = models[2].sample(50000)
bike_concept_4 = models[3].sample(50000)

bike_abrupt_1 = drifting_functions.simulate_abrupt_drift(1, bike_concept_1, bike_concept_3, shuffle=False)
bike_abrupt_2 = drifting_functions.simulate_abrupt_drift(1, bike_concept_4, bike_concept_2, shuffle=False)
bike_abrupt = drifting_functions.simulate_abrupt_drift(1, bike_abrupt_1, bike_abrupt_2, shuffle=False)

save_dataset(bike_abrupt, 'bike_3abrupts', 'csv')
save_dataset(bike_abrupt, 'bike_3abrupts', 'arff')

bike_gradual_1 = drifting_functions.simulate_gradual_drift(1, bike_concept_1, bike_concept_3, shuffle=False, drift_lengths=10000)
bike_gradual_2 = drifting_functions.simulate_gradual_drift(1, bike_concept_4, bike_concept_2, shuffle=False, drift_lengths=10000)
bike_gradual = drifting_functions.simulate_gradual_drift(1, bike_gradual_1, bike_gradual_2, shuffle=False, drift_lengths=10000)

save_dataset(bike_gradual, 'bike_3graduals', 'csv')
save_dataset(bike_gradual, 'bike_3graduals', 'arff')


bike_incremental = drifting_functions.simulate_incremental_drift(
    bike_model.sample(100000), 2, 20000, method='pearson'
)
save_dataset(bike_incremental, 'bike_incremental', 'csv')
save_dataset(bike_incremental, 'bike_incremental', 'arff')


# House8L
drifting_feature = check_feature_correlation('./original_data/House8L.arff', method='pearson')
House8L_origin = read_arff_to_dataframe('./original_data/House8L.arff')
House8L_model = CTGANSynthesizer.load('./trained_models/ctgan_House8L.pkl')
models = train_ctgan_models(House8L_origin, drifting_feature, 4)

House8L_concept_1 = models[0].sample(50000)
House8L_concept_2 = models[1].sample(50000)
House8L_concept_3 = models[2].sample(50000)
House8L_concept_4 = models[3].sample(50000)

House8L_abrupt_1 = drifting_functions.simulate_abrupt_drift(1, House8L_concept_1, House8L_concept_3, shuffle=False)
House8L_abrupt_2 = drifting_functions.simulate_abrupt_drift(1, House8L_concept_4, House8L_concept_2, shuffle=False)
House8L_abrupt = drifting_functions.simulate_abrupt_drift(1, House8L_abrupt_1, House8L_abrupt_2, shuffle=False)

save_dataset(House8L_abrupt, 'House8L_3abrupts', 'csv')
save_dataset(House8L_abrupt, 'House8L_3abrupts', 'arff')

House8L_gradual_1 = drifting_functions.simulate_gradual_drift(1, House8L_concept_1, House8L_concept_3, shuffle=False, drift_lengths=10000)
House8L_gradual_2 = drifting_functions.simulate_gradual_drift(1, House8L_concept_4, House8L_concept_2, shuffle=False, drift_lengths=10000)
House8L_gradual = drifting_functions.simulate_gradual_drift(1, House8L_gradual_1, House8L_gradual_2, shuffle=False, drift_lengths=10000)

save_dataset(House8L_gradual, 'House8L_3graduals', 'csv')
save_dataset(House8L_gradual, 'House8L_3graduals', 'arff')


House8L_incremental = drifting_functions.simulate_incremental_drift(
    House8L_model.sample(100000), 2, 20000, method='pearson'
)
save_dataset(House8L_incremental, 'House8L_incremental', 'csv')
save_dataset(House8L_incremental, 'House8L_incremental', 'arff')



# Superconductivity
drifting_feature = check_feature_correlation('./original_data/superconductivity.arff', method='pearson')
superconductivity_origin = read_arff_to_dataframe('./original_data/superconductivity.arff')
superconductivity_model = CTGANSynthesizer.load('./trained_models/ctgan_superconductivity.pkl')
models = train_ctgan_models(superconductivity_origin, drifting_feature, 4)

superconductivity_concept_1 = models[0].sample(50000)
superconductivity_concept_2 = models[1].sample(50000)
superconductivity_concept_3 = models[2].sample(50000)
superconductivity_concept_4 = models[3].sample(50000)

superconductivity_abrupt_1 = drifting_functions.simulate_abrupt_drift(1, superconductivity_concept_1, superconductivity_concept_3, shuffle=False)
superconductivity_abrupt_2 = drifting_functions.simulate_abrupt_drift(1, superconductivity_concept_4, superconductivity_concept_2, shuffle=False)
superconductivity_abrupt = drifting_functions.simulate_abrupt_drift(1, superconductivity_abrupt_1, superconductivity_abrupt_2, shuffle=False)

save_dataset(superconductivity_abrupt, 'superconductivity_3abrupts', 'csv')
save_dataset(superconductivity_abrupt, 'superconductivity_3abrupts', 'arff')

superconductivity_gradual_1 = drifting_functions.simulate_gradual_drift(1, superconductivity_concept_1, superconductivity_concept_3, shuffle=False, drift_lengths=10000)
superconductivity_gradual_2 = drifting_functions.simulate_gradual_drift(1, superconductivity_concept_4, superconductivity_concept_2, shuffle=False, drift_lengths=10000)
superconductivity_gradual = drifting_functions.simulate_gradual_drift(1, superconductivity_gradual_1, superconductivity_gradual_2, shuffle=False, drift_lengths=10000)

save_dataset(superconductivity_gradual, 'superconductivity_3graduals', 'csv')
save_dataset(superconductivity_gradual, 'superconductivity_3graduals', 'arff')


superconductivity_incremental = drifting_functions.simulate_incremental_drift(
    superconductivity_model.sample(100000), 2, 20000, method='pearson'
)
save_dataset(superconductivity_incremental, 'superconductivity_incremental', 'csv')
save_dataset(superconductivity_incremental, 'superconductivity_incremental', 'arff')