import capymoa
from capymoa.regressor import (
    FIMTDD,
    KNNRegressor as KNN,
    AdaptiveRandomForestRegressor as ARF,
    SOKNL
)
from capymoa.prediction_interval import MVE, AdaPI
from capymoa.stream import stream_from_file
from capymoa.evaluation import prequential_evaluation, prequential_evaluation_multiple_learners
from capymoa.evaluation.visualization import plot_windowed_results, plot_prediction_interval

dataset_path = '/Users/spencer/PycharmProjects/drift_simulation/new_generated_datasets/'

aba_a = stream_from_file(dataset_path + 'abalone_3abrupts.arff')
aba_g = stream_from_file(dataset_path + 'abalone_3graduals.arff')
aba_i = stream_from_file(dataset_path + 'abalone_incremental.arff')

bik_a = stream_from_file(dataset_path + 'bike_3abrupts.arff')
bik_g = stream_from_file(dataset_path + 'bike_3graduals.arff')
bik_i = stream_from_file(dataset_path + 'bike_incremental.arff')

house_a = stream_from_file(dataset_path + 'House8L_3abrupts.arff')
house_g = stream_from_file(dataset_path + 'House8L_3graduals.arff')
house_i = stream_from_file(dataset_path + 'House8L_incremental.arff')

sup_a = stream_from_file(dataset_path + 'superconductivity_3abrupts.arff')
sup_g = stream_from_file(dataset_path + 'superconductivity_3graduals.arff')
sup_i = stream_from_file(dataset_path + 'superconductivity_incremental.arff')

nzep_a = stream_from_file(dataset_path + 'nzep_3abrupts.arff')
nzep_g = stream_from_file(dataset_path + 'nzep_3graduals.arff')

akl = stream_from_file(dataset_path + 'alb_incremental.arff')
ham = stream_from_file(dataset_path + 'ham_incremental.arff')
wil = stream_from_file(dataset_path + 'wil_incremental.arff')
sdn = stream_from_file(dataset_path + 'sdn_incremental.arff')

learners = {
    'FIMTDD': FIMTDD(schema=aba_a.get_schema(), grace_period=200, split_confidence=0.01),
    'KNN': KNN(schema=aba_a.get_schema(), k=10),
    'ARF-Reg': ARF(schema=aba_a.get_schema(), ensemble_size=30),
    'SOKNL': SOKNL(schema=aba_a.get_schema(), ensemble_size=30)
}

results = prequential_evaluation_multiple_learners(aba_a, learners)

plot_windowed_results(
    results["FIMTDD"],
    results["KNN"],
    results["ARF-Reg"],
    results["SOKNL"],
    metric="rmse",
    figure_path="aba_a_rmse.png",
    figure_name="Abalone Abrupt RMSE",
    save_only=False,
)