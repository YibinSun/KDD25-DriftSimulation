import matplotlib.pyplot as plt
from capymoa.regressor import KNNRegressor, AdaptiveRandomForestRegressor, SOKNL, FIMTDD
from capymoa.stream import stream_from_file
from capymoa.evaluation import prequential_evaluation, prequential_evaluation_multiple_learners
from capymoa.evaluation.visualization import plot_windowed_results
import os


save_path = "./second_plottings/"
file_list = []
for root, _, files in os.walk(
    "./new_generated_datasets/"
):
    for file in files:
        if file.endswith(".arff"):
            # Define the learners + an alias (dictionary key)
            file_list.append(os.path.join(root, file))


file_list = sorted(file_list)

for file in file_list:
    stream = stream_from_file(file)
    learners = {
        "KNN": KNNRegressor(schema=stream.get_schema()),
        "ARFReg": AdaptiveRandomForestRegressor(schema=stream.get_schema(), ensemble_size=10),
        "SOKNL": SOKNL(schema=stream.get_schema(), ensemble_size=10),
        "FIMTDD": FIMTDD(schema=stream.get_schema()),
    }

    results = {}
    for learner_name, learner in learners.items():
        print(f'Evaluating {learner_name}...')
        results[learner_name] = prequential_evaluation(stream=stream, learner=learner, window_size=1000)
        print(f'Evaluation finished for {learner_name}.')

    name = file.split('/')[-1].split('.')[0]

    plot_windowed_results(
        results["FIMTDD"],
                          results["KNN"],
                          results["ARFReg"],
                          results["SOKNL"],
                          metric="rmse", plot_title=f'RMSE on {name}', figure_path=save_path,
                          figure_name=f"rmse_on_{name}.png", ymin=0)

    plot_windowed_results(
        results["FIMTDD"],
                          results["KNN"],
                          results["ARFReg"],
                          results["SOKNL"],
                          metric="r2", plot_title=f'R2 on {name}', figure_path=save_path,
                          figure_name=f"r2_on_{name}.png", ymin=-1)

    print(f"Plotted {name} results.")
