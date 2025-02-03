# from drifting_functions import simulate_incremental_drift
# import numpy as np
# import pandas as pd
#
# data = np.random.uniform(1, 10, size=1000)
# data = pd.DataFrame(data, columns=["value"])
#
# data = simulate_incremental_drift(data, num_drifts=2, drift_lengths=200, drifting_feature="value", drop_drifting_feature=False)
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=data, x=data.index, y="value", linewidth=1, color="dodgerblue", label="Data with Incremental Drifts")
# plt.title("Example of Incremental Drifts", fontsize=18, fontweight='bold')
# plt.xlabel("Timestamp", fontsize=14)
# plt.ylabel("Value", fontsize=14)
# plt.legend(loc="upper left", fontsize=12)
# plt.tight_layout()
# plt.savefig("./incremental_drifts.png", dpi=300)
# plt.show()


from capymoa.regressor import KNNRegressor, AdaptiveRandomForestRegressor, SOKNL, FIMTDD
from capymoa.stream import stream_from_file
from capymoa.evaluation import prequential_evaluation, prequential_evaluation_multiple_learners
from capymoa.evaluation.visualization import plot_windowed_results



stream = stream_from_file("./generated_datasets/superconductivity_incremental.arff")

fimtdd = FIMTDD(schema=stream.get_schema(), grace_period=200)
# arf = AdaptiveRandomForestRegressor(schema=stream.get_schema())

results_fim = prequential_evaluation(stream=stream, learner=fimtdd, window_size=5000)
# results_arf = prequential_evaluation(stream=stream, learner=arf, window_size=100)
print('FIMTDD R2:', results_fim.rmse())
# print('ARF R2:', results_arf.r2())

plot_windowed_results(results_fim, metric="rmse", save_only=False)


