# User Guide of Synthetic Data Generation

This project is a synthetic data simulation tool provided for data stream regression research.

## 1. Using CTGANs as Infinite Data Source
This work focuses on CTGANs as the data source, and simulate concept drifts in the data stream. The CTGANs are trained on the original dataset, and the generated data is used as the infinite data source. 
We used this approach because that the regression research in streaming scenario has suffered from the lack of proper data source.

## 2. Abstracting Concepts
Regression is different from classification in that the target value is continuous. 
Therefore, we select the most correlated feature with the target value as the drifting feature, and train CTGANs with different chucks of data as an individual concept.

The `training_CTGANs.py` provides the code to train CTGANs on the original datasets.
## 3. Simulating Drifts
We provide three types of drift simulation methods:
- **Sudden Drift**: The mean of drifting features abruptly switches.
- **Gradual Drift**: A random sampling from two concepts during a defined period.
- **Incremental Drift**: The mean of drifting features gradually increases or decreases until it reaches to the other concept.

Please see the `drifting_functions.py` for more details.

## 4. Examples
We provide generated datasets with different types of drifts and different source of real-world datasets in `new_generated_datasets`.
Furthermore, if you are keen on generating your own dataset, please see the `generateing.py` for more details.

## Cite Us
If you use the code or the datasets in your research, please cite us as follows:
```
@misc{sun2025evaluationregressionanalysesevolving,
      title={Evaluation for Regression Analyses on Evolving Data Streams}, 
      author={Yibin Sun and Heitor Murilo Gomes and Bernhard Pfahringer and Albert Bifet},
      year={2025},
      eprint={2502.07213},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.07213}, 
}
```