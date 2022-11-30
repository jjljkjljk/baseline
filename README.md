# Baselines for Retrosynthetic Planning With Experience-Guided Monte Carlo Tree Search


#### 1. Setup the environment
##### 1) Download the repository
    cd baseline
##### 2) Create a conda environment
    conda env create -f environment.yml
    conda activate eg_mcts_env

#### 2. Download the data
##### 1) Download the building block set, pretrained one-step retrosynthetic model, and (optional) reactions from USPTO.
Download the files from this [link](https://drive.google.com/drive/folders/1fu70cs0-POpMpsPzbqIfkpdaQa24Iqk2?usp=sharing).
Put all the files in ```dataset``` (```origin_dict.csv```,  ```uspto.reactions.json```, ```chembl.csv```, ```reaction_graph.gml```) under the ```baseline/dataset``` directory.
Put the folder ```one_step_model/``` under the ```baseline``` directory.
#### 3. Install lib
    pip install -e baseline/packages/mlp_retrosyn
    pip install -e baseline/packages/rdchiral


#### 4. Data instruction

Three test sets in dataset.


#### 5. Reproduce experiment results


    cd baseline

DFS_plan.py for greedy DFS.
DFPN_E_plan.py for dfpn-e.
MCTS_rollout_plan.py for MCTS-rollout.

You can change the plan class's test_file_name to change the test file.
The hyperparameter settings are detailed in the properties of the classes.





