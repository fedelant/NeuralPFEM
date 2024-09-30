# NeuralPFEM
A Graph Neural Network (GNN) based data-driven simulator to build surrogate models for lagrangian fluid simulations.

Based on https://doi.org/10.48550/arXiv.2002.09405 (https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) 
and https://doi.org/10.1016/j.compgeo.2023.106015 (https://github.com/geoelements/gns/)

Dependencies and installation
-----------------------------
TO DO 

Usage
-----
All the following command must be launched in NeuralPFEM/ folder.
For a list of all the possible flags that can be used to change options in the code:
python3 -m npfem/NPFEM.py --help 

Launch training:
```
python3 -m gns.GNS --data_path="<input-training-data-path>" --model_path="<path-to-save-model-file>" --model_file --output_path="<path-to-save-output>" -ntraining_steps=NSTEPS
# For example:
python3 -m gns.GNS --data_path="gns/datasets/data1/" --model_path="gns/models/model1/" -ntraining_steps=1000000
```
The training can be interuppted whenever is necessary with cntrl+C in the terminal: the last model computed will be saved.

Continue training:
```
python3 -m gns.GNS --data_path="<input-data-path>" --model_path="<path-to-load-and-save-model-file>" --model_file=<"model-file-name.pt"> --train_state_file=<"train-state-file-name.pt"> -ntraining_steps=NSTEPS
# For example:
python3 -m gns.GNS --data_path="gns/datasets/data1/" --model_path="gns/models/model1/" --model_file="model-1000.pt" --train_state_file="train_state-1000.pt" -ntraining_steps=1000000
# To resume from the last model generated:
python3 -m gns.GNS --data_path="gns/datasets/data1/" --model_path="gns/models/model1/" --model_file="latest" --train_state_file="latest" -ntraining_steps=1000000
```
Model and train state file must be in model path.

Predict trajectories from valid/test dataset using learned simulator:
```
python3 -m gns.GNS --mode=<"valid" or "test"> --data_path="<input-data-path>" --model_path="<path-to-learned-models>" --model_file=<"model-file-name.py"> --output_path=<"path-to-save-outputs">
# For example:
python3 -m gns.GNS --mode="test" --data_path="gns/datasets/data1/" --model_path="gns/models/model1/" --model_file="model-1000.pt" --output_path="gns/outputs/output1/"
```

Generate vtk files for outputs:
```
python3 -m gns.render_output --output_mode=<"gif" or "vtk"> --output_path=<"output-data-path"> --output_file=<"file-with-output-data-to-visulize">
# For example:
python3 -m gns.render_output --output_mode="gif" --output_path="ns/outputs/output1/" --output_file="test_ex1"
```


