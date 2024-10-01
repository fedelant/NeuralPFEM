# NeuralPFEM
A Graph Neural Network (GNN) based data-driven simulator to build surrogate models for lagrangian fluid simulations.

Based on https://doi.org/10.48550/arXiv.2002.09405 (https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) 
and https://doi.org/10.1016/j.compgeo.2023.106015 (https://github.com/geoelements/gns/)

Dependencies and installation
-----------------------------
TO DO
Compile fortran code of the mesher using f2py to bind it with python code:
```
gfortran -free -c 01_modules.f90
f2py -c mesh_gen.f90 01_modules.o -m mesh_gen
```

Usage
-----
All the following command must be launched in NeuralPFEM/ folder.
For a list of all the possible flags that can be used to change options in the code:
python3 -m npfem/NPFEM.py --help 

Launch training:
```
python3 -m npfem.NPFEM --data_path="<input-training-data-path>" --model_path="<path-to-save-model-file>" --flags -n_training_steps=NSTEPS
# For example:
python3 -m npfem.NPFEM --data_path="npfem/datasets/data1/" --model_path="npfem/models/model1/" --continue_training=False --input_velocity_steps=5 --batch_size=2 --noise_std_weight=5 --velocity_scale_weight=0.01 --n_message_passing_steps=10 -n_training_steps=1000000
```
The training can be interuppted whenever is necessary with cntrl+C in the terminal: the last model computed will be saved.

To continue training, set True the dedicated flag:
```
python3 -m npfem.NPFEM --data_path="<input-training-data-path>" --model_path="<path-to-save-model-file>" --continue_training=True -n_training_steps=NSTEPS

```

Predict trajectories from valid/test dataset using learned simulator:
```
python3 -m npfem.NPFEM --mode="<valid-or-test>" --data_path="<input-data-path>" --model_path="<path-to-learned-models>" --output_path="<path-to-save-output-files>" --output_path="<path-to-save-outputs>" --flags
# For example:
python3 -m npfem.NPFEM --mode="valid" --data_path="gns/datasets/data1/" --model_path="gns/models/model1/" --model_file="model-1000.pt" --output_path="gns/outputs/output1/" --input_velocity_steps=5 --batch_size=2 --velocity_scale_weight=0.01 --n_message_passing_steps=10
```

Generate vtk files for outputs:
```
python3 -m npfem.write_vtk --output_path=<"output-data-path"> --output_file=<"file-with-output-data-to-visulize">
# For example:
python3 -m gns.npfem_vtk --output_path="npfem/outputs/output1/" --output_file="test_ex1"
```


