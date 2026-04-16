## DimensionalReduction.py 

This is not the best name for this area, but oh well. This is all the analysis functions I made and used throughout the first year of my thesis. I would not recomend looking at all the functions here for two reason: (1) there are a lot and many are not used, (2) it is a mess with little organization. 

Within the scripts below I call to several functions in this area, so I would stick to looking at the relevant ones as needed. 


## Actual analysis

The point is to plot: Invariant mass, kinematics, event shape variables, momentum fraction plots, and planarity vs. mass plots. 

Since we are using larger datasets, this is broken up into many steps. 

Start in `make_DR_list.py` and make the file list for what DR scale you want. Note to add an appropriate tag so that you can track the data easier. 

-This outputs in ClusterData/FileLists under the appropriate energy/process/DR folder 

Go to `worker_jobs.py` and change all the input information at the top to make sure it is pointing to the proper file list. Check `worker_jobs.slurm` to make sure there is a proper amount of arrays and file per task.

- This should output in ClusterData/PartialOutputs under the appropriate energy/process/DR folder 

**Note:** I tried running multiple of these at once and it wasn't really working so that's why it is part of a loop now over the process and DR types.

Go to `merge.py` and change all the input information under "Safety" block to make sure it is pointing to the proper file list. Run `merge.slurm`

- Outputs should be in ClusterData/MergedOutputs under the appropriate energy/process/DR folder 

**Note:** These can be run for multiple files at the same time. There was no issues in that. 

With the merged files, we can run the analysis. This is done in `kinematics.py` and `PlanarVsMass.py`. Specifically, `kinematics` handles all the kinematic-like variables (x fractions, invariant mass, event shape variables, kinematics), and `PlanarVsMass` handles the plots like biplanarity as a function of mass and the overlays of DRs with that. 

**Note:** These can be run for multiple files at the same time. There was no issues in that. 

## PDF_Plotting.ipynb 

All this is doing is plotting some PDF related physics and should be easy enough to follow as is (no markdowns). 


## Biplanarity Study 

This is the notebook I used to conduct the biplanarity test. Comments are in the notebook. 