## DimensionalReduction.py 

This is not the best name for this area, but oh well. This is all the analysis functions I made and used throughout the first year of my thesis. I would not recomend looking at all the functions here for two reason: (1) there are a lot and many are not used, (2) it is a mess with little organization. 

Within the scripts below I call to several functions in this area, so I would stick to looking at the relevant ones as needed. 


## LHE_Analysis.ipynb 

This is the analysis tool for all things LHE related. There are markdowns above each cell for direction on what is going on. The outputs of this go into the `Plots` directory. 

TL;DR reads LHE files and plots: Invariant mass, kinematics, event shape variables, ID counts, momentum fraction plots. 

This notebook works fine for small datasets, but when running on something with e.g. 10million events it is not great. Because of that I broke it up into the `worker_jobs.py` and `merge_and_run.py` (each with their respective slurm jobs) so that it can be done on the cluster. 

### How to use

Start in `make_DR_list.ipynb` and make the file list for what DR scale you want. 

-This should output in ClusterData/FileLists under the appropriate energy/process/DR folder 

Go to `worker_jobs.py` and change all the input information at the top to make sure it is pointing to the proper file list. Check `worker_jobs.slurm` to make sure there is a proper amount of arrays and file per task.

- This should output in ClusterData/PartialOutputs under the appropriate energy/process/DR folder 

**Note:** I tried running multiple of these at once and it wasn't really working. It's annoying but just do one FileList file at a time...

Go to `merge_and_run.py` and change all the input information under "Safety" block to make sure it is pointing to the proper file list. Run `merge_run.slurm`

- Outputs should be in Plots under the appropriate energy/process/DR folder 



## PDF_Plotting.ipynb 

All this is doing is plotting some PDF related physics and should be easy enough to follow as is (no markdowns). 


## Biplanarity Study 

This is the notebook I used to conduct the biplanarity test. Comments are in the notebook. 