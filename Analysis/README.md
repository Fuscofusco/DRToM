The main area is all the new analysis code I have been using. All the intermediate steps (all those lists and storing of event shape variables as .pkl's) are stored in the /raid/ area under /DRToM/Analysis/. The final plots are in this main section under /Plots/. The /Misc/ folder is discussed below. 

If starting an analysis process, and using ~10mil events, I found that it takes about 2hr to get through everything. This may sound like a lot, but my first couple of scripts were taking at least a day and they could only handle a total of ~1 mil events. The two parts of this long time frame is dependent on the `worker_jobs.py` step and how I handle the array size and files per task as well as the plotting of the kinematic variables. How to run this code (along with info on what each step does) is discussed below. 

If using this please use TeV13p0_110 to test how it all works. 


## DimensionalReduction.py 

We start with this first. This is all the analysis functions I made and used throughout the first year of my thesis. I would not recomend looking at all the functions here for two reason: 

- (1) there are a lot and many are not used or redundant or just outright bad, 
- (2) it is a mess with little organization. 

I made this when I was new to the splitting up functions from running code so it got out of hand and then I never cleaned it up or split it apart. 

Within the scripts below I call to several functions in this area, so I would stick to looking at the relevant ones as needed if you are going to use this. Ideally I would clean this all up, but alas I am running out of time.

--- 

# Actual analysis

The point of this all is to plot a handful of things, e.g. invariant mass, kinematics, event shape variables, momentum fraction plots, and planarity vs. mass plots. Since we are using larger datasets (~10mil events), this is broken up into several steps. 

Start in `MakeData.py` and make the file list for what DR scale you want. This outputs in /raid/.../FileLists/ under the appropriate energy/process/DR folder. All this is is a list of files that the next step will use to calculated all the relevant info.  

Go to `worker_jobs.py` and change all the input information at the top to make sure it is pointing to the proper file list. Check `worker_jobs.slurm` to make sure there is a proper amount of arrays and file per task. This should output in /raid/.../PartialOutputs/ under the appropriate energy/process/DR folder. This step takes the file list from the first step and does all the kinematic calculations and storeage which is loacated in `worker_jobs.py`. **Note:** I tried running multiple of these at once and it wasn't really working so that's why it is part of a loop now over the process and DR types.

At this point you have 5 different pickle subdirectories. We now want to merge the meta data (for the cross sections). Go to `merge_meta.py` and change all the input information under the "Configuration" block to make sure it is pointing to the proper area. This can also be done in `merge.slurm`. After running, the output is in /raid/.../MergedOutputs/.../meta. **Note:** These can be run for multiple files at the same time. There was no issues in that. 

With the merged meta files, we can now get our plots. This is done with six different scripts, one for each pickle subdirectory, and one to run it. The name are self explanatory: 

- `event_shape_from_partials.py` This gets all the event shape variables, outputs in /Plots/EventShapeVariables 
- `invar_mass_from_partials.py` This gets the invariant mass plot, outputs in /Plots/InvariantMass. 
- `kinematics_from_partials.py` This gets all the kinematic plots, outputs in /Plots/MomentumAndAngles
- `planarity_low_memory.py` This gets the A/B vs. mass plots and the AB correlation plots, outputing into /Plots/ and /Plots/AB_cor/, respectively
- `xa_xb_from_partials.py` This gets the xa and xb distribution which outputs into /Plots/
-  `plots_from_partials_array.slurm` is the running script and where all the inputs for the above are changed depending on what physics parameters you are using

The final pieace is the `PlanarVsMass.py` which handles the large overlay plots of many DR scales of the A/B vs. mass. 

---

# Misc 

This folder isn't required for anything, but I wanted to keep some code that I thought could be useful. Note that I haven't ran most of these in a long time so there are likely bugs, but they should be straight forward fixes. 

- `functions_py_test.ipynb` There were several functions in the Generator that I wanted to check and this is where I did them 
- `LambdifyAndpT.ipynb` Has plots and equations related to QCD matrix elements 
- `m2_values_test.py` I was testing to see what tolerance bounds I need for massless particles of different $2 \to n$ types 
- `PDF_Plotting.ipynb` Plotting PDF distributions 