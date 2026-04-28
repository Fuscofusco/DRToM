# INFO 

There are four sections in this readme: 
- (1) DATA STRUCTURE for a quick debrief on how the data is stored 
- (2) MAIN THREE which are the main scripts used to run the generator 
- (3) CLUSTER THREE which are the main cluster scripts

--- 

# (1) DATA STRUCTURE 

There are several things we need to consider in the naming scheme. The data goes into \LHEF\ and within there are energy folders of e.g. TeV3p0_tag or TeV13p6_tag. The tag is for the user to decide what to do with (if there is a generation specific thing that should be noted). 

Within the energy folder are the different mc23_ folders. Let us consider the example 

- /hepusers2/fuscomus/DRToM/Generation/LHEF/TeV13p0_test/mc23_13p0TeV.100000.STRPy8EG_STR_2D_O2_L_020_U_030.evgen.TXT.e0000

We note the important quantities after mc23_: 

- CoM energy scale is labeled as TeV##p#, e.g. TeV13p0
- The DSID for the specific run (made in the DRToM/DSID folder)
- Dimensionality of the event, e.g. 2D or 3D 
- The number of outgoing partons in the LHE file, e.g O2 is 2 outgoing partons (so 2 to 2), O3 is 3 (so 2 to 3), and so on. 
- The start energy of the slice; L_020 means a lower bound of 2.0 TeV.
- The end energy of the slice; L_030 means an upper bound of 3.0 TeV 

Within these folders are the lhe files (labelled as .events) and tar.gz's which have their respective .events in them and are needed for the hadronization step. 

The Summary folder has the same structure as above, but instead of .lhe files, there are .txt files that contain the event generation summary. This is mainly used in the Analysis part for ID counting. 

--- 

# (2) MAIN THREE 

## configuration.py 

This is the main configuration area for each run. Here I list the main blocks with a summary for each. 

**Random Stuff** 

Right now, if the random generated mass of the event is larger than the DR scale, then the phases space generation is spherical. Setting `DR_flag` to 1 would turn on the probabilistic case, and the liklihood is controlled by `DR_prob` which is just a binomial probability parameter.  

**Event Settings** 

This is the main configuration block in which the generation setting can be changed, the options are self-explanitory. Note: if generating $2 \to 2$, one must choose to either generate in phase space (PS) or proper QCD kinematics (QCD). See `output_type`.

After this is the Cluster parameters, which operates using the same variables but are supplied through the `params.sh` file. 

**QCD Settings**

This is the process map in which events can be turned on or off. From left to right the columns signify: written name of the event, (process ID, active or not). The `process_map` then uses that info and extends it for additional use in folder finding. 

**Event Settings Build**

Here we take the inputs from `Event Settings` and give them proper local names which can be pulled across files and are used to build directory names. 

**Directories**

Most important thing here is the `dir_tag` for after the TeV##p# case. 


## functions.py

This file contains all the functions used across the generation. There are multiple sections of simple functions that have small descriptors at their top. For brevity, here I will list each section with a simple explanation

- **Generation Parameters**: For PDF choice and integration slices 
- **Baseline Functions**: Double integral function and lorentz boost
- **Outgoing Kinematics**: 2D and 3D phase space generation and $2 \to 2$ QCD dynamics 
- **Matrix Elements and Colour Flow**: The $M^2$ of each QCD subprocess is here along with the particle ID selection for incoming and outgoing partons
- **Colour Flow Helper**: These functions are the ones that properly go into the ColourFlow folder and files and select which ones to use based on a weighted selection 
- **QCD Functions**: This is where the QCD dynamics live 
- **Generate Events**: The main generation function for MC events. Steps are labeled in the function for easy following. 
- **LHE Write**: The function that takes the output of generation and properly organizes it in LHE format
- **Output**: Where we control what the output looks like after geneartion


## main.py

This is the generators main guard. Nothing in this file needs to be changed. The choices made in the configuration area are pulled from and used in the generation process. 

---

# (3) CLUSTER THREE

params.sh, run.slurm, submit.sh 

These are cluster-use helpers and aren't essential to running DRToM.

- params.sh is where you change the generation info and it auto-calculates the array size
- run.slurm is what runs main.py (at the bottom with the params.sh parameters). This is where you can change the output type from QCD to PS   
- ./submit.sh is the submit key 