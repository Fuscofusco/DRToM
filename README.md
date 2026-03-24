# INFO 

There are four sections here: 
- (1) DATA STRUCTURE for a quick debrief on how the data is stored 
- (2) MAIN THREE which are the main scripts used to run the generator 
- (3) CLUSTER THREE which talks about how one can upgrade this to use on a cluster (what I am currently using)
- (4) ANALYSIS DIRECTORY which talks about the analysis code that can be used for after

--- 

# (1) DATA STRUCTURE 

This will incorporate the /Data/ and /Summary/ structure. As of uploading there should be some LHE files that help with the explanation. 

The way the data is organized, from top down, is as follows: 
- CoM energy scale labeled as TeV##p#, e.g. TeV13p0
- Dimensionality of the event, either 2D or 3D 
- The collision process: 2to2, 2to3, 2to4, 2to5, or 2to2_QCD
- The LHE itself named after the range it was generated from in TeV, e.g. 9.0to9.1_0 has events generated from 9.0 TeV to 9.1 TeV 

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

Builds where the output data goes and the naming of each file based on all the previous settings (creation from the Cluster inputs is also here).


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

This is the generators main guard. Nothing in this file needs to be changed. The choices made in the configuration area are pulled from and used in the generation process. Of note are the two different generation methods of "*FullRange*" or "*Slices*". The internal generation process of each are the same (i.e. both QCD or phase space from the 'Generate Events' code), what changes is if the events are generated over the full range of start and end TeV entered (to which there is a bias selection at the low TeV range) or in slices (so that there is a proper spread of generated events across the whole TeV range).

---

# (3) CLUSTER THREE

params.sh, run.slurm, submit.sh 

These are cluster-use helpers and aren't essential to running DRToM, but as I moved everything to a cluster they are now what I use to run the generator.  
- params.sh is where you change the generation info and it auto-calculates the array size
- run.slurm is what runs main.py (at the bottom with the params.sh parameters)  
- ./submit.sh is the submit key 

---

# (4) ANALYSIS DIRECTORY 

I've included my old analysis code that was used for validation in the /Analysis/ directory, the main piece being `LHE_Analysis.ipynb`. These are jupyter notebooks so all have markdown cells to talk about what is going on, and there is a readme in that folder for extra info. 