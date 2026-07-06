# INFO 

Main sections in this readme: 
- (1) **DATA STRUCTURE** for a quick debrief on how the data is stored 
- (2) **MAIN THREE** which are the main scripts used to run the generator 
- (3) **CLUSTER THREE** which are the main cluster scripts
- (4) **EXAMPLE** I walk through an example on how to use the generator

--- 

# (1) DATA STRUCTURE 

There are several factors we need to consider in the naming scheme. The data is stored in the \raid\ area in \LHEF\. Within this are the energy folders with names like TeV3p0_tag or TeV13p6_tag. The _tag is for the user to decide (if there is a generation specific quantity that should be noted, e.g. how many events per energy slice, if it is QCD, etc). 

Within the energy folder are the different mc23_ folders. Let us consider the example 

- /DRToM/LHEF/TeV13p0_110/mc23_13p0TeV.100000.STRPy8EG_STR_2D_O2_L_020_U_030.evgen.TXT.e0000

We note the important quantities after mc23_: 

- CoM energy scale is labeled as TeV##p#. Here TeV13p0 means $\sqrt{s} = 13$ TeV
- The DSID for the specific run (made in the DRToM/DSID folder). Here it is 100000
- Dimensionality of the event, e.g. 2D or 3D. Here it is 2D
- The number of outgoing partons in the LHE file, e.g. O2 is 2 outgoing partons (so 2 to 2), O3 is 3 (so 2 to 3), and so on. Here we see it is O2 so we are looking at a 2 to 2 process.
- The start energy of the slice is labelled as L_###. Here, L_020 means a lower bound of 2.0 TeV.
- The end energy of the slice is laballed as U_030. Here, U_030 means an upper bound of 3.0 TeV. 

Within these folders are .tar.gz's that contain the lhe files (labelled as .events).

The Summary folder has the same structure as above, but instead of LHE files, there are .txt files that contain the event generation summary. This is mainly used in Analysis for ID counting. 

--- 

# (2) MAIN THREE 

## configuration.py 

This is the main configuration area for each run. Here I list the main blocks with a summary for each. 

**Random Stuff** 

Right now, if the random generated mass of the event is larger than the DR scale, then the phases space generation is spherical. Setting `DR_flag` to 1 would turn on the probabilistic case, and the liklihood is controlled by `DR_prob` which is just a binomial probability parameter.

**Event Settings** 

This is the main configuration block in which the generation setting can be changed, the options are self-explanitory. Note: if generating $2 \to 2$, one must choose to either generate in phase space (PS) or proper QCD kinematics (QCD). See `output_type`.

After this is the Cluster parameters, which operate using the same variables but are supplied through the `params.sh` file. 

**QCD Settings**

This is the process map in which events can be turned on or off. From left to right the columns signify: written name of the event, (process ID, active or not). The `process_map` then uses that info and extends it for additional use in folder finding. 

**Event Settings Build**

Here we take the inputs from `Event Settings` and give them proper local names which can be pulled across files and are used to build directory names. This is where you can change the yMax cut, where I chose 4 as an arbitrary value. 

**Directories**

Most important thing here is the `dir_tag` for after the TeV##p# case. This is tied to `params.sh` DIR_TAG. 


## functions.py

This file contains all the functions used across the MC generation. There are multiple sections, here I will list each one with a simple explanation

- **Generation Parameters**: For PDF choice and integration slices 
- **Baseline Functions**: Double integral function and lorentz boost
- **Outgoing Kinematics**: 2D and 3D phase space generation, and $2 \to 2$ QCD kinematics 
- **Matrix Elements and Colour Flow**: The $\abs{M}^2$ of each QCD subprocess is here along with the particle ID selection for incoming and outgoing partons
- **Colour Flow Helper**: These functions are the ones that properly go into the ColourFlow folder and files and select which ones to use based on a weighted selection 
- **QCD Functions**: This is where the QCD kinematics live 
- **Generate Events**: The main generation function for MC events. Steps are labeled in the function for easy following. 
- **LHE Write**: The function that takes the output of generation and properly organizes it in LHE format
- **Output**: Where we control what the output looks like after geneartion (not important for the generate, QoL output thing)


## main.py

This is the generators main guard. Nothing in this file needs to be changed. The choices made in the configuration area (and later cluster code) are pulled from and used in the generation process. 

---

# (3) CLUSTER THREE

These include: params.sh, run.slurm, and submit.sh. These are cluster helpers and aren't essential to running DRToM, unless of course using a cluster. Below I list the names and their general purpose.

- params.sh. This is where you change the generation info (that overides configuration.py) and gets fed into main.py. 
- run.slurm is what runs main.py (at the bottom with the params.sh parameters). 
- ./submit.sh is the submit key 


---

# (4) EXAMPLE

I consider the case of running this on a cluster. The first thing to do is make sure that the QCD processes that you want active are indeed active in process_map in configurations.py. For this example, I want all of them active so I set all of them to True. 

Next, we go to params.sh and check our physics parameters in the first section called 'Physics parameters.' I want:

- Phase space so I make sure OUTPUT_TYPE is PS, 
- A CoM energy of 13.0 and 13.6 so I set CM_ENERGIES=( 13.0 13.6 ) 
- Both 2D and 3D events so I set DIMENSIONALITY_LIST=( 2 3 )
- All outgoing parton types so I set NPARTONS_LIST=( 2 3 4 5 )
- The generation to start at 2 TeV so I set WINDOW_START=2.0
- The generation to end at 11 TeV so I set WINDOW_END=11.0
- The step sizes for the event generation to be 0.1 TeV so I set WINDOW_STEP=0.1 
- The number of events per mass slice to be 22000 so I set WINDOW_EVENTS=22000 
- Five files per mass slice of 22000 events so ITERATIONS=5, which gives me a total of 110,000 events per mass slice 
- A tag on the generation of 22000 to signal how many events per LHE file so I set DIR_TAG="22000"

With all my parameters set, I run ./submit.sh. 