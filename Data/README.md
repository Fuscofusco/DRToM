# GENERAL OVERVIEW

The data sructure follows first from the main process (2->2 up to 2->5). 
Once in the specific (2->n) process then one can choose what CoM energy they want to look at (either 13.0 or 13.6 TeV).
Then, within each cm folder is another folder that dictates the specific generation parameters that was used for that run. 
Further then is another subset that represents the dimensional reduction (DR) scale.
Lastly are the three folders that contain all the generation info. The 'CoM' folder stores all the slices of LHE files for events in the CoM. The 'lab' folder stores all the slices of LHE files for events in the lab frame, and 'summary' folder holds the generation summary of each generation slice.


## 2->4 EXAMPLE 

We want to look at our recent generation that ranged from 2 TeV to 11 TeV that had 500 events per 0.5 TeV with a CoM energy of 13.0 TeV. 
We start by going into "2to4" where we see the folder "cm13.0" and within that "tag_2.0to11.0_500_0.5" which is what we are looking for. 
- The '2.0to11.0' label signals the start and end generation values in TeV
- The first '_500' signals how many events are generated per slice 
- The last '_0.5' represents the slice steps used in the generation in TeV (2.0-2.5, 2.5-3.0, etc...)
- 'tag_' is here for use in the hardonization step to identify a "run tag"

Within the folder we see that there are many DR options. We want to see the case where the events are SM, so we look at the 'DR11' folder as this signals that DR happens at 11 TeV and if our generation ends at 11 TeV then these will all be SM events (i.e. DRX signals that the dimensional reduction turn-on scale is at X TeV). 

Lastly, in the DR11 folder we have the CoM, lab, and summary folders that contain all the LHE files from the generation. 

---

Of course this folder structure and naming convention is arbitrary and can be changed based on the preference of the user. I found what I have here works well for organizing and easily identifying the parameters of the generation + use in my hadronization code.