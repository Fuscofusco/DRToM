import json

Start = 0.5       # Starting TeV energy of generation
End = 11.0        # Ending TeV energy of generation 
Step = 0.1        # Step size in TeV 
Dimensionality = [2, 3]

# Setting this so that each can have upto 500 unique DSIDs
base_map = {
    2: 100000,
    3: 100500
}

# No round point error
n_steps = int(round((End - Start) / Step))

dsid_map = {}

# Generate DSID map for each dimensionality and energy bin
for dim in Dimensionality:
    key = f"{dim}D"
    base = base_map[dim]

    dsid_map[key] = {}

    for i in range(n_steps):
        low = Start + i * Step
        high = low + Step

        tag = f"{low:.1f}to{high:.1f}"
        dsid = base + i

        dsid_map[key][tag] = dsid

with open("dsid_map.json", "w") as f:
    json.dump(dsid_map, f, indent=2)

print("✅ DSID map written to dsid_map.json")