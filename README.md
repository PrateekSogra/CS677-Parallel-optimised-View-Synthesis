# CS677-Parallel-optimised-View-Synthesis

##Setup for our code

1. Create a conda environment using our environment.yml file.

   'code' conda env create --name envname --file=environments.yml

   'code' conda activate envname

2. Create viewpoints using make_data.py

   'code' python3 make_data.py

4. Run the MPI code surf-view-syn for surface rendering for NUM_PROCESSES processes

    'code' mpiexec -n NUM_PROCESSES pvbatch surf-view-syn.py

   Run the code vol-view-syn for volume rendering for NUM_PROCESSES processes

    'code' mpiexec -n NUM_PROCESSES pvbatch vol-view-syn.py

5. The time taken for surface rendering optimization is stored in results1.csv and volume rendering optimization is stored in results2.csv.
   
6. Similarity results and optimization time vs for time-trails can be reproduced using similaritygraphs.ipynb and opttimegraphs.ipynb.
