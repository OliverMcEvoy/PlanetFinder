The models can be found at:

https://computingservices-my.sharepoint.com/:f:/g/personal/om518_bath_ac_uk/EvxLoph4OA1Dl2GH3u4E39kB0rOhQASqWnFvQq1EivEAjA?e=fR2ElF

Anyone with a uni of bath email should be able to access this and download it, any issues email me and let me know.

They should be saved as
'Models/CnnModel.py'
and 
'Models/RnnModel.py'

A Synthetic light curve can be generated with the following command for example 

python SyntheticLightCurveGeneration.py --num_systems 1 --max_planets_per_system 5 --num_iterations 400 --total_time 365 --cadence 0.0208333 --output_file light_curves.hdf5 --plot

The command to generate the light curve data used to train the final models was

python SyntheticLightCurveGeneration.py --num_systems 100 --max_planets_per_system 8 --num_iterations 400 --total_time 1600 --snr_threshold 5 --cadence 0.0208333 --output_file TrainingData/LightCurveTrainingData.hdf5 --plot