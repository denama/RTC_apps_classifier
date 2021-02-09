# RTC_apps_classifier
This is an ML classifier to distinguish the Online meeting application used based on TLS data. It tries many configurations of parameters and runs classification experiments for all of them.

## Quickstart
main_debug.py is the main, while main_debug_mp.py is the main using multiprocessing.

Before running, open config.py and specify the parameters you want in:
config_dict_grid --> 
output_name --> name of the json file with output
n_cores --> number of processes to use in case you run main_debug_mp.py

The output will appear in a folder Output_data/
