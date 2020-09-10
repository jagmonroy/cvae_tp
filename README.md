# cvae_tp
Conditional Variational Autoencoder for Trajectory Prediction.

This repository contains the code that I used to obtain the presented results in my master thesis. The VAEs implementations are in the folder "models". Currently, there are only two datasets available and each one has its own folder: ETH-UCY and crossroad. In each dataset folder there are the next things:

* README.md. In this file, the datasets folder is explained. 
* datasets. The trajectories are here. Each dataset has its own format.
* config_files. This folder contains the configuration files. 
* dataset_processing.py. It contains the functions to process the data. They are different for each dataset.
* results.ipynb. This is a notebook to visualize the results. Actually, all thesis figures were generated with these notebooks.

requirements.txt contains my environment configuration (tensorflow, cv2 and numpy versions, for example).

No weights are provided to exactly replicate thesis results. The models have to be trained from scratch and then use results.ipynb. To train, the script training.py is used. It receives a configuration file as argument. For example, to get the best ETH-UCY SDVAE results, the command is:

python3 training.py ETH-UCY/config_files/batch1/sdvae_tf0_feat0.yaml

The results are saved in a folder results. If it does not exist, it is going to be created. Each dataset will have its folder in results: results_crossroad and results_ethucy. 

The results notebook for ETH-UCY is designed to have all the possibilities of ablation studies. For example, in SDVAE, these four commands have to be executed:

python3 training.py ETH-UCY/config_files/batch1/sdvae_tf0_feat0.yaml

python3 training.py ETH-UCY/config_files/batch1/sdvae_tf0_feat1.yaml

python3 training.py ETH-UCY/config_files/batch1/sdvae_tf1_feat0.yaml

python3 training.py ETH-UCY/config_files/batch1/sdvae_tf1_feat1.yaml

To facilitate this, there is the script run_folder.py. It receives two arguments: the folder with a bunch of config files and the gpu id wich will be used. In this way, if the computer has 2 gpu's, it is possible replicatating SDVAE results executing this two commands at the same time:

python3 run_folder.py ETH-UCY/config_files/batch1/ 0

python3 run_folder.py ETH-UCY/config_files/batch2/ 1

By default, all ETH-UCY config files has only one run. In the thesis, each configuration was run 5 times. To run the configuration more times, just add ids (strings) in id_list in the yaml file. By default, only sdvae_tf1_feat1_pe0 and sdvae_tf0_feat1_pe0 are run one time, the rest are run 5 times (they can be seen as example if it is desired to do more than one run in ETH-UCY).






