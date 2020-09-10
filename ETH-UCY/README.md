ETH-UCY dataset can be downloaded (it is possible to use wget) from:

'https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=0'

This is the oficial data split from Social-GAN:

https://github.com/agrimgupta92/sgan

There are six folders: 1 for each scene and a raw folder with all trajectories. In each scene folder (eth, hotel, univ, zara1 and zara2) there are a train, validation and test folder, each one contains the corresponding trajectories for each phase. The folder name is the teste scene. The trajectories are in txt files with formart "frame_id pedestrian_id x y", where (x, y) is the position of pedestrian 'pedestrian_id' in the frame 'frame_id'.

The script dataset_processing.py contains the functions to process the data and get the trajectories in the desired format.
