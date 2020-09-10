In datasets are two folders: segmentations and trajs. In segmentations are the SS in numpy files for each scene. In trajs are four txt files with the trajectories in the same format as ETH-UCY: "frame_id pedestrian_id x y", where (x, y) is the position of pedestrian 'pedestrian_id' in the frame 'frame_id'.

The functions to process this dataset are in dataset_processing.py.
