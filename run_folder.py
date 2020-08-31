# arguments:
# 1: folder with config files
# 2: gpu id

import sys
import os

for config_file in os.listdir(sys.argv[1]):

	if not 'yaml' in config_file: continue

	cf_path = os.path.join(sys.argv[1], config_file)
	cmd1 = 'source ~/activate_env'
	cmd2 = 'cd cvae_tp'
	cmd3 = 'CUDA_VISIBLE_DEVICES={0} python3 training.py {1}'
	cmd3 = cmd3.format(sys.argv[2], cf_path)
	cmd = 'ssh g-0-2 ' + '"' + " && ".join([cmd1, cmd2, cmd3]) + '"'
	os.system(cmd)
