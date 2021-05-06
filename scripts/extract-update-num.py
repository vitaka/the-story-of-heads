import sys
import numpy as np

data=np.load(sys.argv[1])
print(data.files['global_step:0'])
