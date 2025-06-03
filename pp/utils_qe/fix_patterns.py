import re
import numpy as np


match = "<QPOINT_NUMBER>" 

for k in [int(y) for y in np.linspace(1,81,81)[::2]][1:]:
    with open(f"patterns.{k}.xml","r") as file:
        lines  = file.readlines()
    with open(f"patterns.{k}.xml","w") as file:
        for line in lines:
            if match in line.strip("\n"):
                file.write(f"   <QPOINT_NUMBER>{k}</QPOINT_NUMBER>\n")
            else:
                file.write(line)
