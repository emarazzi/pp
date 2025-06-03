import numpy as np
import itertools
import sys
from typing import List
def create_list(num_elements):
    step = 1 / num_elements
    result = [step * i for i in range(num_elements)]
    return result

if __name__ == "__main__":
    result_list = []
    if len(sys.argv) != 4:
        print("Usage: python script.py <kx> <ky> <kz>")
    else:
        try:
            for i in range(1,4):
                num_elements = int(sys.argv[i])
                if num_elements <= 0:
                    print("Number of elements must be a positive integer.")
                else:
                    result_list.append(np.linspace(0,1,num_elements,endpoint=False).tolist())
            res = list(itertools.product([r if isinstance(r,List) else [L] for r in result_list]))
            print(res)
            unique_res_map = set(map(tuple, res))
            unique_res_list = list(map(list, unique_sublists))
            print(f"{len(unique_res_list)} crystal")
            for u in unique_res_list:
                print(f"  {u[0]} {u[1]} {u[2]} {1/len(unique_res_list)}")
        except ValueError:
            print("Invalid input. Number of elements must be an integer.")
