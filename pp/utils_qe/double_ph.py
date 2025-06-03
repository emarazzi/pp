import shutil
import numpy as np
from distutils.dir_util import copy_tree

#shutil.copyfile(f"b.dyn1.xml",f"../120q/dyn/b.dyn1.xml")
#shutil.copyfile(f"_ph0/b.phsave/patterns.1.xml",f"../120q/dyn/_ph0/b.phsave/patterns.1.xml")
#shutil.copyfile(f"_ph0/b.phsave/dynmat.1.0.xml",f"../120q/dyn/_ph0/b.phsave/dynmat.1.0.xml")
#shutil.copyfile(f"_ph0/b.phsave/dynmat.1.1.xml",f"../120q/dyn/_ph0/b.phsave/dynmat.1.1.xml")
#shutil.copyfile(f"_ph0/b.phsave/dynmat.1.2.xml",f"../120q/dyn/_ph0/b.phsave/dynmat.1.2.xml")


k = 1

for r in np.linspace(2,41,40):
    shutil.copyfile(f"b.dyn{int(r)}.xml",f"../160q/dyn/b.dyn{int(r)+k}.xml")
    copy_tree(f"_ph0/b.q_{int(r)}",f"../160q/dyn/_ph0/b.q_{int(r)+k}")
    shutil.copyfile(f"_ph0/b.phsave/patterns.{int(r)}.xml",f"../160q/dyn/_ph0/b.phsave/patterns.{int(r)+k}.xml")
    shutil.copyfile(f"_ph0/b.phsave/dynmat.{int(r)}.0.xml",f"../160q/dyn/_ph0/b.phsave/dynmat.{int(r)+k}.0.xml")
    shutil.copyfile(f"_ph0/b.phsave/dynmat.{int(r)}.1.xml",f"../160q/dyn/_ph0/b.phsave/dynmat.{int(r)+k}.1.xml")
    shutil.copyfile(f"_ph0/b.phsave/dynmat.{int(r)}.2.xml",f"../160q/dyn/_ph0/b.phsave/dynmat.{int(r)+k}.2.xml")
    k += 1




