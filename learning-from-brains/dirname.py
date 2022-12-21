import os
#dirname=basePath = os.path.dirname(__file__)
import pathlib
# current working directory
basePath=pathlib.Path().absolute().as_posix()
print(basePath)