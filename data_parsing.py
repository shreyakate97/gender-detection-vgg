import os
import shutil
import re

cwd = os.getcwd()
root_dir = os.path.join(cwd + "/aligned")

dst_dir = os.path.join(cwd + "/female/")
if os.path.exists(dst_dir) and os.path.isdir(dst_dir):
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)

#go in each file ending in F and copy all contents to female
regex = re.compile('(.*_F$)')

for root, dirs, files in os.walk(root_dir):
  for dir in dirs:
    if regex.match(dir):
       src_dir = os.path.join(root, dir)

       for f in os.listdir(src_dir):
            if f.endswith(".jpg"):
                i = os.path.join(src_dir, f)
                shutil.copy(i, dst_dir)
                
######################################################################################         
root_dir = os.path.join(cwd + "/valid")

dst_dir = os.path.join(cwd + "/female/")

#go in each file ending in F and copy all contents to female
regex = re.compile('(.*_F$)')

for root, dirs, files in os.walk(root_dir):
  for dir in dirs:
    if regex.match(dir):
       src_dir = os.path.join(root, dir)

       for f in os.listdir(src_dir):
            if f.endswith(".jpg"):
                i = os.path.join(src_dir, f)
                shutil.copy(i, dst_dir)
                
######################################################################################

root_dir = os.path.join(cwd + "/aligned")

dst_dir = os.path.join(cwd + "/male/")
if os.path.exists(dst_dir) and os.path.isdir(dst_dir):
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)

#go in each file ending in F and copy all contents to female
regex = re.compile('(.*_M$)')

for root, dirs, files in os.walk(root_dir):
  for dir in dirs:
    if regex.match(dir):
       src_dir = os.path.join(root, dir)

       for f in os.listdir(src_dir):
            if f.endswith(".jpg"):
                i = os.path.join(src_dir, f)
                shutil.copy(i, dst_dir)
                
######################################################################################         
root_dir = os.path.join(cwd + "/valid")

dst_dir = os.path.join(cwd + "/male/")

#go in each file ending in F and copy all contents to female
regex = re.compile('(.*_M$)')

for root, dirs, files in os.walk(root_dir):
  for dir in dirs:
    if regex.match(dir):
       src_dir = os.path.join(root, dir)

       for f in os.listdir(src_dir):
            if f.endswith(".jpg"):
                i = os.path.join(src_dir, f)
                shutil.copy(i, dst_dir)



