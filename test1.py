from fextractor import extractor
from os import walk

def extract(path):
    count=-1
    dirs=None
    #print(dirs)
    for subdir, dirss, files in walk(path):
        if count==-1:
            dirs=dirss
            count+=1
        else:
            print(dirs[count%7])
            
            #extractor(path,dirs[0], [1])
            extractor(path,dirs[count%7], [1])
            count+=1
            continue
            #break
    print(dirs)
#extract("pics/test")
#extract("pics/Stirling")
extract("pics/mmi")
#extract("pics/jaffe")
#extract("pics/ck")