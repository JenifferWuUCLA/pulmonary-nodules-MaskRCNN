import os
path = 'jpg'
filenames=os.listdir(path)
strText = ""

with open("train_list.csv", "w") as fid:
    for a in range(len(filenames)):
        strText = path+os.sep+filenames[a]  + "," + filenames[a].split('_')[0]  + "\n"
        fid.write(strText)
fid.close()
