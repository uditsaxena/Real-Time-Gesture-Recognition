import csv
import sys
import os

def breakdownfile(fname):
    f = open(fname, 'r')
    try:
        reader = csv.reader(f,quoting=csv.QUOTE_NONE)
        top = ['X_R','Y_R','Z_R','ROLL_R','PITCH_R','YAW_R','THUMB_R','FORE_R','MID_R','RING_R','LIT_R','X_L','Y_L','Z_L','ROLL_L','PITCH_L','YAW_L','THUMB_L','FORE_L','MID_L','RING_L','LIT_L','LABEL']
        newfilen = os.path.splitext(fname)[0]
        w = open(newfilen+'_new'+'.csv','a')
        writ = csv.writer(w)
        writ.writerow(top)
        for row in reader:
            # w = open(str(fo)+'.csv','a')
            # writ = csv.writer(w)
            # print row
            writ.writerow(row)
            # fo+=1
    finally:
        f.close()
        w.close()

def lookforcsvfile():
    root = '.'
    for dirName, subdirlist, fileList in os.walk(root):
        if dirName is '.':
            for fname in fileList:
                if fname.endswith(".csv"):
                    # filename = os.path.join(dirName, fname)
                    # f = open(filename,'r')
                    breakdownfile(fname)
                    # print subdirlist

if __name__ == '__main__':
    lookforcsvfile()