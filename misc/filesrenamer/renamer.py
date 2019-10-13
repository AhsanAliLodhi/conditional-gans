# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
from argparse import ArgumentParser
import random
  
# Function to rename multiple files 
def rename_files(_dir,_prefix="",_suffix="",rand_min=None,rand_max=None): 
    i = 0
    total_files =  len([name for name in os.listdir(_dir) if os.path.isfile(_dir+'\\'+name)])  
    for filename in os.listdir(_dir):
        prefix = _prefix
        suffix = _suffix
        #print("filename: ",filename)
        if not os.path.isfile(_dir+'\\'+filename):
            continue
        if rand_min is not None and rand_max is not None and rand_min>-1 and rand_min<rand_max:
            suffix+="_"+str(random.randint(rand_min, rand_max))
        if prefix is not None and len(prefix)>0:
            prefix+="_"
        filename_without_extention, file_extension = os.path.splitext(filename)
        #print("prefix: ",prefix)
        #print("suffix: ",suffix)
        #print(filename,prefix+filename_without_extention+suffix)
        suffix += file_extension
        src = _dir+'\\'+filename
        dst = _dir+'\\'+prefix+filename_without_extention+suffix
        os.rename(src, dst)
        i+=1
        print(str(i/total_files*100)+" %   "+ str(i) +" / "+str(total_files)+ " renamed")

if __name__ == "__main__" :
    parser = ArgumentParser()

    parser.add_argument("-d", "--directory", dest="dirname",
                        help="directory to change names"),
    parser.add_argument("-p", "--prefix", dest="prefix",
                        help="prefix that you want to append to all filenames", default = "")
    parser.add_argument("-s", "--suffix", dest="suffix",
                        help="suffix that you want to append to all filenames", default = "")                   
    parser.add_argument("-m", "--random_min", dest="rand_min",
                        help="lower limit for the random number appended with filename", default = None)
    parser.add_argument("-n", "--random_max", dest="rand_max",
                        help="upper limit for the random number appended with filename", default = None)

    
    args = parser.parse_args()

    if args.dirname is  None or len(args.dirname) < 1:
        print("please provide a valid directory")
        quit()
    if len(args.prefix)<1 and len(args.suffix)<1 and (args.rand_min is None or args.rand_max is None or args.rand_max>args.rand_min):
        print("please provide atleast one parameter to change the name of file")
        quit()
    rename_files(args.dirname,args.prefix,args.suffix,args.rand_min,args.rand_max) 
