# 참고 https://lynnshin.tistory.com/46
# 비율로 만 나누는 코드 ! stratify 속성 값은 따로 코드 찾아보고 업데이트 필요

import os, sys, pickle
import argparse, sys
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('-datapath', help=' : Please set the Dataset Path') 
parser.add_argument('-outputpath', help=' : spilt output path', default='TrainDataSet')
args = parser.parse_args()

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        image = file.split('/')[-1] + ".jpg"
        txt = file.split('/')[-1] + ".txt"
        shutil.copy(os.path.join(source_path+"/images",image), destination_path+"/images/")
        shutil.copy(os.path.join(source_path+"/labels",txt), destination_path+"/labels/")
    return

def create_folder(directory):
    try:
        if not os.path.exists(directory+'/images'):
            os.makedirs(directory+'/images')
        
        if not os.path.exists(directory+'/labels'):
            os.makedirs(directory+'/labels')
    except OSError:
        print('Error : Creating directory.' + directory+'/images')
        print('Error : Creating directory.' + directory+'/labels')


def main(argv, args) : 
    print('\n')
    print(f'argv : ', argv)
    print(f'args : ', args)
    
    print(f'args.datapath : ', args.datapath)
    print(f'args.outputpath : ', args.outputpath)
    print('\n')

    
    file_path = args.datapath
    output_path = args.outputpath 

    print("DataSet Split path IMG : " + file_path + '/images')

    image_files = glob(file_path + '/images'+ "/*.jpg")

    images = [name.replace(".jpg","") for name in image_files]

    train_names, test_names = train_test_split(images, test_size=0.2, random_state=42,shuffle=True)
    val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42,shuffle=True)

    train_dir = output_path + "/train"
    print("train save path : " + train_dir )
    create_folder(train_dir)

    val_dir = output_path + "/valid"
    print("valid save path : " + val_dir )
    create_folder(val_dir)
    
    test_dir = output_path + "/test"
    print("test save path : " + test_dir )
    create_folder(test_dir)

    batch_move_files(train_names,file_path,train_dir)
    batch_move_files(val_names,file_path,val_dir)
    batch_move_files(test_names,file_path,test_dir)

    
if __name__ == '__main__' :
    argv = sys.argv
    main(argv, args)

