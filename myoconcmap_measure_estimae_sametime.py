import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import re
import shutil
import datetime
from tqdm import tqdm
import pandas as pd
import datetime
from PIL import Image

def aquisitiontimechange(dataset):
    dtime = datetime.datetime.strptime(dataset[0x0008,0x0032].value, '%H%M%S.%f')
    stime = datetime.datetime.strptime(starttime, '%H%M%S')
    delta = dtime - stime
    return delta.total_seconds()/60

def func(t, a, alpha, b, beta):
    return a /(beta-alpha)*( (np.exp(-alpha * (t))) - (np.exp(-beta * (t)))) +a /(beta-alpha)*( (np.exp(-alpha * (t-t2))) - (np.exp(-beta * (t-t2)))) 
    
def dicom_save(inputdata,ww,wl,file,title):
    ds = files[0] 

    
    data = inputdata.astype(np.float32) #float32
    
    ds[0x0010,0x0010].value = title #patient name
    ds[0x0028,0x0100].value = 32    #32bits
    ds.add_new([0x0028,0x1050],'DS', str(wl)) #WL
    ds.add_new([0x0028,0x1051],'DS', str(ww)) #WW

    pydicom.pixel_data_handlers.util.pixel_dtype(ds,as_float=True)
    ds.FloatPixelData = data.tobytes()               
    ds.save_as(file)
    print(file)
                   
def dicomtoarray(filename):
        
        print(filename)
        try:
            ds = pydicom.dcmread(filename)

        except OSError as e:
            print(e)
            print("file not found:", filename)
            
        # create 3D array
        
        pixel_data = ds[0x7fe0, 0x0008].value
        map = np.frombuffer(pixel_data, dtype=np.float32)
        map = map.reshape(ds.Rows, ds.Columns)  
        print(map.shape)
        return map
    
#main
basepath="E:/国立循環器病研究センター　放射線部 Dropbox/Ohta Yasutoshi/ohtancvc/NCVC研究/dynamicT1map/データ/テストデータ/*"

paths = sorted(glob.glob(basepath,recursive=False))
paths = [path for path in paths if "stats_" not in path]

resultfolder = "resultsBrix"

relaxivity = 5.0 # r1 @3.0T

for path in paths:
    resultpath = os.path.join(path,resultfolder)
    print("calculating:",resultpath)

    with open(os.path.join(path,'readme.txt'), 'r') as f:

        for line in f:
                if 'time 1' in line:

                    match = re.search(r'(\d{6})', line)

                    if match:

                        time = match.group(1)

                    else:
                        print("no time 1")


                if 'time 2' in line:
                    match = re.search(r'(\d{6})', line)
                    if match:

                        time2 = match.group(1)

                    else:
                        print("no time 2")



    if time < time2:
        starttime, starttime2nd = time, time2
    else:
        starttime, starttime2nd = time2, time 


    time1 = datetime.datetime.strptime(starttime, '%H%M%S')
    time2 = datetime.datetime.strptime(starttime2nd, '%H%M%S')
    delta=time2-time1
    t2=delta.total_seconds()/60 #t2, global variable
    print(f"1st bolus:{starttime}, 2nd bolus:{starttime2nd}, interval:{t2:.3f}min")

    execfolders=["base","mid","apex"]
    masked_exist_sum=np.array([None,None], dtype=np.float32)
    masked_estimated_sum=np.array([None,None], dtype=np.float32)
    masked_delta_sum=np.array([None,None], dtype=np.float32)
    
    for position, folder in enumerate(execfolders):
        execpath=path+'/'+folder+'/'
        print("calculating",folder)

        files = []
        print('DICOM file: {}'.format(path))
        for fname in glob.glob(execpath+'/'+'*min.dcm', recursive=False): 
            files.append(pydicom.dcmread(fname))
            print(fname)

        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, 'SliceLocation'):
                slices.append(f)
            else:
                skipcount = skipcount + 1

        print("files: {}".format(len(slices)))
        print("skipped, no SliceLocation: {}".format(skipcount))

        # ensure they are in the correct order
        slices = sorted(slices, key=lambda s: s.AcquisitionTime)

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(slices):
            img2d = s.pixel_array
            img3d[:, :, i] = img2d

        #native_load
        nativeT1 = np.zeros(img_shape)
        native =[]
        for nativefile in glob.glob(execpath + '*native.dcm', recursive = False):
            print(nativefile)
            native.append (pydicom.dcmread(nativefile))
        nativeT1 = native[0].pixel_array

        row, cor, phase = img3d.shape
        i,j=0,0

        #alpha
        filebase=folder+"_aplha.dcm"   
        alphafile=os.path.join(resultpath,filebase)
        alpha = dicomtoarray(alphafile)
       
        #beta
        filebase=folder+"_beta.dcm"
        betafile=os.path.join(resultpath,filebase)
        beta = dicomtoarray(betafile)
               
        #A
        filebase=folder+"_A.dcm"
        afile=os.path.join(resultpath,filebase)
        A = dicomtoarray(afile)
        
        #B  dummy  
        B = A
        
        exist_concmap = np.zeros((row,cor,phase))
        estimated_concmap = np.zeros((row,cor,phase))
        parameter = np.zeros((row,cor,phase))
 
        # time calculation
        xt = np.zeros(len(slices))
        for i, s in enumerate(slices):
            xt[i]= aquisitiontimechange(s)

        native_inv = 1 / nativeT1[:, :, np.newaxis]
        img_inv = 1 / img3d
        yt = (img_inv - native_inv) / relaxivity * 1000

        yt[np.isnan(yt)] = 0
        yt[np.isinf(yt)] = 0


        exist_concmap = (img_inv - native_inv) / relaxivity * 1000

        a_exp = A[:, :, np.newaxis]
        alpha_exp = alpha[:, :, np.newaxis]
        b_exp = B[:, :, np.newaxis]
        beta_exp = beta[:, :, np.newaxis]

        estimated_concmap = func(xt[0:], a_exp, alpha_exp, b_exp, beta_exp)
        delta_concmap = exist_concmap-estimated_concmap
             
        for k, val in enumerate(slices): 
            #write
            path_write=os.path.join(path, str(xt[k])) 
            dicom_save(exist_concmap[:,:,k],1.0,0.5,os.path.join(resultpath,str(xt[k])+"min_"+folder+"conc_existed.dcm"),str(xt[k])+"min_measured")
            dicom_save(estimated_concmap[:,:,k],1.0,0.5,os.path.join(resultpath,str(xt[k])+"min_"+folder+"conc_estimated.dcm"),str(xt[k])+"min_estimated")
            dicom_save(delta_concmap[:,:,k],1.0,0.5,os.path.join(resultpath,str(xt[k])+"min_"+folder+"conc_delta.dcm"),str(xt[k])+"min_delta")
            