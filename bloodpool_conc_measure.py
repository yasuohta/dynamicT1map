
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
from scipy import optimize
import pandas as pd
import datetime
from PIL import Image
from scipy.stats import kurtosis,skew,mode




def aquisitiontimechange(dataset):
    dtime = datetime.datetime.strptime(dataset[0x0008,0x0032].value, '%H%M%S.%f')
    stime = datetime.datetime.strptime(starttime, '%H%M%S')
    delta = dtime - stime
    return delta.total_seconds()/60


def func(t, a, alpha, b, beta):
    #2-compartment for LV
    return a*np.exp(-alpha*t)+b*np.exp(-beta*t) +(a*np.exp(-alpha*(t-t2))+b*np.exp(-beta*(t-t2)))


def dicom_save(inputdata,ww,wl,file,title):
    ds = files[0] #files 0

    
    data = inputdata.astype(np.float32) #float32
    
    ds[0x0010,0x0010].value = title #patient name
    ds[0x0028,0x0100].value = 32  #32bits
    ds.add_new([0x0028,0x1050],'DS', str(wl)) #WL
    ds.add_new([0x0028,0x1051],'DS', str(ww)) #WW

    pydicom.pixel_data_handlers.util.pixel_dtype(ds,as_float=True)
    ds.FloatPixelData = data.tobytes()  
    ds.save_as(file)
    print(file)
                 

def statswrite_partial(writefilename, path,file,maskfile, mean, sd, median, q25, q75, kurto, skewness, one): 
    dt_now=datetime.datetime.now()
    dt_date = dt_now.strftime('%Y/%m/%d')
    dt_time = dt_now.strftime('%H:%M:%S')
    df=[]
    #index
    cols = ['case','slice','mean', 'SD', 'median' ,'q25','q75','kurtosis' , 'skewness','processed_date','processed_time']
    lists= np.array([[os.path.basename(os.path.dirname(path))+"/"+os.path.basename(path), file, mean, sd, median, q25, q75, kurto, skewness, dt_date, dt_time]],dtype =object)


    if not os.path.exists(os.path.join(statspath,writefilename)):
        df = pd.DataFrame(lists, columns = cols)
        df.to_csv(os.path.join(statspath,writefilename),index=False)

        print('new filecreated')
    else:
        df = pd.read_csv(os.path.join(statspath,writefilename))
        df2= pd.DataFrame(lists, columns = cols)
        df2.to_csv(os.path.join(statspath,writefilename),mode='a',index=False,header=False)        
        
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

def statsfunc(map, filename, statsparam):
        one =[]
        one = map.ravel()  
        one = one[one.nonzero()] 
    
       
        mean = np.average(one)
        sd = np.std(one)
        median = np.median(one)
        q75, q25 = np.percentile(one, [75 ,25])
        kurto = kurtosis(one,nan_policy = 'omit')
        skewness = skew(one)
        print("processing",filename)
        print(statsparam)
        print(f"mean of masked region:{mean:.3f}±{sd:.3f} median: {median:.3f} IQR:{q25:.3f},{q75:.3f}")
        print('kurtosis: {:.3f} skewness: {:.3f}'.format(kurto, skewness))   
    
        params = [mean, sd, median, q25, q75, kurto, skewness, one]
        return params
    
#main
basepath="E:/"

paths = sorted(glob.glob(basepath,recursive=False))
paths = [path for path in paths if "stats_" not in path]

resultfolder = "LVresults"
statsfolder = "stats_concdiff"

statspath=os.path.join(os.path.dirname(os.path.dirname(basepath)),statsfolder) #
if not os.path.exists(statspath):
        print("making a path:",statspath)
        os.makedirs(statspath)

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
    t2=delta.total_seconds()/60 
    print(f"1st bolus:{starttime}, 2nd bolus:{starttime2nd}, duration{t2:.3f}min")

    execfolders=["base","mid","apex"]
    masked_exist_sum=np.array([None,None], dtype=np.float32)
    masked_estimated_sum=np.array([None,None], dtype=np.float32)
    masked_delta_sum=np.array([None,None], dtype=np.float32)
    
    #loop slice
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


        # initial
        row, cor, phase = img3d.shape
        i,j=0,0

        #parameters
        filebase=folder+"_LV_aplha.dcm"   #
        alphafile=os.path.join(resultpath,filebase)
        alpha = dicomtoarray(alphafile)
       

        #beta   
        filebase=folder+"_LV_beta.dcm"
        betafile=os.path.join(resultpath,filebase)
        beta = dicomtoarray(betafile)
               
        #A
        filebase=folder+"_LV_A.dcm"
        afile=os.path.join(resultpath,filebase)
        A = dicomtoarray(afile)
        
        #B  
        filebase=folder+"_LV_B.dcm"
        bfile=os.path.join(resultpath,filebase)
        B = dicomtoarray(bfile)
        
        exist_concmap = np.zeros((row,cor,phase))
        estimated_concmap = np.zeros((row,cor,phase))
        parameter = np.zeros((row,cor,phase))
        masked_exist_concmap = np.zeros((row,cor,phase))
        masked_estimated_concmap = np.zeros((row,cor,phase))
        masked_delta_concmap = np.zeros((row,cor,phase))

        if position == 0:
            masked_exist_sum=np.zeros((row,cor,len(slices),len(execfolders)))
            masked_estimated_sum=np.zeros((row,cor,len(slices),len(execfolders)))
            masked_delta_sum=np.zeros((row,cor,len(slices),len(execfolders)))
            print("initialized sum")
        
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

        # bloadcast
        a_exp = A[:, :, np.newaxis]
        alpha_exp = alpha[:, :, np.newaxis]
        b_exp = B[:, :, np.newaxis]
        beta_exp = beta[:, :, np.newaxis]

        # estimated conc
        estimated_concmap = func(xt[0:], a_exp, alpha_exp, b_exp, beta_exp)
        delta_concmap = exist_concmap-estimated_concmap



        #mask
        maskfile=glob.glob(os.path.join(path, folder, "*native_LV_mask.png"), recursive = False)
        try:
            myomask = np.asarray(Image.open(maskfile[0]))

        except OSError as e:
            print(e)
            print("file not found:",maskfile)
        
        im_bool=myomask>128  #mask
        
        for k, val in enumerate(slices): #time loop
            
            #mask
            masked_exist_concmap[:,:,k] = np.nan_to_num(exist_concmap[:,:,k]*im_bool) 
            masked_estimated_concmap[:,:,k] = np.nan_to_num(estimated_concmap[:,:,k]*im_bool)
            masked_delta_concmap[:,:,k] = np.nan_to_num(delta_concmap[:,:,k]*im_bool)
            
        
            #stats
            masked_exist_params = statsfunc(masked_exist_concmap[:,:,k], "", "masked_exist_params")
            masked_exist_sum[:,:,k,position] = masked_exist_concmap[:,:,k] #global
            
            
            masked_estimated_params = statsfunc(masked_estimated_concmap[:,:,k], "", "masked_estimated_params")
            masked_estimated_sum[:,:,k,position] = masked_estimated_concmap[:,:,k] #global
            
            masked_delta_params = statsfunc(masked_delta_concmap[:,:,k], "", "masked_delta_params")
            masked_delta_sum[:,:,k,position] = masked_delta_concmap[:,:,k]#global
    
            
        
            path_write=os.path.join(path, str(xt[k])) 
            print("writing:",folder,k,path_write)
            statswrite_partial("existmap_LV.csv",path_write, folder, maskfile, *masked_exist_params)
            dicom_save(exist_concmap[:,:,k],1.0,0.5,os.path.join(resultpath,str(xt[k])+"min_"+folder+"_LVconc_existed.dcm"),str(xt[k])+"min_measured")
            
            statswrite_partial("estimatedmap_LV.csv",path_write, folder, maskfile, *masked_estimated_params)
            dicom_save(estimated_concmap[:,:,k],1.0,0.5,os.path.join(resultpath,str(xt[k])+"min_"+folder+"_LVconc_estimated.dcm"),str(xt[k])+"min_estimated")
            
            statswrite_partial("deltamap_LV.csv",path_write, folder, maskfile, *masked_delta_params)
            dicom_save(delta_concmap[:,:,k],1.0,0.5,os.path.join(resultpath,str(xt[k])+"min_"+folder+"_LVconc_delta.dcm"),str(xt[k])+"min_delta")
 
    
    for m, val in enumerate(slices):
     
        exist_params_sum = statsfunc(masked_exist_sum[:,:,m,:], "global", "masked_exist_params")    
        estimated_params_sum = statsfunc(masked_estimated_sum[:,:,m,:] , "global", "masked_estimated_params") 
        delta_params_sum = statsfunc(masked_delta_sum[:,:,m,:], "global", "masked_delta_params") 
        path_write=os.path.join(path, str(xt[m]))
        statswrite_partial("existmap_LV.csv",path_write, "global", maskfile, *exist_params_sum)
        statswrite_partial("estimatedmap_LV.csv",path_write,"global", maskfile, *estimated_params_sum)
        statswrite_partial("deltamap_LV.csv",path_write, "global", maskfile, *delta_params_sum)    