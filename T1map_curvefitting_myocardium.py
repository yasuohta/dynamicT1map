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


def sort_dicom_files(dicom_files):
    sorted_files = sorted(dicom_files, key=lambda x: pydicom.read_file(x).SliceLocation)
    return sorted_files

def aquisitiontimechange(dataset):
    dtime = datetime.datetime.strptime(dataset[0x0008,0x0032].value, '%H%M%S.%f')
    stime = datetime.datetime.strptime(starttime, '%H%M%S')
    delta = dtime - stime
    return delta.total_seconds()/60

def func(t, a, alpha, b, beta):
   
    return a /(beta-alpha)*( (np.exp(-alpha * (t))) - (np.exp(-beta * (t)))) +a /(beta-alpha)*( (np.exp(-alpha * (t-t2))) - (np.exp(-beta * (t-t2)))) #brix 2nd
 
def func2(param,t, c, T=1.0):
    A = param[0]
    alpha = param[1]
    B = param[2]
    beta = param[3]
    return func(t, *param) - c


def filetoconc(path):

    files = []
    print('DICOMfile: {}'.format(path))
    for fname in glob.glob(path+'/'+'*min.dcm', recursive=False): 
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
    for nativefile in glob.glob(path + '*native.dcm', recursive = False):
        print(nativefile)
        native.append (pydicom.dcmread(nativefile))
    nativeT1 = native[0].pixel_array
    

    #concentration
    # initialize
    row, cor, phase = img3d.shape
    i,j=0,0
    A = np.zeros((img3d.shape[0],img3d.shape[1]))
    alpha = np.zeros((img3d.shape[0],img3d.shape[1]))
    B = np.zeros((img3d.shape[0],img3d.shape[1]))
    beta = np.zeros((img3d.shape[0],img3d.shape[1]))
    resi = np.zeros((img3d.shape[0],img3d.shape[1]))
    print('cleared',resi)
    parameter = np.zeros((img3d.shape[0],img3d.shape[1],4))
    relaxivity = 5.0 # r1 @3.0T
    initial = [0.8,0.0003,0.8,0.05]   

    # time calculation
    xt = np.zeros(len(slices))
    for i, s in enumerate(slices):
        xt[i]= aquisitiontimechange(s)


        for i in tqdm(range(img3d.shape[0])):  #fitting
            for j in range(img3d.shape[1]):

                yt = np.zeros(len(slices))

                for k, val in enumerate(slices):
                    yt[k] = (1/(img3d[i,j,k])-1/(nativeT1[i,j]))/relaxivity*1000

                    if np.isnan(yt[k]):
                        yt[k] = 0
                    if np.isinf(yt[k]):
                        yt[k] = 0   

                result_param = optimize.leastsq(func2, initial, args=(xt, yt))
                parameter[i,j,:] = result_param[0][:]
                A[i,j] = parameter[i,j,0]
                alpha[i,j] = parameter[i,j,1]
                B[i,j] = parameter[i,j,2]
                beta[i,j] = parameter[i,j,3]

                #residuals
                residuals =  yt -func(xt, A[i,j], alpha[i,j], B[i,j], beta[i,j])
                rss = np.sum(residuals**2) 
                tss = np.sum((yt-np.mean(yt))**2)
                r_squared = 1 - (rss / tss)
                resi[i,j] = r_squared
            
    
    
    return A,alpha,B,beta,resi,parameter, files

def imagesave(path,A,alpha,B,beta,resi):
    print('save results in folder:'+path)
    np.savetxt(path+'A_Brix2.csv', A, delimiter=",")
    np.savetxt(path+'alpha_Brix2.csv', alpha, delimiter=",")
    np.savetxt(path+'B_Brix2.csv', B, delimiter=",")
    np.savetxt(path+'beta_Brix2.csv', beta, delimiter=",")
    np.savetxt(path+'r_squared_Brix2.csv', resi, delimiter=",")

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

def conc_time_map(folder,parameter):
    conc_time=[3,5,7,10,15,30,60]  #output min

    conc = np.zeros((A.shape[0],A.shape[1],len(conc_time)))

    def conc_map(time):
        map = np.zeros((A.shape[0],A.shape[1]))  

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                    map[i,j] =  func(time,*parameter[i,j,:])

        return map[:,:]

    for i, time in enumerate(conc_time):
            conc[:,:,i] = conc_map(time) 
            dicom_save(conc[:,:,i],1.0,0.5,os.path.join(resultpath,str(conc_time[i])+"min_"+folder+"_conc.dcm"),str(conc_time[0])+"min")


    fig, axis = plt.subplots(2, 4, figsize=(15, 8))
    axis[0,0].set_title(str(conc_time[0])+"min")
    axis[0,1].set_title(str(conc_time[1])+"min")
    axis[0,2].set_title(str(conc_time[2])+"min")
    axis[0,3].set_title(str(conc_time[3])+"min")
    axis[1,0].set_title(str(conc_time[4])+"min")
    axis[1,1].set_title(str(conc_time[5])+"min")
    axis[1,2].set_title(str(conc_time[6])+"min")
    axis[1,3].set_title("residuals")
    
    im = axis[0,0].imshow(conc[:,:,0],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im,ax=axis[0,0], shrink=0.62)
    
    im2 = axis[0,1].imshow(conc[:,:,1],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im2,ax=axis[0,1], shrink=0.62)
    
    im3 = axis[0,2].imshow(conc[:,:,2],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im3,ax=axis[0,2], shrink=0.62)
    
    im4 = axis[0,3].imshow(conc[:,:,3],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im4,ax=axis[0,3], shrink=0.62)
    
    im5 = axis[1,0].imshow(conc[:,:,4],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im5,ax=axis[1,0], shrink=0.62)
    
    im6 = axis[1,1].imshow(conc[:,:,5],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im6,ax=axis[1,1], shrink=0.62)
    
    im7 = axis[1,2].imshow(conc[:,:,5],vmin=0, vmax=0.6,cmap = "jet")
    fig.colorbar(im7,ax=axis[1,2], shrink=0.62)
    
    im8 = axis[1,3].imshow(resi,vmin=0, vmax=1.0,cmap = "jet")
    fig.colorbar(im8,ax=axis[1,3], shrink=0.62)
    
    plt.savefig(os.path.join(path,files[0].StudyDate+folder+"conc_figure.png")) #まとめて保存
    

def conc_time_map2(folder,parameter):
    conc_time=[20,25]  # output min

    conc = np.zeros((A.shape[0],A.shape[1],len(conc_time)))

    def conc_map(time):
        map = np.zeros((A.shape[0],A.shape[1]))  

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                    map[i,j] =  func(time,*parameter[i,j,:])

        return map[:,:]

    for i, time in enumerate(conc_time):
            conc[:,:,i] = conc_map(time) 
            dicom_save(conc[:,:,i],1.0,0.5,os.path.join(resultpath,str(conc_time[i])+"min_"+folder+"_conc.dcm"),str(conc_time[0])+"min")


#main
basepath="E:/"
paths = [path for path in glob.glob(basepath, recursive=False) if os.path.isdir(path)]
paths = [path for path in paths if "stats_" not in path]


for path in paths:
    print("計算中:",path)

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
    print(f"1st bolus:{starttime}, 2nd bolus:{starttime2nd}, duration:{t2:.3f}min")


    folders = sorted(glob.glob((path+"/*min")))
    prefolder = os.path.join(path,'native')
    resultfolder = "resultsBrix"
    resultpath = os.path.join(path,resultfolder)

    folders.append(prefolder) 

    execfolders=["base","mid","apex"]


    for folder in execfolders:
        if not os.path.exists(os.path.join(path,folder)):
            print("making path:",folder)     
            os.makedirs(os.path.join(path,folder))


    if not os.path.exists(os.path.join(path,"native")):
            print("making path:",prefolder)
            os.makedirs(prefolder)

    if not os.path.exists(os.path.join(path,resultfolder)):
            print("making path:",resultfolder)
            os.makedirs(os.path.join(path,resultfolder))


    for infolder in folders: 
        outfilename=[]
        files = sorted(glob.glob(infolder+"/*.dcm"))
        for i,file in enumerate(files): 
            print(file)

            if file.endswith(".dcm"):
                file_path = os.path.join(infolder, file)
                ds = pydicom.dcmread(file_path)
            
                date = ds.StudyDate
                
                time = ds.AcquisitionTime[:6]
                
                loc = str(ds.SliceLocation)
                
                new_file_name = f"{loc}_{time}_{date}"           

            outfilename.append([file, new_file_name+"_"+os.path.split(infolder)[1]+".dcm", loc])

        sortedlist = sorted(outfilename, key = lambda x: x[2])
        print("sorted:",sortedlist,"\n")

        for i,file in enumerate(sortedlist):
            sortedfile=os.path.join(path,execfolders[i],sortedlist[i][1])
            shutil.copy(sortedlist[i][0],sortedfile)
            print(i,sortedlist[i][0],"--",os.path.split(infolder)[1],"--",sortedfile)

    for folder in execfolders:

        execpath=path+'/'+folder+'/'
        print("calculating",folder)
        A,alpha,B,beta,resi,parameter,files = filetoconc(execpath)   

        imagesave(execpath,A,alpha,B,beta,resi)

        print('save DICOM files in folder:'+path)
        dicom_save(A, 1, 0.5, os.path.join(resultpath,folder+"_A.dcm"), "A")
        dicom_save(alpha, 0.2, 0.1, os.path.join(resultpath,folder+"_aplha.dcm"), "alpha")
        dicom_save(beta, 50, 25, os.path.join(resultpath, folder+"_beta.dcm"), "beta")
        dicom_save(resi, 1, 0.5, os.path.join(resultpath, folder+"_r_squared.dcm"), "r_square")

        conc_time_map(folder,parameter)
        conc_time_map2(folder,parameter) 
