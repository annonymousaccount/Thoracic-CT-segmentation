'''
Created on Aug 4, 2016

@author: annonymous
'''
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import h5py
import os
import SimpleITK as sitk
import scipy.io as scio

def process_eso(vol_out):
    voleso=vol_out==1
    voleso=voleso.astype(np.uint8)
    idxesoini=np.where(voleso>0)
    
    id_eso=np.where(vol_out==1)
    seg_eso=np.zeros_like(vol_out)
    seg_eso[id_eso]=1
    listorgan=np.where(seg_eso>0)
    zmin=np.min(listorgan[0])
    zmax=np.max(listorgan[0])
    ini_found=False
    for idx in xrange(zmin,zmax):
        eso_slice=seg_eso[idx]
        centroid=center_of_mass(eso_slice)
        if not ini_found:#if we have not found the first slice empty
            if np.isnan(centroid).any():#look for the first emppty slice
                #print 'is NAN ',idx
                ini=idx-1
                pini=list(center_of_mass(seg_eso[idx-1]))
                pini.append(idx-1)
                ini_found=True
        else:#if we have already found the first empty slice, look for the final one
            idvox=np.where(eso_slice==1)
            nvoxels=len(idvox[0])
            if not np.isnan(centroid).any() and nvoxels>5:#the slice with data and enough voxels

                #print 'final nan ',idx
                fin=idx
                pfin=list(center_of_mass(seg_eso[fin]))
                pfin.append(idx)
                #print 'pini ',pini
                #print 'pfin ',pfin
                for z in xrange(ini,fin):#we will fill the empty slices here
                    newcenter=interpolateline(pini,pfin,z)
                    #print 'new center ',newcenter
                    #print 'prev center ',center_of_mass(seg_eso[z-1])
                    translation=np.int16(np.array(newcenter)-np.array(center_of_mass(seg_eso[z-1])))
                    #print 'trans ',translation
                    #tx = tf.SimilarityTransform(translation=(0,0))#tuple(translation)
                    if z==ini:
                        slicetmp = shift(seg_eso[z-5],translation)#tf.warp(seg_eso[z-1], tx)
                    else:
                        slicetmp = shift(seg_eso[z-1],translation)#tf.warp(seg_eso[z-1], tx)
                    #print 'unique slice befor trans ',np.unique(seg_eso[z-1])
                    #print 'unique slice tmp ',np.unique(slicetmp)
                    seg_eso[z]=slicetmp
                ini_found=False
    idxeso=np.where(seg_eso>0)
    volfinal=np.copy(vol_out)
    volfinal[idxesoini]=0
    volfinal[idxeso]=1
    return volfinal
    
    
def interpolateline(p0,p1,z):
    #p1 and p2 are 3d points x,y,z and z is the slice for which we want to compute x and y
    
    x=(float(z-p0[2])/(p1[2]-p0[2]))*(p1[0]-p0[0])+p0[0]
    y=(float(z-p0[2])/(p1[2]-p0[2]))*(p1[1]-p0[1])+p0[1]
    print 'x ',x
    print 'y ',y
    return x,y



def combine_or(vol_out1,pathvolout):
    vol_out2=np.load(pathvolout)
    # volheart1=vol_out1==2
    # volheart1=volheart1.astype(np.uint8)
    # idxheartini1=np.where(volheart1>0)
    
    # volheart2=vol_out2==2
    # volheart2=volheart2.astype(np.uint8)
    # idxheartini2=np.where(volheart2>0)

    # volaorta1=vol_out1==4
    # volaorta1=volaorta1.astype(np.uint8)
    # idxaortaini1=np.where(volaorta1>0)
    
    # volaorta2=vol_out2==4
    # volaorta2=volaorta2.astype(np.uint8)
    # idxaortaini2=np.where(volaorta2>0)

    # voltrach1=vol_out1==3
    # voltrach1=voltrach1.astype(np.uint8)
    # idxtrachini1=np.where(voltrach1>0)
    
    # voltrach2=vol_out2==3
    # voltrach2=voltrach2.astype(np.uint8)
    # idxtrachini2=np.where(voltrach2>0)

    voleso1=vol_out1==1
    voleso1=voleso1.astype(np.uint8)
    idxesoini1=np.where(voleso1>0)
    
    voleso2=vol_out2==1
    voleso2=voleso2.astype(np.uint8)
    idxesoini2=np.where(voleso2>0)
    
    # maskheart=np.logical_or(volheart1,volheart2)
    # maskaorta=np.logical_or(volaorta1,volaorta2)
    # masktrachea=np.logical_or(voltrach1,voltrach2)
    maskeso=np.logical_or(voleso1,voleso2)
    
    # idxheart=np.where(maskheart>0)
    # idxaorta=np.where(maskaorta>0)
    # idxtrachea=np.where(masktrachea>0)
    idxeso=np.where(maskeso>0)
    
    volfinal=np.copy(vol_out1)
    # volfinal[idxheartini1]=0
    # volfinal[idxheartini2]=0
    # volfinal[idxheart]=2
    # volfinal[idxaortaini1]=0
    # volfinal[idxaortaini2]=0
    # volfinal[idxaorta]=4
    # volfinal[idxtrachini1]=0
    # volfinal[idxtrachini2]=0
    # volfinal[idxtrachea]=3
    volfinal[idxesoini1]=0
    volfinal[idxesoini2]=0
    volfinal[idxeso]=1
    
    return volfinal

def postprocess(vol_out):
    r=int(vol_out.shape[1]/2.0)
    c=int(vol_out.shape[2]/2.0)
    sizecropup=150
    sizecropdown=100
    sizecrop=200

    mask=np.zeros_like(vol_out)
    mask[20:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=1
    mask[-25:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=0
    vol_out*=mask
    print vol_out.shape
    print np.unique(vol_out)

    volheart=vol_out==2
    volheart=volheart.astype(np.uint8)
    idxheartini=np.where(volheart>0)

    volaorta=vol_out==4
    volaorta=volaorta.astype(np.uint8)
    idxaortaini=np.where(volaorta>0)

    voltrach=vol_out==3
    voltrach=voltrach.astype(np.uint8)
    idxtrachini=np.where(voltrach>0)

    voleso=vol_out==1
    voleso=voleso.astype(np.uint8)
    idxesoini=np.where(voleso>0)

    cc = sitk.ConnectedComponentImageFilter()

    vol_out1 = cc.Execute(sitk.GetImageFromArray(volheart))
    voltmp=sitk.RelabelComponent(vol_out1)
    volheartfiltered=sitk.GetArrayFromImage(voltmp)
    volheartfiltered=volheartfiltered==1

    vol_out2 = cc.Execute(sitk.GetImageFromArray(volaorta))
    voltmp=sitk.RelabelComponent(vol_out2)
    volaortafiltered=sitk.GetArrayFromImage(voltmp)
    volaortafiltered=volaortafiltered==1

    vol_out3 = cc.Execute(sitk.GetImageFromArray(voltrach))
    voltmp=sitk.RelabelComponent(vol_out3)
    voltrachfiltered=sitk.GetArrayFromImage(voltmp)
    voltrachfiltered=voltrachfiltered==1

    vol_out4 = sitk.GetImageFromArray(voleso)
    voltmp=sitk.BinaryMedian(vol_out4)
    volesofiltered=sitk.GetArrayFromImage(voltmp)

    maskheart=np.logical_and(volheartfiltered,volheart)
    maskaorta=np.logical_and(volaortafiltered,volaorta)
    masktrachea=np.logical_and(voltrachfiltered,voltrach)
    maskeso=volesofiltered>0#np.logical_and(volesofiltered,volesofiltered)

    for ind in xrange(volheartfiltered.shape[0]):
        maskheart[ind]=binary_fill_holes(maskheart[ind]).astype(int)
        maskaorta[ind]=binary_fill_holes(maskaorta[ind]).astype(int)
        masktrachea[ind]=binary_fill_holes(masktrachea[ind]).astype(int)
        maskeso[ind]=binary_fill_holes(maskeso[ind]).astype(int)
    idxheart=np.where(maskheart>0)
    idxaorta=np.where(maskaorta>0)
    idxtrachea=np.where(masktrachea>0)
    idxeso=np.where(maskeso>0)
    vol_out[idxheartini]=0
    vol_out[idxheart]=2
    vol_out[idxaortaini]=0
    vol_out[idxaorta]=4
    vol_out[idxtrachini]=0
    vol_out[idxtrachea]=3
    vol_out[idxesoini]=0
    vol_out[idxeso]=1

    volfinal=np.copy(vol_out)
    return volfinal


def dice(im1, im2,organid):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1=im1==organid
    im2=im2==organid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

# Load the net, list its data and params, and filter an example image.
caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('infant_fcn_solver.prototxt')
netroger = caffe.Net('deploy.prototxt','thoraxnw_iter_5000.caffemodel',caffe.TRAIN)
print("blobs {}\nparams {}".format(netroger.blobs.keys(), netroger.params.keys()))

misblobs=netroger.blobs.keys()
for i in xrange(len(misblobs)):
    print misblobs[i]
    print netroger.blobs[misblobs[i]].data.shape 
    
path_patients='/home/warehouse/CT_patients/test_set/'
_, patients, _ = os.walk(path_patients).next()#every folder is a patient
volumes_dict={}
listdceso=[]
listdcheart=[]
listdcaorta=[]
listdctrachea=[]

listdceso_p=[]
listdcheart_p=[]
listdcaorta_p=[]
listdctrachea_p=[]

#combined
listdceso_c=[]
listdcheart_c=[]
listdcaorta_c=[]
listdctrachea_c=[]
patientstmp=[patients[2]]
for idx,namepatient in enumerate(patients):
    print namepatient
    ctitk=sitk.ReadImage(os.path.join(path_patients,namepatient,namepatient+'.nii.gz')) 
    ctnp=sitk.GetArrayFromImage(ctitk)
    ctnp[np.where(ctnp>3000)]=3000#we clap the images so they are in range -1000 to 3000  HU
    muct=np.mean(ctnp)
    stdct=np.std(ctnp)

    ctnp=(1/stdct)*(ctnp-muct)#normalize each patient

    segitk=sitk.ReadImage(os.path.join(path_patients,namepatient,'GT.nii.gz'))
    segnp=sitk.GetArrayFromImage(segitk)
    vol_out=np.zeros_like(segnp,dtype='uint8')

    for i in xrange(segnp.shape[0]):
        ctslice=ctnp[i,...]
        netroger.blobs['data'].data[0,0,...] = ctslice
        netroger.forward()
        out = netroger.blobs['pred'].data[0].argmax(axis=0)
        vol_out[i]=out
        print 'slice {} done'.format(i)
    print namepatient


   

    dceso=dice(vol_out, segnp,1)
    dcheart=dice(vol_out, segnp,2)
    dctrachea=dice(vol_out, segnp,3)
    dcaorta=dice(vol_out, segnp,4)
    listdceso.append(dceso)
    listdcheart.append(dcheart)
    listdcaorta.append(dcaorta)
    listdctrachea.append(dctrachea)
    print 'eso {}'.format(dceso) 
    print 'heart {}'.format(dcheart)
    print 'trachea {}'.format(dctrachea)
    print 'aorta {}'.format(dcaorta)

    np.save(namepatient+'_volout.npy',vol_out)

    
    print 'with postprocessing '
    vol_out=postprocess(vol_out)
    vol_out=process_eso(vol_out)
    pathvolout=os.path.join('/home/Desktop/Caffes/caffe/examples/autocontext_2dCT',namepatient+'_voloutac.npy')
    vol_outcombined=combine_or(vol_out,pathvolout)
    dceso=dice(vol_out, segnp,1)
    dcheart=dice(vol_out, segnp,2)
    dctrachea=dice(vol_out, segnp,3)
    dcaorta=dice(vol_out, segnp,4)
    listdceso_p.append(dceso)
    listdcheart_p.append(dcheart)
    listdcaorta_p.append(dcaorta)
    listdctrachea_p.append(dctrachea)
    print 'eso {}'.format(dceso) 
    print 'heart {}'.format(dcheart)
    print 'trachea {}'.format(dctrachea)
    print 'aorta {}'.format(dcaorta)

    print 'combined with fcn+ac'
    dceso=dice(vol_outcombined, segnp,1)
    dcheart=dice(vol_outcombined, segnp,2)
    dctrachea=dice(vol_outcombined, segnp,3)
    dcaorta=dice(vol_outcombined, segnp,4)
    listdceso_c.append(dceso)
    listdcheart_c.append(dcheart)
    listdcaorta_c.append(dcaorta)
    listdctrachea_c.append(dctrachea)
    print 'eso {}'.format(dceso) 
    print 'heart {}'.format(dcheart)
    print 'trachea {}'.format(dctrachea)
    print 'aorta {}'.format(dcaorta)

print 'Global normal'   
print 'mean eso ',np.mean(listdceso),'+- ',np.std(listdceso)
print 'mean heart ',np.mean(listdcheart),'+- ',np.std(listdcheart)
print 'mean trachea ',np.mean(listdctrachea),'+- ',np.std(listdctrachea)
print 'mean aorta ',np.mean(listdcaorta),'+- ',np.std(listdcaorta)

print 'Global with postprocessing'   
print 'mean eso ',np.mean(listdceso_p),'+- ',np.std(listdceso_p)
print 'mean heart ',np.mean(listdcheart_p),'+- ',np.std(listdcheart_p)
print 'mean trachea ',np.mean(listdctrachea_p),'+- ',np.std(listdctrachea_p)
print 'mean aorta ',np.mean(listdcaorta_p),'+- ',np.std(listdcaorta_p)

print 'Global combined'   
print 'mean eso ',np.mean(listdceso_c),'+- ',np.std(listdceso_c)
print 'mean heart ',np.mean(listdcheart_c),'+- ',np.std(listdcheart_c)
print 'mean trachea ',np.mean(listdctrachea_c),'+- ',np.std(listdctrachea_c)
print 'mean aorta ',np.mean(listdcaorta_c),'+- ',np.std(listdcaorta_c)
