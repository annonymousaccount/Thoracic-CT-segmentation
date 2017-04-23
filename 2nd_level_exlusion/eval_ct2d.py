'''
Created on Aug 4, 2016

@author: annonymous
'''
import time
import argparse
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')


import h5py
import os
os.environ['GLOG_minloglevel'] = '3' #verbose caffe
import caffe
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


def postprocess(vol_out):
    r=int(vol_out.shape[1]/2.0)
    c=int(vol_out.shape[2]/2.0)
    sizecropup=150
    sizecropdown=100
    sizecrop=150#200 heart

    mask=np.zeros_like(vol_out)
    #sizecrop=200
    #mask[20:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=1
    #mask[-25:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=0

    #trachea
    sizecrop=150
    mask[20:,r-sizecrop/2:r+sizecrop/2,c-sizecrop/2:c+sizecrop/2]=1
    mask[-25:,r-sizecrop/2:r+sizecrop/2,c-sizecrop/2:c+sizecrop/2]=0


    vol_out*=mask
    print vol_out.shape
    print np.unique(vol_out)

    
    voleso=vol_out==1
    voleso=voleso.astype(np.uint8)
    idxesoini=np.where(voleso>0)

    cc = sitk.ConnectedComponentImageFilter()

    
    vol_out4 =cc.Execute(sitk.GetImageFromArray(voleso))
    #voltmp=sitk.BinaryMedian(vol_out4)
    voltmp=sitk.RelabelComponent(vol_out4)
    volesofiltered=sitk.GetArrayFromImage(voltmp)
    volesofiltered=volesofiltered==1
    
    maskeso=np.logical_and(volesofiltered,volesofiltered)

    for ind in xrange(volesofiltered.shape[0]):        
        maskeso[ind]=binary_fill_holes(maskeso[ind]).astype(int)


    idxeso=np.where(maskeso>0)
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




parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--model', help='number of model to use', required=True)
parser.add_argument('-p','--prefix', help='prefix of protoxt (thoraxw  or  thoraxnw)', required=True)
args = vars(parser.parse_args())
nmodel=args['model']
prefix=args['prefix']
print 'arguments '+nmodel+','+prefix

# Load the net, list its data and params, and filter an example image.
caffe.set_device(1)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('infant_fcn_solver.prototxt')
netroger = caffe.Net('deploy.prototxt','model_trachea/'+prefix+'_iter_{}.caffemodel'.format(nmodel),caffe.TEST)
print("blobs {}\nparams {}".format(netroger.blobs.keys(), netroger.params.keys()))

misblobs=netroger.blobs.keys()
for i in xrange(len(misblobs)):
    print misblobs[i]
    print netroger.blobs[misblobs[i]].data.shape 
    
path_patients='/home/annonymous/CT_cleaned/'#
path_constraints='./subject_constraints_eso'#'./subject_constraints_eso'#
organ_target=3#eso 1, heart 2, trachea 3, aorta 4
_, patients, _ = os.walk(path_patients).next()#every folder is a patient
patients.sort()
patients_tmp=patients[-4:]#last four
volumes_dict={}
listdceso=[]
listdcheart=[]
listdcaorta=[]
listdctrachea=[]

listdceso_p=[]
listdcheart_p=[]
listdcaorta_p=[]
listdctrachea_p=[]
for idx,namepatient in enumerate(patients_tmp):
    print namepatient
    ctitk=sitk.ReadImage(os.path.join(path_patients,namepatient,namepatient+'.nii.gz')) 
    ctnp=sitk.GetArrayFromImage(ctitk)
    ctnp[np.where(ctnp>3000)]=3000#we clap the images so they are in range -1000 to 3000  HU
    muct=np.mean(ctnp)
    stdct=np.std(ctnp)

    ctnp=(1/stdct)*(ctnp-muct)#normalize each patient

    segitk=sitk.ReadImage(os.path.join(path_constraints,namepatient,namepatient+'_out_unet.nii.gz'))#out unet
    #segitk=sitk.ReadImage(os.path.join(path_constraints,namepatient,namepatient+'_volout_combined.nii.gz'))#out unet+const_eso
    #segitk=sitk.ReadImage(os.path.join(path_patients,namepatient,'GT.nii.gz'))
    segnp=sitk.GetArrayFromImage(segitk)

    constrains=np.copy(segnp)
    constrains=constrains.astype(np.float32)

    cnt_tmp=1.0
    for idorgan in xrange(1,5):
        if idorgan==organ_target:
            constrains[np.where(segnp==idorgan)]=0
        else:
            constrains[np.where(segnp==idorgan)]=cnt_tmp/3.0
            cnt_tmp+=1

    """cnt_tmp=1.0
    for idorgan in xrange(1,5):
        #print 'organ ',idorgan
        if idorgan==organ_target:
            constrains[np.where(segnp==idorgan)]=0
        else:
            indices_organ=np.where(segnp==idorgan)
            if len(indices_organ[0]>0):
                constrains[np.where(segnp==idorgan)]=cnt_tmp/3.0
                cnt_tmp+=1"""
    

    print 'unique whole constraints ', np.unique(constrains)

    vol_out=np.zeros_like(segnp,dtype='uint8')
    start = time.time()
    for i in xrange(segnp.shape[0]):
        ctslice=ctnp[i,...]
        const_slice=constrains[i,...]
        netroger.blobs['data'].data[0,0,...] = ctslice
        netroger.blobs['data'].data[0,1:,...] = const_slice
        netroger.forward()
        out = netroger.blobs['softmax'].data[0].argmax(axis=0)
        vol_out[i]=out
        print 'slice {} done'.format(i)
    end = time.time()
    print 'time elapsed ', end-start
    print namepatient
    

    volout=sitk.GetImageFromArray(vol_out)
    sitk.WriteImage(volout,namepatient+'_volout_constrained_sm_trachea.nii.gz')

    seg_compare=(segnp==organ_target)
    dceso=dice(vol_out, seg_compare,1)
    listdceso.append(dceso)
    print 'organ {0}, dice: {1}'.format(organ_target, dceso) 
            
    print 'with postprocessing '
    vol_out=postprocess(vol_out)
    dceso=dice(vol_out, seg_compare,1)
    listdceso_p.append(dceso)
    print 'eso {}'.format(dceso) 

    volout=sitk.GetImageFromArray(vol_out)
    sitk.WriteImage(volout,namepatient+'_volout_constrained_sm_trachea_post.nii.gz') 
    
print 'Global normal'   
print 'mean organ target ',np.mean(listdceso),'+- ',np.std(listdceso)

print 'Global with postprocessing'   
print 'mean organ target ',np.mean(listdceso_p),'+- ',np.std(listdceso_p)
