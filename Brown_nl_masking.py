#!/usr/bin/env python3
# standard lib

'''
fixed atlas image: temp_nih_T2w_atl_float.nii.gz

moving images for each subject: T2w_acpc_dc.nii.gz

Nipype ants registration documentation: https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.ants/registration.html#registration

Aplly tranfsorm: https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.ants/resampling.html

path = "/home/exacloud/lustre1/fnl_lab/projects/INFANT/GEN_INFANT/masking_test"
'''

import argparse
import os
from glob import glob
import random

# external libs
import nipype.pipeline.engine as pe
from nipype.interfaces import ants, utility, fsl


def main():
    parser = generate_parser()
    args = parser.parse_args()
    path = args.path
    njobs = args.njobs

    atlas = ['/home/groups/brainmri/infant/NIH_ATLASES/nihpd_asym_00-02_t1w.nii.gz',
                   '/home/exacloud/lustre1/fnl_lab/projects/INFANT/GEN_INFANT/masking_test/temp_nih_T2w_atl_float.nii.gz']
    atlas_brain = '/home/exacloud/lustre1/fnl_lab/projects/INFANT/GEN_INFANT/masking_test/temp_nih_T2w_atl_brain_float.nii.gz'

    randint = '_twochannel'
    warped_dir = os.path.join('./nr_masking_dir', 'warped{}'.format(randint))

    #create list of T1w and T2w images for each subject
    pattern = os.path.join(path, '*/T1w_acpc_dc.nii.gz')
    t1w_list = glob(pattern)

    pattern = os.path.join(path, '*/T2w_acpc_dc.nii.gz')
    t2w_list = glob(pattern)

    #t1w_t2w_list = zip(t1w_list, t2w_list)
    #t1w_t2w_list = list(t1w_t2w_list)

    t1w_t2w_tuple = zip(t1w_list, t2w_list)
    t1w_t2w_tuple = list(t1w_t2w_tuple)
    t1w_t2w_list = [list(i) for i in t1w_t2w_tuple]

    register(warped_dir, atlas, atlas_brain, t1w_t2w_list, t2w_list, n_jobs=njobs)

def generate_parser():
    parser = argparse.ArgumentParser(description='non-linear registration from Brown')
    parser.add_argument('path', help='path to images')
    parser.add_argument('--njobs', default=1, type=int, help='number of cpus to utilize')
    return parser

def register(warped_dir, atlas_image, atlas_image_brain, subject_T1ws_T2ws, subject_T2ws, n_jobs):

    input_spec = pe.Node(
        utility.IdentityInterface(fields=['subject_image_list', 'subject_image', 'atlas_image', 'atlas_image_brain']),
        iterables=[('subject_image_list', subject_T1ws_T2ws),
                   ('subject_image', subject_T2ws)],
        synchronize=True,
        name='input_spec'
    )
    # set input_spec
    input_spec.inputs.subject_image_list = subject_T1ws_T2ws
    input_spec.inputs.subject_image = subject_T2ws
    input_spec.inputs.atlas_image = atlas_image
    input_spec.inputs.atlas_image_brain = atlas_image_brain

    '''
    CC[x, x, 1, 8]: [fixed, moving, weight, radius]cd 
    -t SyN[0.25]: Syn transform with a gradient step of 0.25
    -r Gauss[3, 0]: sigma 0
    -I 30x50x20
    use - Histogram - Matching
    number - of - affine - iterations 10000x10000x10000x10000: 4 level image pyramid with 10000 iterations at each level
    MI - option 32x16000: 32 bins, 16000 samples
    '''

    reg = pe.Node(
        ants.Registration(
            dimension=3,
            output_transform_prefix="output_",
            #interpolation='BSpline',
            transforms=['Affine', 'SyN'],
            transform_parameters=[(2.0,), (0.25,)], #default values syn
            shrink_factors=[[8,4,2,1], [4, 2, 1]],
            smoothing_sigmas=[[3, 2, 1, 0], [2, 1, 0]], #None for Syn?
            sigma_units=['vox']*2,
            sampling_percentage=[0.05,None], #just use default?
            sampling_strategy=['Random', 'None'],
            number_of_iterations=[[10000,10000,10000,10000], [30,50,20]],
            metric=['MI', 'CC'],
            metric_weight=[1, 1],
            radius_or_number_of_bins=[(32), (8)],
            #winsorize_lower_quantile=0.05,
            #winsorize_upper_quantile=0.95,
            verbose=True,
            use_histogram_matching=[True, True]
        ),
        name='calc_registration')

    applytransforms = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='NearestNeighbor'),
        name='apply_warpfield')

    #Make warped atlas binary image
    #https://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.fsl/preprocess.html#bet
    binarize = pe.Node(
        fsl.UnaryMaths(operation='bin'),
        name='binarize')

    #apply binary warped atlas as mask to T2w
    #https://nipype.readthedocs.io/en/0.12.0/interfaces/generated/nipype.interfaces.fsl.maths.html#applymask
    applymask = pe.Node(
        fsl.ApplyMask(),
        name='apply_mask')

    wf = pe.Workflow(name='wf', base_dir=warped_dir)

    wf.connect(
        [(input_spec, reg, [('atlas_image', 'moving_image'),
                            ('subject_image_list', 'fixed_image')]), #create warp field to register atlas to subject
         (input_spec, applytransforms, [('atlas_image_brain', 'input_image'),
                                        ('subject_image', 'reference_image')]),
         (reg, applytransforms, [('forward_transforms', 'transforms')]) #apply warpfield to register atlas brain to subject
         ]
    )
    wf.connect(applytransforms, 'output_image', binarize, 'in_file') #turn warped atlas brain into binary image to use as mask
    wf.connect(binarize, 'out_file', applymask, 'mask_file')
    wf.connect(input_spec, 'subject_image', applymask, 'in_file')

    wf.config['execution']['parameterize_dirs'] = False

    wf.write_graph()
    output = wf.run(plugin='MultiProc', plugin_args={'n_procs' : n_jobs})

if __name__ == '__main__':
    main()
