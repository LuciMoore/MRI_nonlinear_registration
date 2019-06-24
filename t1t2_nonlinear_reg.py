#!/usr/bin/env python3
# standard lib

'''
moving brain images for each subject: T2w_acpc_dc_brain.nii.gz
path = /home/exacloud/lustre1/fnl_lab/data/HCP/processed/BCP/BCP_NEO_ATROPOS_4/sub-116056/ses-3m/files/T1w/
'''

import argparse
import os
from glob import glob
import random

# external libs
import nipype.pipeline.engine as pe
from nipype.interfaces import ants, utility


def main():
    parser = generate_parser()
    args = parser.parse_args()
    folder = args.folder
    njobs = args.njobs

    #randint = '_nlreg'
    warped_dir = os.path.join('./nlreg_dir')

    #subject T1w and T2w images
    t1w = os.path.join(folder, 'T1w_acpc_dc_restore_brain.nii.gz')
    t2w = os.path.join(folder, 'T2w_acpc_dc_restore_brain.nii.gz')

    register(warped_dir, t1w, t2w, n_jobs=njobs)

def generate_parser():
    parser = argparse.ArgumentParser(description='non-linear registration from Brown')
    parser.add_argument('folder', help='path to images')
    parser.add_argument('--njobs', default=1, type=int, help='number of cpus to utilize')
    return parser

def register(warped_dir, subject_T1w, subject_T2w, n_jobs):

    input_spec = pe.Node(
        utility.IdentityInterface(fields=['subject_T1w', 'subject_T2w']),
        name='input_spec'
    )

    # set input_spec
    input_spec.inputs.subject_T1w = subject_T1w
    input_spec.inputs.subject_T2w = subject_T2w

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

    wf = pe.Workflow(name='wf', base_dir=warped_dir)

    wf.connect(
        [(input_spec, reg, [('subject_T1w', 'moving_image'),
                            ('subject_T2w', 'fixed_image')]), #create warp field to register atlas to subject
         (input_spec, applytransforms, [('subject_T1w', 'input_image'),
                                        ('subject_T2w', 'reference_image')]),
         (reg, applytransforms, [('forward_transforms', 'transforms')]) #apply warpfield to register atlas brain to subject
         ]
    )

    wf.config['execution']['parameterize_dirs'] = False

    wf.write_graph()
    output = wf.run(plugin='MultiProc', plugin_args={'n_procs' : n_jobs})

if __name__ == '__main__':
    main()