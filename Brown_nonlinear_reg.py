#!/usr/bin/env python3
# standard lib

'''
fixed atlas image: temp_nih_T2w_atl_float.nii.gz

moving images for each subject: T2w_acpc_dc.nii.gz

'''

import argparse
import os
from glob import glob
import numpy as np

# external libs
import nipype.pipeline.engine as pe
from nipype.interfaces import ants, utility


def main():
    parser = generate_parser()

    args = parser.parse_args()
    path = args.path
    njobs = args.njobs

    path = "/home/exacloud/lustre1/fnl_lab/projects/INFANT/GEN_INFANT/masking_test"
    fixed_image = "/home/exacloud/lustre1/fnl_lab/projects/INFANT/GEN_INFANT/masking_test/temp_nih_T2w_atl_float.nii.gz"

    randint = np.array([1000])
    warped_dir = os.path.join('./warped_dir', 'warped{}'.format(randint[0]))

    pattern = os.path.join(path, '*/T2w_acpc_dc.nii.gz')
    t2w_list = glob(pattern)

    register(warped_dir, fixed_image, t2w_list, n_jobs=njobs)

def generate_parser():
    parser = argparse.ArgumentParser(description='non-linear registration from Brown')

    parser.add_argument('path', help='path to images')
    parser.add_argument('--njobs', default=1, type=int, help='number of cpus to utilize')

    return parser

def register(warped_dir, atlas_image, moving_images, n_jobs):

    input_spec = pe.Node(
        utility.IdentityInterface(fields=['moving_image', 'fixed_image']),
        iterables=[('moving_image', moving_images)],
        synchronize=True,
        name='input_spec'
    )
    # set input_spec
    input_spec.inputs.moving_image = moving_images
    input_spec.inputs.fixed_image = atlas_image

    '''
    CC[x, x, 1, 8]: [fixed, moving, weight, radius]
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
            dimension=3),
        name='apply_warpfield')

    wf = pe.Workflow(name='wf', base_dir=warped_dir)
    wf.connect(
        [(input_spec, reg, [('fixed_image', 'fixed_image'),
                            ('moving_image', 'moving_image')]),
         (input_spec, applytransforms, [('moving_image', 'input_image'),
                                        ('fixed_image', 'reference_image')])
         ]
    )
    wf.connect(reg, 'forward_transforms', applytransforms, 'transforms')

    wf.config['execution']['parameterize_dirs'] = False

    wf.write_graph()
    output = wf.run(plugin='MultiProc', plugin_args={'n_procs' : n_jobs})

if __name__ == '__main__':
    main()