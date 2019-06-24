#!/usr/bin/env python3
# standard lib
'''

Template:
/home/exacloud/lustre1/fnl_lab/code/internal/pipelines/HCP_release_20161027_Infant_v2.0/global/templates/babyCouncil
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
    subject_T1w_folder = args.subject_T1w_folder
    #subject_T2w = args.subject_T2w
    jlf_folder = args.joint_fusion_folder
    subid = args.subject_id
    njobs = args.njobs

    pattern = os.path.join(jlf_folder, 'Template*')
    template_list = glob(pattern)

    atlas_images = []
    for i in template_list:
        atlas_images.append(os.path.join(i, "T1w_brain.nii.gz"))

    atlas_segmentations = []
    for i in template_list:
        atlas_segmentations.append(os.path.join(i, "Segmentation.nii.gz"))

    #randint = random.randint(1,100)
    warped_dir = os.path.join('./jlf_2chreg_dir', 'jlf{}'.format(subid))

    #subject T1w brain image
    subject_T1w = os.path.join(subject_T1w_folder, 'T1w_acpc_dc_restore_brain.nii.gz')
    subject_T2w = os.path.join(subject_T1w_folder, 'T2w_acpc_dc_restore_brain.nii.gz')

    #make list of subject T1w and T2w
    subject_Tws = [subject_T1w, subject_T2w]

    register(warped_dir, subject_Tws, atlas_images, atlas_segmentations, n_jobs=njobs)

def generate_parser():
    parser = argparse.ArgumentParser(description='non-linear registration from Brown')

    parser.add_argument('subject_T1w_folder', help='path to subject T1w restored brain')
    #parser.add_argument('subject_T2w', help='path to subject T2w restored brain')
    parser.add_argument('joint_fusion_folder', help='path to joint label fusion atlas directory')
    parser.add_argument('subject_id', help='subject id')
    parser.add_argument('--njobs', default=1, type=int, help='number of cpus to utilize')

    return parser

def register(warped_dir, subject_Tws, atlas_images, atlas_segmentations, n_jobs):

    #create list for subject T1w and T2w because Nipype requires inputs to be in list format specifically fr JLF node
    sub_T1w_list = []
    sub_T1w_list.append(subject_Tws[0])

    sub_T2w_list = []
    sub_T2w_list.append(subject_Tws[1])

    input_spec = pe.Node(
        utility.IdentityInterface(fields=['subject_Txw', 'subject_Txw_list', 'subject_dual_Tws', 'atlas_image', 'atlas_segmentation']),
        iterables=[('atlas_image', atlas_images), ('atlas_segmentation', atlas_segmentations)],
        synchronize=True,
        name='input_spec'
    )
    # set input_spec
    input_spec.inputs.subject_Txw = subject_Tws[0] #using T1w here
    input_spec.inputs.subject_Txw_list = sub_T1w_list
    input_spec.inputs.subject_dual_Tws = subject_Tws

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
            sampling_percentage=[0.05, None], #just use default?
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

    applytransforms_atlas = pe.Node(
        ants.ApplyTransforms(
            interpolation='BSpline',
            dimension=3,
        ),
        name='apply_warpfield_atlas')

    applytransforms_segs = pe.Node(
        ants.ApplyTransforms(
            interpolation='NearestNeighbor',
            dimension=3
        ),
        name='apply_warpfield_segs')

    jointlabelfusion = pe.JoinNode(
        ants.AntsJointFusion(
            dimension=3,
            alpha=0.1,
            beta=2.0,
            patch_radius=[2, 2, 2],
            search_radius=[3, 3, 3],
            out_label_fusion = 'out_label_fusion.nii.gz',
        ),
        joinsource='input_spec',
        joinfield=['atlas_image', 'atlas_segmentation_image'],
        name='joint_label_fusion'
    )

    wf = pe.Workflow(name='wf', base_dir=warped_dir)

    wf.connect(input_spec, 'subject_dual_Tws', reg, 'fixed_image')
    wf.connect(input_spec, 'atlas_image', reg, 'moving_image')

    wf.connect(reg, 'forward_transforms', applytransforms_atlas, 'transforms')
    wf.connect(input_spec, 'atlas_image', applytransforms_atlas, 'input_image')
    wf.connect(input_spec, 'subject_Txw', applytransforms_atlas, 'reference_image')

    wf.connect(reg, 'forward_transforms', applytransforms_segs, 'transforms')
    wf.connect(input_spec, 'atlas_segmentation', applytransforms_segs, 'input_image')
    wf.connect(input_spec, 'subject_Txw', applytransforms_segs, 'reference_image')

    wf.connect(input_spec, 'subject_Txw_list', jointlabelfusion, 'target_image')
    wf.connect(applytransforms_atlas, 'output_image', jointlabelfusion, 'atlas_image')
    wf.connect(applytransforms_segs, 'output_image', jointlabelfusion, 'atlas_segmentation_image')

    wf.config['execution']['parameterize_dirs'] = False
    #wf.config['execution']['remove_unnecessary_outputs'] = False
    #wf.config['execution']['stop_on_first_crash'] = True

    #create workflow graph
    wf.write_graph()

    #Nipype plugins specify how workflow should be executed
    output = wf.run(plugin='MultiProc', plugin_args={'n_procs' : n_jobs})

if __name__ == '__main__':
    main()