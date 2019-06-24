#!/usr/bin/env python3
# standard lib


import argparse
import os
from glob import glob

# external libs
import nipype.pipeline.engine as pe
from nipype.interfaces import ants, utility
from nipype.interfaces.image import Rescale


def main():
    parser = generate_parser()

    args = parser.parse_args()
    subject_T1w_folder = args.subject_T1w_folder
    jlf_folder = args.joint_fusion_folder
    njobs = args.njobs

    pattern = os.path.join(jlf_folder, 'Template*')
    template_list = glob(pattern)

    atlas_images = []
    for i in template_list:
        atlas_images.append(os.path.join(i, "T1w_brain.nii.gz"))

    atlas_segmentations = []
    for i in template_list:
        atlas_segmentations.append(os.path.join(i, "Segmentation.nii.gz"))

    randint = random.randint(1,100)
    warped_dir = os.path.join('./invert_dir', 'jlf{}'.format(randint))

    #subject T1w brain image
    subject_T1w = os.path.join(subject_T1w_folder, 'T1w_acpc_dc_restore_brain.nii.gz')
    subject_T2w = os.path.join(subject_T1w_folder, 'T2w_acpc_dc_restore_brain.nii.gz')

    #make list of subject T1w and T2w
    subject_Tws = [subject_T1w, subject_T2w]

    register(warped_dir, subject_Tws, atlas_images, atlas_segmentations, n_jobs=njobs)

def generate_parser():
    parser = argparse.ArgumentParser(description='non-linear registration from Brown')

    parser.add_argument('subject_T1w_folder', help='path to subject T1w restored brain')
    parser.add_argument('joint_fusion_folder', help='path to joint label fusion atlas directory')
    parser.add_argument('--njobs', default=1, type=int, help='number of cpus to utilize')

    return parser

def register(warped_dir, subject_Tws, atlas_images, atlas_segmentations, n_jobs):

    #create list for subject T1w and T2w because Nipype requires inputs to be in list format specifically fr JLF node
    sub_T1w_list = []
    sub_T1w_list.append(subject_Tws[0])

    sub_T2w_list = []
    sub_T2w_list.append(subject_Tws[1])

    atlas_forinvert = atlas_images[0] #just use

    def main():
        # sub_T2w_inverted =

        subject_T2w = '/home/exacloud/lustre1/fnl_lab/data/HCP/processed/BCP/BCP_NEO_ATROPOS_4/sub-375518/ses-1m/files/T1w/T1w_acpc_dc_restore_brain.nii.gz'

        inv(subject_T2w)

    def inv(subject_T2w):
        fsl.maths.MathsCommand(in_file=subject_T2w, args="-recip", out_file="T1w_acpc_dc_restore_brain_inverse.nii.gz")

    if __name__ == '__main__':
        main()

    input_spec = pe.Node(
        utility.IdentityInterface(fields=['subject_Txw', 'subject_Txw_list', 'subject_dual_Tws', 'atlas_image', 'atlas_segmentation', 'atlas_forinvert']),
        #iterables=[('atlas_image', atlas_images), ('atlas_segmentation', atlas_segmentations)],
        #synchronize=True,
        name='input_spec'
    )
    # set input_spec
    input_spec.inputs.subject_Txw = subject_Tws[1] #using T2w here
    input_spec.inputs.subject_Txw_list = sub_T2w_list
    input_spec.inputs.subject_dual_Tws = subject_Tws
    input_spec.inputs.atlas_forinvert = atlas_forinvert


    invert = pe.Node(Rescale(invert=True,
                             percentile = 1.), name='invert')


    wf = pe.Workflow(name='wf', base_dir=warped_dir)

    wf.connect(input_spec, "subject_Txw", invert, "in_file")
    wf.connect(input_spec, "atlas_forinvert", invert, "ref_file") #should I use the subject or atlas T1w here?


    wf.config['execution']['parameterize_dirs'] = False

    #create workflow graph
    wf.write_graph()

    #Nipype plugins specify how workflow should be executed
    output = wf.run(plugin='MultiProc', plugin_args={'n_procs' : n_jobs})

if __name__ == '__main__':
    main()