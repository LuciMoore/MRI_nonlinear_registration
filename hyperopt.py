#!/usr/bin/env python3
# standard lib

'''
nifti location for testing: /home/groups/brainmri/infant/EXITO/unprocessed/niftis/*
test registration of T1w and T2w (.nii.gz files)

'''

import argparse
import os
import shutil

# external libs
import numpy as np
import skopt
import nipype.pipeline.engine as pe
from nipype.interfaces import ants, utility
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

def main():
    parser = generate_parser()

    args = parser.parse_args()
    path = args.path
    njobs = args.njobs
    ncalls = args.ncalls

    paired_image_list = get_images(path)
    print(paired_image_list)

    # truncate
    paired_image_list = paired_image_list[:20]

    optimize(os.path.join(path, 'optimize'), paired_image_list=paired_image_list, n_jobs=njobs, n_calls=ncalls)


def generate_parser():
    parser = argparse.ArgumentParser(description='meta-optimizer for registration')

    parser.add_argument('path', help='path to T1w and T2w images')
    parser.add_argument('--njobs', default=1, type=int, help='number of cpus to utilize')
    parser.add_argument('--ncalls', default=11, type=int, help='number of calls to fitness function')

    return parser


def get_images(path):

    assert os.path.isdir(path), '%s is not a directory!' % path
    image_pairs = []
    for root, directories, files in filter(lambda x: os.path.basename(x[0]) == 'anat',
                                           os.walk(path)):
        T1ws = [f for f in files if 'T1w.nii.gz' in f]
        T2ws = [f for f in files if 'T2w.nii.gz' in f]
        if len(T1ws) and len(T2ws):
            image_pairs.append((os.path.join(root, T1ws[0]),
                                os.path.join(root, T2ws[0])))
        else:
            print('%s does not have a pair of T1w, T2w' % root)

    return image_pairs

def optimize(wd='./optimize', paired_image_list=[], n_jobs=1, n_calls=10):
    if not os.path.exists(wd):
        os.makedirs(wd)
    os.chdir(wd)
        # make new directory for registered output images
    best_images = os.path.join(wd, 'best_images')
    # warped_dir = os.path.join(wd, 'warped_images')

    # initialize best accuracy score
    global best_fitness
    best_fitness = 0.0

    dim_metric = Categorical(categories=['CC', 'MI'], name='metric')
    dim_radius = Integer(2, 8, name='radius')
    dim_n_subsamples = Integer(2, 8, name='n_subsamples')
    dim_n_bins = Integer(16, 64, name ='mi_bins')
    dim_histogram = Categorical(categories=[True, False], name='histomatching')

    x0=("MI",2,2,16,False)

    dimensions = [
        dim_metric,
        dim_radius,
        dim_n_subsamples,
        dim_n_bins,
        dim_histogram
    ]

    @use_named_args(dimensions=dimensions)
    def fitness(metric, radius, n_subsamples, mi_bins, histomatching):
        similarities = np.zeros((len(paired_image_list),))
        #randint = np.random.randint(100000, 999999, +1)
        randint = np.array([1000])
        warped_dir = os.path.join(wd, 'warped{}'.format(randint[0]))
        #for i, (t1w, t2w) in enumerate(paired_image_list):
        t1w, t2w = zip(*paired_image_list)
        similarities = register(warped_dir, t1w, t2w, metric, radius, n_subsamples, mi_bins, histomatching, n_jobs)
        fitness = np.mean(similarities)

        global best_fitness
        if fitness > best_fitness:
            # copy folder of output transforms, maybe with values of metric, radius, etc.
            print('new best fitness score: %s\nprevious: %s' % (fitness, best_fitness))
            if os.path.exists(best_images):
                shutil.rmtree(best_images)
            shutil.copytree(warped_dir, best_images)
            best_fitness = fitness
        shutil.rmtree(warped_dir)

        return -fitness

    def register(warped_dir, fixed_images, moving_images, metric, radius, n_subsamples, mi_bins, histomatching, ncpus=1):
        shrink_factors = [2 ** i for i in range(n_subsamples - 1, -1, -1)]
        smoothing_sigmas = list(range(n_subsamples - 1, -1, -1))
        convergence = [100 + i for i in shrink_factors]

        #save all warped output images to a folder and save them w/ subject id
        #file_basename = os.path.basename(fixed_image)
        #split_text1 = os.path.splitext(file_basename)
        #split_text2 = os.path.splitext(split_text1[0])
        #sub_id = split_text2[0]

        inputs = pe.Node(
            utility.IdentityInterface(fields=['moving_image', 'fixed_image']),
            iterables=[('moving_image', moving_images), ('fixed_image', fixed_images)],
            synchronize=True,
            name='inputs'
        )

        reg = pe.Node(
            ants.Registration(
            dimension=3,
            output_transform_prefix="output_",
            interpolation='BSpline',
            transforms=['Rigid'],
            transform_parameters=[(0.1,)],  # gradient step/learning rate
            shrink_factors=[shrink_factors],
            smoothing_sigmas=[smoothing_sigmas],
            sigma_units=['vox'],
            sampling_percentage=[0.05],
            sampling_strategy=['Random'],
            number_of_iterations=[convergence],
            metric=[metric],
            radius_or_number_of_bins=[radius],
            winsorize_lower_quantile=0.05,
            winsorize_upper_quantile=0.95,  # clips high and low intensity data
            verbose=True,
            use_histogram_matching=[histomatching]
        ),
        name='calc_registration')
        if metric == 'MI':
            reg.inputs.radius_or_number_of_bins = [mi_bins]

        applytransforms = pe.Node(
            ants.ApplyTransforms(
                dimension=3,
                interpolation='BSpline'),
                name='apply_warpfield'
        )

        # Image similarity
        sim = pe.Node(ants.MeasureImageSimilarity(), name='calc_similarity')
        # assign attribute values to sim:
        sim.inputs.dimension = 3
        sim.inputs.metric = 'MI'
        sim.inputs.radius_or_number_of_bins = 32
        sim.inputs.sampling_percentage = 1.0

        merge = pe.JoinNode(
            utility.Merge(1, ravel_inputs=True),
            joinsource='inputs',
            joinfield=['in1'],
            name='merge'
        )

        wf = pe.Workflow(name='wf', base_dir=warped_dir)
        wf.connect(
            [(inputs, reg, [('fixed_image', 'fixed_image'), ('moving_image', 'moving_image')]),
             (inputs, applytransforms, [('fixed_image', 'reference_image'), ('moving_image', 'input_image')]),
             (inputs, sim, [('fixed_image', 'fixed_image')])
             ]
        )
        wf.connect(reg, 'forward_transforms', applytransforms, 'transforms')
        wf.connect(applytransforms, 'output_image', sim, 'moving_image')
        wf.connect(sim, 'similarity', merge, 'in1')
        wf.config['execution']['parameterize_dirs'] = False

        wf.write_graph()
        output = wf.run(plugin='MultiProc', plugin_args={'n_procs' : ncpus})

        out_nodes = [n.name for n in output.nodes]
        similarities = list(output.nodes)[(out_nodes.index('merge'))].result.outputs.out
        # similarities = list(output.nodes)[0].result.outputs.out
        return similarities

    checkpoint_callback = CheckpointSaver('./result.pkl', store_objective=False)

    search_result = skopt.gp_minimize(
        func=fitness, dimensions=dimensions, acq_func='EI', n_calls=n_calls, n_jobs=1, callback=[checkpoint_callback],
        x0=x0
    )

    print(search_result.x)

#execute:
if __name__ == '__main__':
    main()
