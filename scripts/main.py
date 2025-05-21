import pickle
from os.path import dirname, realpath
import sys
import git
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import vmramar.datasets.factory as dataset_factory
import vmramar.models.factory as model_factory
from vmramar.learn import train
import vmramar.transformers.factory as transformer_factory
import vmramar.visualize as visualize
import vmramar.utils.parsing as parsing
import warnings
import vmramar.learn.state_keeper as state
from vmramar.utils.get_dataset_stats import get_dataset_stats
import vmramar.utils.stats as stats
import pdb
import csv

#Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

if __name__ == '__main__':
    args = parsing.parse_args()
    if args.ignore_warnings:
        warnings.simplefilter('ignore')

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object
    args.commit = commit.hexsha
    print("vmramar main running from commit: \n\n{}\n{}author: {}, date: {}".format(
        commit.hexsha, commit.message, commit.author, commit.committed_date))

    if args.get_dataset_stats:
        print("\nComputing image mean and std...")
        args.img_mean, args.img_std = get_dataset_stats(args)
        print('Mean: {}'.format(args.img_mean))
        print('Std: {}'.format(args.img_std))

    print("\nLoading data-augmentation scheme...")
    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args)
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args)
    # Load dataset and add dataset specific information to args
    print("\nLoading data...")
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, transformers, test_transformers)
    # Load model and add model specific information to args
    if args.snapshot is None:
        # Load image encoder from snapshot if provided
        if hasattr(args, 'img_encoder_snapshot') and args.img_encoder_snapshot is not None:
            args.image_encoder = model_factory.load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
            if args.freeze_image_encoder:
                for param in args.image_encoder.parameters():
                    param.requires_grad = False
        
        # Initialize VMRNN with specified parameters
        vmrnn_args = args.vmrnn_params if hasattr(args, 'vmrnn_params') else {}
        args.vmrnn = model_factory.get_model('vmrnn', vmrnn_args)
        
        # Initialize asymmetry modules if enabled
        if hasattr(args, 'asymmetry_params') and args.asymmetry_params.get('use_asymmetry', False):
            args.sad = model_factory.get_model('sad', args.asymmetry_params)
            args.lat = model_factory.get_model('lat', args.asymmetry_params)
        
        # Get the full VMRA-MaR model
        model = model_factory.get_model(args)
    else:
        model = model_factory.load_model(args.snapshot, args)

    print(model)
    # Load run parameters if resuming that run.
    args.model_path = state.get_model_path(args)
    print('Trained model will be saved to [%s]' % args.model_path)
    if args.resume:
        try:
            state_keeper = state.StateKeeper(args)
            model, optimizer_state, epoch, lr, epoch_stats = state_keeper.load()
            args.optimizer_state = optimizer_state
            args.current_epoch = epoch
            args.lr = lr
            args.epoch_stats = epoch_stats
        except:
            print("\n Error loading previous state. \n Starting run from scratch.")
            args.optimizer_state = None
            args.current_epoch = None
            args.lr = None
            args.epoch_stats = None
    else:
        print("\n Starting run from scratch.")

    # Print parameters
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'patient_to_partition_dict', 'path_to_hidden_dict', 
                       'exam_to_year_dict', 'exam_to_device_dict', 'image_encoder', 'vmrnn', 'sad', 'lat']:
            print("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    print()
    if args.train:
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        args.epoch_stats = epoch_stats

        if args.plot_losses:
            visualize.viz_utils.plot_losses(epoch_stats)
        print("Save train/dev results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    print()
    if args.dev:
        print("-------------\nDev")
        args.dev_stats = train.compute_threshold_and_dev_stats(dev_data, model, args)
        print("Save dev results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if args.test:
        print("-------------\nTest")
        args.test_stats = train.eval_model(test_data, model, args)
        print("Save test results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if (args.dev or args.test) and args.prediction_save_path is not None:
        exams, probs = [], []
        if args.dev:
            exams.extend(args.dev_stats['exams'])
            probs.extend(args.dev_stats['probs'])
        if args.test:
            exams.extend(args.test_stats['exams'])
            probs.extend(args.test_stats['probs'])
        legend = ['patient_exam_id']
        if args.callibrator_snapshot is not None:
            callibrator = pickle.load(open(args.callibrator_snapshot, 'rb'))
        for i in range(args.max_followup):
            legend.append("{}_year_risk".format(i+1))
        export = {}
        with open(args.prediction_save_path, 'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=legend)
            writer.writeheader()
            for exam, arr in zip(exams, probs):
                export['patient_exam_id'] = exam
                for i in range(args.max_followup):
                    key = "{}_year_risk".format(i+1)
                    raw_val = arr[i]
                    if args.callibrator_snapshot is not None:
                        val = callibrator[i].predict_proba([[raw_val]])[0,1]
                    else:
                        val = raw_val
                    export[key] = val
                writer.writerow(export)
        print("Exported predictions to {}".format(args.prediction_save_path))


