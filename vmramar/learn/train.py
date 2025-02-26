import os
import math
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import vmramar.models.factory as model_factory
import vmramar.learn.state_keeper as state
import vmramar.utils.stats as stats
from vmramar.learn.utils import cluster_results_by_exam, ignore_None_collate

def train_model(train_data, dev_data, model, args):
    """
    Train VMRA-MaR model with asymmetry-aware risk prediction
    """
    # Initialize data loaders with appropriate batch size and collate function
    train_data_loader, dev_data_loader = get_train_and_dev_dataset_loaders(
        train_data, dev_data, args, ignore_None_collate
    )

    # Initialize models dictionary with main model and optimizers
    models = {'model': model}
    optimizers = {
        'model': torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    }

    # Initialize state keeper for model checkpointing
    state_keeper = state.StateKeeper(args)
    
    # Set up metrics tracking
    epoch_stats = init_metrics_dictionary(args)
    tuning_key = f'dev_{args.tuning_metric}'
    no_tuning_on_dev = args.num_epochs == 1 or args.cross_val_id is not None
    
    num_epoch_sans_improvement = 0
    num_epoch_since_reducing_lr = 0

    # Training loop
    for epoch in range(args.num_epochs):
        print('Starting epoch {}'.format(epoch + 1))
        
        for mode, data_loader in [('Train', train_data_loader), ('Dev', dev_data_loader)]:
            train_model = mode == 'Train'
            key_prefix = mode.lower()
            
            # Run epoch
            loss, golds, preds, probs, exams, reg_loss, censor_times, adv_loss = run_epoch(
                data_loader,
                train_model=train_model,
                truncate_epoch=True,
                models=models,
                optimizers=optimizers,
                args=args
            )

            # Compute metrics
            log_statement, epoch_stats = compute_eval_metrics(
                args, loss, golds, preds, probs, exams, 
                reg_loss, censor_times, adv_loss, epoch_stats, key_prefix
            )

            # For dev set, compute additional metrics like FNR/TNR/TPR
            if mode == 'Dev' and 'mammo_1year' in args.dataset:
                dev_human_preds = get_human_preds(exams, dev_data.metadata_json)
                threshold, _ = stats.get_thresholds_interval(
                    probs, golds, dev_human_preds,
                    rebalance_eval_cancers=args.rebalance_eval_cancers, 
                    num_resamples=NUM_RESAMPLES_DURING_TRAIN
                )
                print(' Dev Threshold: {:.8f} '.format(threshold))
                
                # Calculate rates
                (fnr, _), (tpr, _), (tnr, _) = stats.get_rates_intervals(
                    probs, golds, threshold,
                    rebalance_eval_cancers=args.rebalance_eval_cancers, 
                    num_resamples=NUM_RESAMPLES_DURING_TRAIN
                )
                
                # Update stats
                epoch_stats[f'{key_prefix}_fnr'].append(fnr)
                epoch_stats[f'{key_prefix}_tnr'].append(tnr)
                epoch_stats[f'{key_prefix}_tpr'].append(tpr)
                log_statement = f"{log_statement} fnr: {fnr:.3f} tnr: {tnr:.3f} tpr: {tpr:.3f}"

            print(log_statement)

        # Model saving logic
        best_func, arg_best = (min, np.argmin) if tuning_key == 'dev_loss' else (max, np.argmax)
        improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]
        
        if improved or no_tuning_on_dev:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            epoch_stats['best_epoch'] = arg_best(epoch_stats[tuning_key])
            state_keeper.save(models, optimizers, epoch, args.lr, epoch_stats)

        # Learning rate adjustment logic
        num_epoch_since_reducing_lr += 1
        if improved:
            num_epoch_sans_improvement = 0
        else:
            num_epoch_sans_improvement += 1

        print('---- Best Dev {} is {} at epoch {}'.format(
            args.tuning_metric,
            epoch_stats[tuning_key][epoch_stats['best_epoch']],
            epoch_stats['best_epoch'] + 1
        ))

        # Learning rate reduction logic
        if (num_epoch_sans_improvement >= args.patience or 
            (no_tuning_on_dev and num_epoch_since_reducing_lr >= args.lr_reduction_interval)):
            
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            num_epoch_since_reducing_lr = 0
            
            if not args.turn_off_model_reset:
                models, optimizer_states, _, _, _ = state_keeper.load()
                
                # Reset optimizers
                for name in optimizers:
                    optimizer = optimizers[name]
                    state_dict = optimizer_states[name]
                    optimizers[name] = state_keeper.load_optimizer(optimizer, state_dict)
            
            # Reduce learning rate
            for name in optimizers:
                optimizer = optimizers[name]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_rate
                    print('Reducing learning rate to {}'.format(param_group['lr']))

            if args.lr_decay_rate == 0:
                print('Learning rate is 0, stopping training...')
                break

    return models, optimizers, epoch_stats