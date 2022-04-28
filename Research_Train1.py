import argparse
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import museparation.waveunet.model.utils as model_utils
import museparation.waveunet.utils as utils
from museparation.waveunet.data.dataset import SeparationDataset
from museparation.waveunet.data.utils import crop_targets, random_amplify
from museparation.waveunet.test import evaluate, validate
from museparation.waveunet.model.waveunet import Waveunet
from museparation.scripts.get_musdb import get_musdbhq



def main(args):
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else [args.features*2**1 for i in range(args.levels)]

	target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, 
	 kernel_size=args.kernel_size, 
	 target_output_size=target_outputs, 
	 depth=args.depth, 
	 strides=args.strides,
	 conv_type=args.conv_type, 
	 res=args.res, 
	 separate=args.separate
	)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    musdb = get_musdbhq(args.dataset_dir)
    crop_func = partial(crop_targets, shapes=model.shapes)
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=augment_func)
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)


    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)


    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    optimizer = Adam(params=model.parameters(), lr=args.lr)
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for example_num, (x, targets) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    for k in list(targets.keys()):
                        targets[k] = targets[k].cuda()

                t = time.time()

                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs, avg_loss = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                writer.add_scalar("train_loss", avg_loss, state["step"])

                if example_num % args.example_freq == 0:
                    input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                    writer.add_audio("input", input_centre, state["step"], sample_rate=args.sr)

                    for inst in outputs.keys():
                        writer.add_audio(inst + "_pred", torch.mean(outputs[inst][0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio(inst + "_target", torch.mean(targets[inst][0], 0), state["step"], sample_rate=args.sr)

                pbar.update(1)

        # VALIDATE
        val_loss = validate(args, model, criterion, val_data)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state["step"])

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        # CHECKPOINT
        print("Saving model...")
        model_utils.save_model(model, optimizer, state, checkpoint_path)

        state["epochs"] += 1

    #### TESTING ####
    # Test loss
    print("TESTING")

    # Load best model based on validation loss
    state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
    test_loss = validate(args, model, criterion, test_data)
    print("TEST FINISHED: LOSS: " + str(test_loss))
    writer.add_scalar("test_loss", test_loss, state["step"])

    # Mir_eval metrics
    test_metrics = evaluate(args, musdb["test"], model, args.instruments)

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # Write most important metrics into Tensorboard log
    avg_SDRs = {inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SIRs = {inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
    for inst in args.instruments:
        writer.add_scalar("test_SDR_" + inst, avg_SDRs[inst], state["step"])
        writer.add_scalar("test_SIR_" + inst, avg_SIRs[inst], state["step"])
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

    writer.close()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--musdb_path", type=str, required=True)
	parser.add_argument("--hdf_dir", type=str, default="hdf")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs/waveunet')
	parser.add_argument("--cuda", action='store_true')
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"])
    parser.add_argument('--separate', type=int, default=1)


	#training hyperparams
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--loss', type=str, default="L1")

	#model hyperparams
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--output_size', type=float, default=2.0)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--strides', type=int, default=4)
    parser.add_argument('--levels', type=int, default=6)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--feature_growth', type=str, default="double")
    parser.add_argument('--conv_type', type=str, default="gn")
    parser.add_argument('--res', type=str, default="fixed")


	args = parser.parse_args()
	main(args)
