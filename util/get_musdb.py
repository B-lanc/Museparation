import argparse
import os
import glob
import numpy as np


def get_musdb(root_path):
	#Not yet implemented
	pass

def get_musdbhq(root_path):
	"""
	Retrieve the path of the audio files (absolute) from MUSDBHQ dataset
	:param root_path: the root directory of MUSDB18HQ (containing train and test directories)
	:return: list containing 2 lists of dictionaries. [[{"bass": path_to_bass, "drums": path_to_drums, ...}, {}, {}, ...], []]
	"""
	subsets = list()

	for subset in ["train", "test"]:
		tracks = glob.glob(os.path.join(root_path, subset, "*"))
		samples = list()

		for track_folder in sorted(tracks):
			track = dict()
			for stem in ["bass", "drums", "mixture", "other", "vocals"]:
				audio_path = os.path.join(track_folder, stem + ".wav")
				track[stem] = audio_path
			samples.append(track)

		subsets.append(samples)
	return subsets


def separate_into_three(dataset):
	"""
	Separate the dataset into train, validation, and test sets
	:param dataset: return value of get_musdb(hq)
	:return: dictionary {"train": [{"bass": path_to_bass, ...}, {}, {}, ...], "val": [], "test": []}
	"""
	########## possibly add into global config file as hyperparams
	SEED = 0
	VALIDATION_SIZE = 0.2
	##########

	train_validation_list = dataset[0]
	test_list = dataset[1]

	middle = int(len(train_validation_list) * VALIDATION_SIZE // 1)
	
	np.random.seed(SEED)
	np.random.shuffle(train_validationa_list)
	val_list   = train_validation_list[:middle]
	train_list = train_validation_list[middle:]
	
	return {"train" : train_list, "val" : val_list, "test" : test_list}

def get_folds(root_path, version="HQ"):
	"""
	Choose which version of get_musdb function to be called, and separate into train, validation, and test lists
	:param root_path: the root directory of MUSDB dataset
	:param version: the version of MUSDB dataset (HQ for HQ, everything else for normal version)
	:return: dictionary {"train": [{"bass": path_to_bass, ...}, {}, {}, ...], "val": [], "test": []}
	"""
	if version == "HQ":
		dataset = get_musdbhq(root_path)
	else:
		dataset = get_musdb(root_path)
	
	return separate_into_three(dataset)

if __name__ == "__main__":
	#Do some argument parsing and call get_folds function
	pass
