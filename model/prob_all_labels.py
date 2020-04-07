"""Add other labels to prob files."""

import argparse
import csv
import glob
import numpy as np
import os
import tensorflow as tf
import utils

CSV_SUFFIX = '*.csv'

def read_prob_file(prob_filename, col_label, col_prob, frame_count_move_back):
    frames = []
    labels = []
    probs = []
    with open(prob_filename) as prob_file:
        for row in csv.reader(prob_file, delimiter=','):
            frames.append(int(row[0]) - frame_count_move_back)
            labels.append(row[col_label])
            probs.append(row[col_prob])
    return frames, labels, probs

def read_eval_file(eval_filename, start_frame):
    frames_eval = []
    labels_1 = []
    labels_2 = []
    labels_3 = []
    labels_4 = []
    p_ids = []
    with open(eval_filename) as evalfile:
        reader = csv.reader(evalfile, delimiter=',')
        next(reader)
        for row in reader:
            frame_eval = int(row[1])
            if frame_eval < start_frame:
                continue
            frames_eval.append(frame_eval)
            p_ids.append(row[0])

            if row[16] == 'Idle':
                label_1 = '0'
            elif row[16] == 'Intake':
                label_1 = '1'
            else:
                raise ValueError('Unxpected value in column label 2', row[17], eval_filename, frame_eval) 
            labels_1.append(label_1)

            #if row[17] == 'Idle':
            #    label_2 = 0
            #elif row[17] == 'Eat':
            #    label_2 = 1
            #elif row[17] == 'Drink':
            #    label_2 = 2
            #elif row[17] == 'Lick':
            #    label_2 = 3
            #else:
            #    raise ValueError('Unxpected value in column label 2', row[17], eval_filename, frame_eval) 
            labels_2.append(row[17])

            #if row[18] == 'Idle':
            #    label_3 = 0
            #elif row[18] == 'Right':
            #    label_3 = 1
            #elif row[18] == 'Left':
            #    label_3 = 2
            #elif row[18] == 'Both':
            #    label_3 = 3
            #else:
            #    raise ValueError('Unxpected value in column label 3', row[18], eval_filename, frame_eval) 
            labels_3.append(row[18])

            #if row[19] == 'Idle':
            #    label_4 = 0
            #elif row[19] == 'Spoon':
            #    label_4 = 1
            #elif row[19] == 'Fork':
            #    label_4 = 2
            #elif row[19] == 'Cup':
            #    label_4 = 3
            #elif row[19] == 'Hand':
            #    label_4 = 4
            #else:
            #    raise ValueError('Unxpected value in column label 4', row[19], eval_filename, frame_eval) 
            labels_4.append(row[19])
    return frames_eval, labels_2, labels_3, labels_4, labels_1, p_ids

def add_other_labels(prob_dir, col_label, col_prob, eval_dir, prob_new_dir, move_original_frames_six_back = True):
    """Add other labels to prob files"""
    prob_filenames = glob.glob(os.path.join(prob_dir, CSV_SUFFIX))
    eval_filenames = glob.glob(os.path.join(eval_dir, CSV_SUFFIX))
    utils.create_dir_if_required(prob_new_dir)
    assert prob_filenames, "No prob files were found"
    for prob_filename in prob_filenames:
        prob_new_filename = os.path.join(prob_new_dir, utils.get_file_name_from_path(prob_filename))
        if utils.is_file(prob_new_filename):
            continue
        eval_filename = next(name for name in eval_filenames if utils.get_file_name_from_path(prob_filename, True) in utils.get_file_name_from_path(name, True))
        assert utils.is_file(eval_filename), "eval file was not found for prob file {}".format(prob_filename)
        frames, labels, probs = read_prob_file(prob_filename, col_label, col_prob, 6 if move_original_frames_six_back else 0)
        frames_eval, labels_2, labels_3, labels_4, labels_1, p_ids = read_eval_file(eval_filename, frames[0])

        assert frames_eval[len(frames)-1] == frames[-1] , 'prob file {0} and eval file {1} are not consistant.'.format(prob_filename, eval_filename)
        with open(prob_new_filename, 'w') as prob_new_file:
            for i in range(0, len(frames)):
                assert frames[i]==frames_eval[i], "frame id {0} and eval frame id {1} at index {2} are not consistent for file {3}".format(str(frames[i]),str(frames_eval[i]),str(i),utils.get_file_name_from_path(prob_filename))
                assert labels[i]==labels_1[i], "label {0} and label1 {1} at index {2} are not consistent for file {3}".format(str(labels[i]),str(labels_1[i]),str(i),utils.get_file_name_from_path(prob_filename))
                prob_new_file.write('{0},{1},{2},{3},{4},{5},{6}'.format(frames[i],labels[i],probs[i],labels_2[i],labels_3[i],labels_4[i],p_ids[i]))
                prob_new_file.write('\n')


def main(args=None):
    col_label = 1
    col_prob = 2
    add_other_labels(args.prob_dir, col_label, col_prob, args.eval_dir, args.prob_new_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add other labels to prob files')
    parser.add_argument('--prob_dir', type=str, default=r'C:\H\PhD\ORIBA\Model\F1\more\est.d4ks1357d2tl.valid.cl.b256.93.64_std_uni_no_smo.fixed_input\best_checkpoints\25000\prob_test', nargs='?', help='Directory that contains prob data.')
    parser.add_argument('--prob_new_dir', type=str, default=r'C:\H\PhD\ORIBA\Model\F1\more\est.d4ks1357d2tl.valid.cl.b256.93.64_std_uni_no_smo.fixed_input\best_checkpoints\25000\prob_test_all_labels', nargs='?', help='Directory that contains prob data.')
    parser.add_argument('--eval_dir', type=str, default=r'C:\H\PhD\ORIBA\Model\FileGen\OREBA\64_std_uni_no_smo', nargs='?', help='Directory that contains eval data.')
    args = parser.parse_args()
    main(args)
