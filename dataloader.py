"""
This file contains the dataloader for the RHM dataset. The dataloader loads the video frames from the RHM dataset
and performs preprocessing on them. The preprocessing includes cropping, normalizing, and converting the frames to
tensor. The dataloader also performs data augmentation on the training set by randomly flipping the frames

The RHM dataset is a multi-view dataset. The dataset contains 4 views: FrontView, BackView, OmniView, and RobotView.
The dataset also contains 8 different frame status: NormalFrame, MotionAggregation, FrameVariationMapper,
DifferentialMotionTrajectory, Normal, Subtract, OpticalFlow, and MotionHistoryImages. The dataset contains 14
different classes.

The dataloader takes in the following parameters:
view: View type.
view_status: Frame status for view.
split: Determines which list file to read from ('train', 'val', or 'test').
clip_len: Number of frames per clip.


The dataloader returns the following:
buffer1: The video frames from view
labels: The label for the video


Author: Mohammad Hossein Bamorovat Abadi
Email: m.bamorovvat@gmail.com

License: GNU General Public License (GPL) v3.0
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

# change this to your own path
Dataset_Path = '/home/abbas/RHM_full'

# Debug mode for printing out information. Set to False if you don't want to debug.
Debug = False


class VideoDataset(Dataset):
    """ RHM Dataset Loader
    Args:
        view (str): First view type.
        view_status (str): Frame status for view.
        split (str): Determines which list file to read from ('train', 'val', or 'test').
        clip_len (int): Number of frames per clip.
        preprocess (bool): If preprocessing is to be done.
    """

    def __init__(self, view='OmniView', view_status='Normal', split='train', clip_len=16, preprocess=False):

        print('Initializing dataloader...')

        self.preprocess = preprocess
        self.dataset_path = Dataset_Path
        self.split = split
        self.clip_len = clip_len
        self.view = view
        self.FrameStatus = view_status
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if Debug:
            print('Frame Height= ', self.resize_height)
            print('Frame Width= ', self.resize_width)
            print('Crop Size= ', self.crop_size)

        self.fnames_1, labels = self.load_frame_paths()

        if Debug:
            print('fnames_1= ', self.fnames_1.__len__())
            print('labels= ', labels)

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def load_frame_paths(self):
        if Debug:
            print('Loading frame paths...')
        fnames_1, labels = [], []
        list_file = os.path.join(self.dataset_path, f'{self.split}list.txt')

        if Debug:
            print('list_file= ', list_file)

        with open(list_file, 'r') as file:
            for line in file:
                video_id = line.strip()
                video_number = video_id.split('_')[-1]

                if Debug:
                    print('video_number= ', video_number)

                label = video_id.split('_')[0]  # Assuming the label is the part before the '_'
                video_path_1 = os.path.join(self.dataset_path, self.view + '_ExtractedFrames',
                                            self.view + '_' + self.FrameStatus, label,
                                            label + '_' + self.view + '_' + video_number)

                if Debug:
                    print('video_path_1= ', video_path_1)

                if os.path.isdir(video_path_1):
                    fnames_1.append(video_path_1)
                    labels.append(label)

        return fnames_1, labels

    def __len__(self):
        return len(self.fnames_1)

    # get the item from the dataset
    def __getitem__(self, index):

        if Debug:
            print('Getting item...')
            print('index: ', index)
            print('view Path: ', self.fnames_1[index])
        # Loading and preprocessing.
        buffer1= self.load_frames(self.fnames_1[index])
        buffer1 = self.crop(buffer1, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer1 = self.randomflip(buffer1)

        buffer1 = self.normalize(buffer1)
        buffer1 = self.to_tensor(buffer1)

        if Debug:
            print('buffer1 shape: ', buffer1.shape)
            print('label: ', self.label_array[index])

        return torch.from_numpy(buffer1), torch.from_numpy(labels)

    # Randomly flip the video frames in the buffer horizontally
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    # Normalize the video frames in the buffer
    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    # Convert the video frames in the buffer to tensor
    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    # Load the video frames from the buffer
    def load_frames(self, view_file_dir):
        if Debug:
            print('file direction: ', view_file_dir)

        frames1 = sorted([os.path.join(view_file_dir, img) for img in os.listdir(view_file_dir)])
        frame_count = len(frames1)
        while frame_count <= 16:
            frame_count += 1
        buffer1 = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        i1 = 0
        i = 0
        frame1 = None

        for _, frame_name1 in enumerate(frames1):
            frame1 = cv2.resize(cv2.imread(frame_name1), (self.resize_width, self.resize_height))
            frame1 = np.array(frame1.astype(np.float32))
            buffer1[i1] = frame1
            i1 += 1

        while i < 16:
            buffer1[i] = frame1
            i += 1

        return buffer1

    # Crop the video frames in the buffer
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    if Debug:
        print('Dataset_Path= ', Dataset_Path)

    # Example usage
    train_data = VideoDataset()

    if Debug:
        print('train_data= ', train_data.__len__())

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs_1, labels = sample
        # Process the samples as needed

        if Debug:
            print('inputs_1.shape', inputs_1.shape)
            print('label.shape', labels.shape)

        if i == 1:
            break
