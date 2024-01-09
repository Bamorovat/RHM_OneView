import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
            view='RobotView', 'FrontView', 'BackView', 'OmniView'
            situation_list = "ConcatenateFrames", "SubFramesAll", "SubFramesGroup", "NormalFrame"
    """

    def __init__(self, view1='FrontView', situation= 'NormalFrame', split='train', clip_len=16, preprocess=True):

        self.dataset_path = '/beegfs/general/mb19aag'
        self.split = split
        self.clip_len = clip_len
        self.view1 = view1
        self.situation = situation

        # self.output_dir_1 = os.path.join(self.dataset_path, 'RHHAR' + '_' + self.view1,
                                        #  'rhhar' + '_' + self.view1 + '_' + 'var')
        
        self.output_dir_1 = os.path.join(self.dataset_path, 'RHHAR' + '_' + self.view1,
                                         'rhhar' + '_' + self.view1 + '_' + self.situation)

        self.split_file_dir_1 = os.path.join(self.output_dir_1, self.split)

        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames_1, labels = [], []

        for label in sorted(os.listdir(self.split_file_dir_1)):
            for fname in os.listdir(os.path.join(self.split_file_dir_1, label)):
                self.fnames_1.append(os.path.join(self.split_file_dir_1, label, fname))
                # self.fnames_2.append(os.path.join(self.split_file_dir_2, label, fname))
                labels.append(label)

        print('Number of {} videos: {:d} in view {}'.format(split, len(self.fnames_1), self.view1))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.fnames_1)

    # def __getitem__(self, index):
    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer1 = self.load_frames(self.fnames_1[index])
        buffer1 = self.crop(buffer1, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer1 = self.randomflip(buffer1)
        buffer1 = self.normalize(buffer1)
        buffer1 = self.to_tensor(buffer1)

        return torch.from_numpy(buffer1), torch.from_numpy(labels)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # print('file direction: ', file_dir)
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        while frame_count <= 16:
            frame_count += 1
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        i = 0
        frame = None

        for _, frame_name in enumerate(frames):
            frame = cv2.resize(cv2.imread(frame_name), (self.resize_width, self.resize_height))
            # print('frame shape 1: ', frame.shape)
            frame = np.array(frame.astype(np.float32))
            # print('frame shape 2: ', frame.shape)
            buffer[i] = frame
            i += 1

        while i < 16:
            buffer[i] = frame
            i += 1

        return buffer

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

    train_data = VideoDataset(view1='FrontView', situation= 'NormalFrame', split='test', clip_len=16, preprocess=True)

    # train_data = VideoDataset(dataset='rhhar', split='test', clip_len=16, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs_1 = sample[0]
        labels = sample[1]
        print(inputs_1.size())
        print(labels)

        if i == 1:
            break
