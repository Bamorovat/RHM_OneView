import os


class Path(object):
    @staticmethod
    def db_dir(database, view1, view2):
        # har_path = '/home/abbas/Desktop/Action_Dataset/RHHAR'
        har_path = '/home/abbas/Desktop/Action_Dataset'
        if database == 'rhhar':
            # folder that contains class labels
            root_dir = os.path.join(har_path, 'RHHAR')

            # Save preprocess data into output_dir
            output_dir_1 = os.path.join(har_path, 'RHHAR' + '_' + view1)
            output_dir_2 = os.path.join(har_path, 'RHHAR' + '_' + view2)
            # print('Output Directory: ', output_dir)

            split_file_dir_1 = os.path.join(output_dir_1, 'rhhar' + '_' + view1 + '_' + 'var')
            split_file_dir_2 = os.path.join(output_dir_2, 'rhhar' + '_' + view2 + '_' + 'var')
            # print(split_file_dir)

            return root_dir, output_dir_1, output_dir_2, split_file_dir_1, split_file_dir_2

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir(model):
        # return '/home/mb19aag/test/PytorchVideoRecognition/saved_models/c3d-pretrained.pth'
        if model == 'C3D':
            return '/home/mb19aag/test/PytorchVideoRecognition/saved_models/c3d-pretrained.pth'
        elif model == 'R3D':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/r3d_18-b3b3357e.pth'
        elif model == 'R2Plus1D':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/r2plus1d_18-91a641e6.pth'
        elif model == 'Slow_Fast':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/SLOWFAST_4x16_R50.pkl'
        elif model == 'X3D':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/'

