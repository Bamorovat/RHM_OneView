
class Path(object):
    @staticmethod
    def model_dir(model):
        if model == 'C3D':
            return 'pretrained_models/c3d-pretrained.pth'
        elif model == 'R3D':
            return 'pretrained_models/r3d_18-b3b3357e.pth'
        elif model == 'R2Plus1D':
            return 'pretrained_models/r2plus1d_18-91a641e6.pth'
        elif model == 'Slow_Fast':
            return 'pretrained_models/SLOWFAST_4x16_R50.pkl'
