params = dict()

params['dataset'] = 'RHM'
params['view'] = 'OmniView'
params['view_status'] = 'Normal'
params['model_name'] = 'C3D_model'
params['num_classes'] = 14
params['epoch_num'] = 300
params['batch_size'] = 50
params['step'] = 200
params['num_workers'] = 32
params['learning_rate'] = 1e-4
params['momentum'] = 0.9
params['weight_decay'] = 5e-5
params['display'] = 1
params['pretrained'] = False
params['gpu'] = [0]
params['clip_len'] = 16
params['frame_sample_rate'] = 1
params['useTest'] = True  # See evolution of the test set when training
params['nTestInterval'] = 1  # Run on test set every nTestInterval epochs


