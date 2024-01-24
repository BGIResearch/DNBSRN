import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--trainmodel', action='store_true')
parser.add_argument('-C', '--testmodel', action='store_true')
parser.add_argument('-E', '--metrics', action='store_true')
parser.add_argument('-m', '--model', type=str, choices=['DNBSRN', 'IMDN', 'RFDN', 'RLFN', 'EDSR', 'RDN', 'RCAN',
'DNBSRN_kernel_size_3', 'DNBSRN_kernel_size_5', 'DNBSRN_kernel_size_9', 'DNBSRN_delete_IIC', 'DNBSRN_delete_SRB', 'DNBSRN_preprocess_false', 'WF'])
parser.add_argument('-t', '--test', type=str, nargs='+', choices=['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8'])
parser.add_argument('-g', '--gpu', type=str, default='0', help='choose GPU')
parser.add_argument('-p', '--preprocess', type=str, choices=['true', 'false'], default='true')
parser.add_argument('-s', '--save_HM', type=str, choices=['true', 'false'], default='false')
parser.add_argument('-i', '--input_HM', type=str, choices=['true', 'false'], default='false')
parser.add_argument('-c', '--cut', type=int, default=1)
parser.add_argument('--model_option', default={
    'DNBSRN':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'DNBSRN_kernel_size_3':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'DNBSRN_kernel_size_5':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'DNBSRN_kernel_size_9':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'DNBSRN_delete_IIC':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'DNBSRN_delete_SRB':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'DNBSRN_preprocess_false':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'IMDN':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'RFDN':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'RLFN':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 512},
    'EDSR':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 128},
    'RDN':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 128},
    'RCAN':
        {'batchsizeT': 8,
         'batchsizeV': 8,
         'imgTsize': 128}})
parser.add_argument('--test_option', default={
    'dataset1': {'size': [1500, 1500],
                 'cycle': 51,
                 'fov_C': 52,
                 'fov_R': 53,
                 'chip': 'DP88',
                 'bit': 8},
    'dataset2': {'size': [1500, 1500],
                 'cycle': 51,
                 'fov_C': 49,
                 'fov_R': 46,
                 'chip': 'DP88',
                 'bit': 8},
    'dataset3': {'size': [1500, 1500],
                 'cycle': 51,
                 'fov_C': 47,
                 'fov_R': 46,
                 'chip': 'DP88',
                 'bit': 8},
    'dataset4': {'size': [1500, 1500],
                 'cycle': 51,
                 'fov_C': 57,
                 'fov_R': 55,
                 'chip': 'DP88',
                 'bit': 8},
    'dataset5': {'size': [2200, 2200],
                 'cycle': 202,
                 'fov_C': 24,
                 'fov_R': 24,
                 'chip': 'DP84',
                 'bit': 16},
    'dataset6': {'size': [2200, 2200],
                 'cycle': 202,
                 'fov_C': 24,
                 'fov_R': 25,
                 'chip': 'DP84',
                 'bit': 16},
    'dataset7': {'size': [2200, 2200],
                 'cycle': 202,
                 'fov_C': 25,
                 'fov_R': 24,
                 'chip': 'DP84',
                 'bit': 16},
    'dataset8': {'size': [2200, 2200],
                 'cycle': 202,
                 'fov_C': 25,
                 'fov_R': 25,
                 'chip': 'DP84',
                 'bit': 16}})
parser.add_argument('--imgLHdir', type=str, default=r'./train_image/LR_HR')
parser.add_argument('--channel', type=list, default=['A', 'C', 'G', 'T'])
parser.add_argument('--nT', type=int, default=3200, help='number of train image')
parser.add_argument('--nV', type=int, default=800, help='number of valid image')
parser.add_argument('--LearningRate', type=float, default=1e-4)
parser.add_argument('--loss_function', type=str, default=r'SmoothL1Loss')
parser.add_argument('--lr_update', type=str, default=r'MultiStepLR')
parser.add_argument('--breakpoint', type=list, default=[False, r'.pth'])
parser.add_argument('--saveinterval', type=int, default=301)
parser.add_argument('--epoch', type=int, default=300)
opt = parser.parse_args()
