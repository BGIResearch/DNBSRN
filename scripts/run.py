import os
import logging
import torch
import data
import train
import test
import metrics
import function
from parameter import opt

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
opt.device = torch.device('cuda:0')

opt.savefile = fr'./result/{opt.model}'
os.makedirs(opt.savefile, exist_ok=True)
logging.basicConfig(filename=fr'{opt.savefile}/record.log',
                    level=logging.INFO,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    filemode='a',
                    format='%(asctime)s-%(message)s')
opt.logger = logging.getLogger()
opt.logger.info(f'-----------------------------------')
opt.logger.info(f'{opt.model}')
opt.logger.info(f'-----------------------------------')
if opt.model == 'DNBSRN' or opt.model == 'DNBSRN_preprocess_false':
    from model.DNBSRN import DNBSRN
    net = DNBSRN()
elif opt.model == 'IMDN':
    from model.IMDN import IMDN
    net = IMDN()
elif opt.model == 'RFDN':
    from model.RFDN import RFDN
    net = RFDN()
elif opt.model == 'RLFN':
    from model.RLFN import RLFN
    net = RLFN()
elif opt.model == 'EDSR':
    from model.EDSR import EDSR
    net = EDSR()
elif opt.model == 'RDN':
    from model.RDN import RDN
    net = RDN()
elif opt.model == 'RCAN':
    from model.RCAN import RCAN
    net = RCAN()
elif opt.model == 'DNBSRN_kernel_size_3':
    from model.DNBSRN import DNBSRN
    net = DNBSRN(kernel_size=3)
elif opt.model == 'DNBSRN_kernel_size_5':
    from model.DNBSRN import DNBSRN
    net = DNBSRN(kernel_size=5)
elif opt.model == 'DNBSRN_kernel_size_9':
    from model.DNBSRN import DNBSRN
    net = DNBSRN(kernel_size=9)
elif opt.model == 'DNBSRN_delete_IIC':
    from model.DNBSRN_delete_IIC import DNBSRN_delete_IIC
    net = DNBSRN_delete_IIC()
elif opt.model == 'DNBSRN_delete_SRB':
    from model.DNBSRN_delete_SRB import DNBSRN_delete_SRB
    net = DNBSRN_delete_SRB()
opt.weight = fr'{opt.savefile}/{opt.model}.pth'

if opt.trainmodel:
    opt.logger.info(f'start train')
    opt.batchsizeT = opt.model_option[opt.model]['batchsizeT']
    opt.batchsizeV = opt.model_option[opt.model]['batchsizeV']
    opt.imgTsize = opt.model_option[opt.model]['imgTsize']
    function.set_seed()
    t, v = data.train_valid_dataloader(opt)
    train.train_valid(opt, t, v, net)
    opt.logger.info(f'finish train')

if opt.testmodel:
    opt.logger.info(f'start test')
    for i in opt.test:
        if opt.input_HM == 'true' and opt.preprocess == 'false':
            opt.input = fr'./test_image/{i}/HM'
        else:
            opt.input = fr'./test_image/{i}/WF'
        test.image_reconstruct(opt, i, net)
    opt.logger.info(f'finish test')

if opt.metrics:
    opt.logger.info(f'metrics')
    metrics.metrics_calculate(opt, net)
