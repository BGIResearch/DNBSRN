import function
import os
import torch
import glob
import tifffile
import cv2
import numpy as np


def image_reconstruct(opt, i, net):
    opt.output = fr'{opt.savefile}/{i}_SR'
    d = opt.test_option[i]
    net.load_state_dict(torch.load(opt.weight, map_location=opt.device))
    net = net.to(opt.device)
    net.eval()
    os.makedirs(fr'{opt.output}', exist_ok=True)
    if opt.preprocess == 'true':
        img_chain1_1 = {}
        img_chain2_1 = {}
        if d['bit'] == 8:
            for k in opt.channel:
                s = glob.glob(fr'{opt.input}/S001/*.{k}.*.tif')
                img = tifffile.imread(s)
                hist = cv2.calcHist([img], [0], None, [256], [0, 255])
                img_chain1_1[k] = function.sum_hist(hist)
                if d['cycle'] > 101:
                    s = glob.glob(fr'{opt.input}/S102/*.{k}.*.tif')
                    img = tifffile.imread(s)
                    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
                    img_chain2_1[k] = function.sum_hist(hist)
        elif d['bit'] == 16:
            for k in opt.channel:
                s = glob.glob(fr'{opt.input}/S001/*.{k}.*.tif')
                img_chain1_1[k] = tifffile.imread(s)
                if d['cycle'] > 101:
                    s = glob.glob(fr'{opt.input}/S102/*.{k}.*.tif')
                    img_chain2_1[k] = tifffile.imread(s)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cyclist = os.listdir(opt.input)
    cyclist.sort()
    for cyc in cyclist:
        os.makedirs(fr'{opt.output}/{cyc}', exist_ok=True)
        max99 = {}
        imagename = {}
        time = 0
        for k in opt.channel:
            lr = np.zeros([1, 1, d['size'][0], d['size'][1]], dtype=np.float32)
            filelist = glob.glob(fr'{opt.input}/{cyc}/*.C{str(d["fov_C"]).zfill(3)}R{str(d["fov_R"]).zfill(3)}.*.{k}.*.tif')
            imagename[k] = os.path.basename(filelist[0])
            image_in = tifffile.imread(filelist)

            #preprocess
            if opt.preprocess == 'true':
                if d['bit'] == 8:
                    if cyclist.index(cyc) < 101:
                        image_in = function.img_preprocess(img_chain1_1[k], image_in)
                    elif cyclist.index(cyc) < 202:
                        image_in = function.img_preprocess(img_chain2_1[k], image_in)
                elif d['bit'] == 16:
                    if cyclist.index(cyc) < 101:
                        image_in = function.img_preprocess_16(img_chain1_1[k], image_in)
                    elif cyclist.index(cyc) < 202:
                        image_in = function.img_preprocess_16(img_chain2_1[k], image_in)
                if opt.save_HM == 'true':
                    os.makedirs(fr'./test_image/{i}/HM/{cyc}', exist_ok=True)
                    tifffile.imwrite(fr'./test_image/{i}/HM/{cyc}/{imagename[k]}', image_in)
            max99[k] = np.percentile(image_in, 99)
            lr[0][0] = np.float32(image_in/max99[k])

            if opt.cut > 1:
                sr = np.zeros([d['size'][0], d['size'][1]], dtype=np.float32)
                sr = torch.tensor(sr).to(opt.device)
                for m in range(opt.cut):
                    for n in range(opt.cut):
                        lr_in = np.zeros([1, 1, d['size'][0]//opt.cut+18*2, d['size'][1]//opt.cut+18*2], dtype=np.float32)
                        lr_in[0][0] = function.image_cut_extend(lr[0][0], m, n, d['size'][0]//opt.cut, d['size'][1]//opt.cut, 18)
                        lr_in = torch.tensor(lr_in).to(opt.device)
                        with torch.no_grad():
                            sr_out = net(lr_in)
                        sr[m*d['size'][0]//opt.cut:(m+1)*d['size'][0]//opt.cut, n*d['size'][1]//opt.cut:(n+1)*d['size'][1]//opt.cut] = \
                            sr_out[0][0][18:18+d['size'][0]//opt.cut, 18:18+d['size'][1]//opt.cut]
            else:
                lr = torch.tensor(lr).to(opt.device)
                with torch.no_grad():
                    start.record()
                    sr = net(lr)
                    end.record()
                    torch.cuda.synchronize()
                    time = time + start.elapsed_time(end)
            sr = sr.cpu().squeeze().numpy()

            #postprocess
            if d['bit'] == 8:
                img_out = sr*max99[k]*257
            elif d['bit'] == 16:
                img_out = sr*max99[k]
            img_out[img_out > 65535] = 65535
            img_out[img_out < 0] = 0

            tifffile.imwrite(fr'{opt.output}/{cyc}/{imagename[k]}', np.uint16(img_out))
