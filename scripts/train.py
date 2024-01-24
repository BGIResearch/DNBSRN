import time
import torch
# from torch.utils.tensorboard import SummaryWriter


def train_valid(opt, tloader, vloader, net):
    # writer = SummaryWriter(f'{opt.savefile}/tensorboard')
    if opt.loss_function == r'L1Loss':
        loss_function = torch.nn.L1Loss()
    if opt.loss_function == r'SmoothL1Loss':
        loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LearningRate, betas=(0.9, 0.999))
    if opt.lr_update == r'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    if opt.lr_update == r'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    if opt.lr_update == r'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 130, 180, 220], gamma=0.5)
    start_epoch = 0
    min_v_loss = 1e9
    if opt.breakpoint[0]:
        point = torch.load(opt.breakpoint[1])
        start_epoch = point['epoch']
        min_v_loss = point['min_v_loss']
        net.load_state_dict(point['net'])
        optimizer.load_state_dict(point['optimizer'])
        scheduler.load_state_dict(point['scheduler'])
    net = net.to(opt.device)
    net.train()
    for epoch in range(start_epoch, opt.epoch):
        t1 = time.time()
        opt.logger.info(f'epoch: {epoch + 1}    learning rate: {optimizer.param_groups[0]["lr"]}')
        t_loss = 0
        for i, img in enumerate(tloader):
            lr, hr = img[0], img[1]
            lr, hr = lr.to(opt.device), hr.to(opt.device)
            optimizer.zero_grad()
            sr = net(lr)
            loss = loss_function(sr, hr)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            print(f'\rtrain[{epoch+1}/{opt.epoch}][{i+1}/{len(tloader)}] loss: {loss.item()}', end='')
        mean_t_loss = t_loss/len(tloader)
        opt.logger.info(f'train loss: {mean_t_loss:.10f}    spend time: {(time.time()-t1):.2f}s')
        # writer.add_scalar('mean_t_loss', mean_t_loss, epoch+1)
        t2 = time.time()
        v_loss = 0
        for i, img in enumerate(vloader):
            lr, hr = img[0], img[1]
            lr, hr = lr.to(opt.device), hr.to(opt.device)
            with torch.no_grad():
                sr = net(lr)
                loss = loss_function(sr, hr)
            v_loss += loss.item()
            print(f'\rvalid[{epoch+1}/{opt.epoch}][{i+1}/{len(vloader)}] loss: {loss.item()}', end='')
        mean_v_loss = v_loss/len(vloader)
        opt.logger.info(f'valid loss: {mean_v_loss:.10f}    spend time: {(time.time() - t2):.2f}s')
        # writer.add_scalar('mean_v_loss', mean_v_loss, epoch+1)
        if opt.lr_update == r'ReduceLROnPlateau':
            scheduler.step(mean_v_loss)
        if opt.lr_update == r'StepLR' or opt.lr_update == r'MultiStepLR':
            scheduler.step()
        if mean_v_loss < min_v_loss:
            min_v_loss = mean_v_loss
            opt.logger.info('save model')
            torch.save(net.state_dict(), fr'{opt.weight}')
        if (epoch+1) % opt.saveinterval == 0:
            point = {'epoch': epoch+1,
                     'min_v_loss': min_v_loss,
                     'net': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict()}
            torch.save(point, f'{opt.savefile}/{epoch+1}.pth')
        if optimizer.param_groups[0]['lr'] < 1e-4:
            break
    # writer.close()
