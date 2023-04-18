### Detail block modules
# Conv
class Conv(nn.Module):
    def __init__(self, k, s, p, c_in, c_out):
        super().__init__()
        # 2d conv layer
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        # batch normalization
        self.bn = nn.BatchNorm2d(c_out)
        # SiLU activation
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, shortcut, h, w , c_in):
        super().__init__()
        self.conv = Conv(k=3, s=1, p=1, c_in=c_in, c_out=c_in//2)
        self.conv1 = Conv(k=3, s=1, p=1, c_in=c_in//2, c_out=c_in)
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            xp = self.conv1(self.conv(x))
            x = x + xp
        else:
            x = self.conv(x)
            x = self.conv1(x)        
        return x

class SPPF(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = Conv(k=1, s=1, p=0, c_in=c_in,c_out=c_in)    
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=32, padding=2) 

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        xtot = x1 + x2 + x3 + x4
        #x = torch.cat((x4, xtot), dim=1)
        x = self.conv(xtot)
        return x

class C2f(nn.Module):
    def __init__(self, shortcut, c_in, c_out, h, w, n):
        super().__init__()
        self.conv1 = Conv(k=1, s=1, p=0, c_in=c_in, c_out=c_out)
        self.conv2 = Conv(k=1, s=1, p=0, c_in=int(0.5*(n+2)*c_out), c_out=c_out)
        self.bottleneck = Bottleneck(shortcut=shortcut, h=h, w=w, c_in=c_out//2)
        self.n = n
        self.cout = c_out
        self.cin = c_in

    def forward(self, x):
        x = self.conv1(x)
        
        # split the input into half and the other half channels
        split_x = torch.split(x, x.size(1)//2, dim=1)        
        # current bottleneck
        bn = split_x[0]
        # list of bottlenecks to be stacked
        bn_x = []
        bn_x.append(bn)        
        # iterate through the number of bottlenecks to be stacked
        for i in range(self.n):
            # apply the bottleneck to the first half channels and store them in bn_x
            bn = self.bottleneck(bn)
            bn_x.append(bn)
               
        # concatenate the first half channels with the second half channels
        bn_x.append(split_x[1])
        # concatenate the bottlenecks
        x = torch.cat(bn_x, dim=1)
        x = self.conv2(x)                
        return x

class Detect(nn.Module):
    def __init__(self, num_classes, reg_max, c_in):
        super().__init__()
        self.conv = Conv(k=3, s=1, p=1, c_in=c_in, c_out=c_in)
        self.conv2d_bbox = nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=c_in, out_channels=4*reg_max)
        self.bn1 = nn.BatchNorm2d(4*reg_max)
        self.conv2d_cls = nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=c_in, out_channels=num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.linear = nn.Linear(in_features=6400, out_features=4)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        x_cls = self.conv2d_cls(x)
        x_cls = self.bn2(x_cls)
        x_bbox = self.conv2d_bbox(x)
        x_bbox = self.bn1(x_bbox)
        #x_bbox = self.calculate_bbox(x_bbox)
        return x_cls, x_bbox

    def calculate_bbox(self, x):
        x_pred = torch.sigmoid(x[:,0,:, :])
        y_pred = torch.sigmoid(x[:,1,:, :])
        w_pred = torch.exp(x[:,2,:, :])
        h_pred = torch.exp(x[:,3,:, :])
        bbox = torch.cat([x_pred, y_pred, w_pred, h_pred], dim=1)
        return bbox

# Main Network architecture
class BBoxNet(nn.Module):
    def __init__(self, w, r, d):
        super(BBoxNet, self).__init__()
    # backbone network modules
        self.conv_0_p1 = Conv(k=3, s=2, p=1, c_in=3, c_out=int(64*w))
        self.conv_1_p2 = Conv(k=3, s=2, p=1, c_in=int(64*w), c_out=int(128*w))
        self.c2f_2 = C2f(shortcut=True, h=160, w=160, n=int(3*d), c_in=int(128*w), c_out=int(128*w))
        self.conv_3_p3 = Conv(k=3, s=2, p=1, c_in=int(128*w), c_out=int(256*w))
        self.c2f_4 = C2f(shortcut=True, h=80, w=80, n=int(6*d), c_in=int(256*w), c_out=int(256*w))
        self.conv_5_p4 = Conv(k=3, s=2, p=1, c_in=int(256*w), c_out=int(512*w))
        self.c2f_6 = C2f(shortcut=True, h=40, w=40, n=int(6*d), c_in=int(512*w), c_out=int(512*w))
        self.conv_7_p5 = Conv(k=3, s=2, p=1, c_in=int(512*w), c_out=int(512*w*r))
        self.c2f_8 = C2f(shortcut=True, h=20, w=20, n=int(3*d), c_in=int(512*w*r), c_out=int(512*r*w))
        self.sppf_9 = SPPF(c_in=int(512*w*r))

    # head network modules
        self.upsample_10 = nn.Upsample(size=(40,40), mode='bilinear', align_corners=False)
        self.concat_11 = torch.cat
        self.c2f_12 = C2f(shortcut=False, c_in=int(512*w*(1+r)), c_out=int(512*w), h=40, w=40, n=int(3*d))
        self.upsample_resolution_13a = nn.Upsample(size=(80,80), mode='bilinear', align_corners=False)
        self.upsample_channels_13b = nn.Conv2d(in_channels=int(512*w), out_channels=int(256*w), kernel_size=1)
        self.concat_14 = torch.cat
        self.c2f_15 = C2f(shortcut=False, c_in=int(512*w), c_out=int(256*w), h=80, w=80, n=int(3*d))
        self.conv_16_p3 = Conv(k=3, s=2, p=1, c_in=int(256*w), c_out=int(256*w))
        self.concat_17 = torch.cat
        # ISSUE HERE, THE ARCHITECTURE OUTPUT CHANNEL SIZE IS PROBABLY WRONG, AS THE CONCATENATION DOES NOT INCREASE THE CHANNEL SIZE
        #self.c2f_18 = C2f(shortcut=False, c_in=int(512*w), c_out=int(512*w), h=40, w=40, n=int(3*d))
        self.c2f_18 = C2f(shortcut=False, c_in=192, c_out=192, h=40, w=40, n=int(3*d))
        self.conv_19 = Conv(k=3, s=2, p=1, c_in=192, c_out=192)
        self.concat_20 = torch.cat
        #self.c2f_21 = C2f(shortcut=False, c_in=int(512*w*(1+r)), c_out=int(512*w), h=20, w=20, n=int(3*d))
        self.c2f_21 = C2f(shortcut=False, c_in=448, c_out=int(512*w), h=20, w=20, n=int(3*d))
    
    # output layers
        self.detect1 = Detect(num_classes=6, reg_max=1, c_in=int(256*w))
        self.detect2 = Detect(num_classes=6, reg_max=1, c_in=192)
        self.detect3 = Detect(num_classes=6, reg_max=1, c_in=int(512*w))

    def forward(self,x):
    # backbone pass
        x = self.conv_0_p1(x)
        x = self.conv_1_p2(x)
        x = self.c2f_2(x)
        x = self.conv_3_p3(x)
        x = self.c2f_4(x)
        # save for concat later
        x_4 = x
        
        x = self.conv_5_p4(x)
        x = self.c2f_6(x)
        
        # save for concat later
        x_6 = x
        x = self.conv_7_p5(x)
        x = self.c2f_8(x)
        x = self.sppf_9(x)
        x_9 = x

    # head pass
        # first brancH
        x = self.upsample_10(x)
        x = self.concat_11((x, x_6), dim=1)
        x = self.c2f_12(x)  
        x_12 = x
        x = self.upsample_resolution_13a(x)
        x = self.upsample_channels_13b(x)
        x = self.concat_14((x, x_4), dim=1)
        x = self.c2f_15(x) 
        x_detect1 = x
        
    # second branch
        x = self.conv_16_p3(x)
        # CHECK CHANNEL ISSUE HEREISSUE HERE
        x = self.concat_17((x_12, x), dim=1)
        x = self.c2f_18(x)
        x_detect2 = x
        x = self.conv_19(x)
        # ISSUE PROPAGATES HERE ALSO
        x = self.concat_20((x, x_9), dim=1)
        x = self.c2f_21(x)
        x_detect3 = x
    
    # output layers
        x_cls1, x_bbox1 = self.detect1(x_detect1)
        x_cls2, x_bbox2 = self.detect2(x_detect2)
        x_cls3, x_bbox3 = self.detect3(x_detect3)        

        return [x_cls1, x_bbox1, x_cls2, x_bbox2, x_cls3, x_bbox3]

def calculate_iou(pred_bboxes, gt_bboxes):
     # Calculate intersection
    inter_xmin = torch.max(pred_bboxes[..., 0], gt_bboxes[..., 0])
    inter_ymin = torch.max(pred_bboxes[..., 1], gt_bboxes[..., 1])
    inter_xmax = torch.min(pred_bboxes[..., 2], gt_bboxes[..., 2])
    inter_ymax = torch.min(pred_bboxes[..., 3], gt_bboxes[..., 3])

    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height

    # Calculate union
    pred_area = (pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1])
    gt_area = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
    union_area = pred_area + gt_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area
    return iou
    
def bbox_regression_loss(pred_bboxes, gt_bboxes):
    # Reshape the predicted bounding boxes to (batch, 4, -1)
    pred_bboxes = pred_bboxes.view(pred_bboxes.size(0), 4, -1)
    # Convert the predicted bounding boxes to absolute coordinates
    pred_bboxes_xy = torch.sigmoid(pred_bboxes[:, :2, :])
    pred_bboxes_wh = torch.exp(pred_bboxes[:, 2:, :])
    # Combine the x, y, width, and height to create the final predicted bounding boxes
    pred_bboxes_abs = torch.cat((pred_bboxes_xy, pred_bboxes_wh), dim=1)
    iou_loss = torch.zeros((pred_bboxes_abs.shape[0], pred_bboxes_abs.shape[2]))
    id_list = torch.zeros((pred_bboxes_abs.shape[0], pred_bboxes_abs.shape[1]))


    for sample in range(pred_bboxes_abs.shape[0]):
        for bbox in range(pred_bboxes_abs.shape[2]):
            
            iou = calculate_iou(pred_bboxes_abs[sample, :, bbox], gt_bboxes[sample, :])
            loss = 1 - iou
            iou_loss[sample, bbox] = loss

        # get the index of the predicted bounding box with the highest IoU
        max_iou, max_iou_idx = torch.max(iou_loss[sample], dim=0)
        # if the IoU is greater than 0.5, then the predicted bounding box is a true positive
        if max_iou > 0.5:
            id_list[sample, :] = pred_bboxes_abs[sample, :, max_iou_idx]

    # Calculate the loss
    loss = torch.sum(torch.abs(id_list - gt_bboxes))
    return loss, id_list

    
# define the training function
def train_net(net, train_loader, n_epochs, optimizer, criterion):
    # loop over the number of epochs
    best_loss = 100000
    for epoch in range(n_epochs):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0

        # set the model to training mode
        net.train()
        for i_batch, data in enumerate(dataloader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data['image'] = data['image'].cuda()
                data['bbox'] = data['bbox'].cuda()
                data['label'] = data['label'].cuda()
                data['image_id'] = data['image_id'].cuda()
                
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data['image'].permute(0,3,1,2).float())
            
            # calculate the batch loss
            bbox = torch.stack([t for t in data['bbox']], dim=0)
            loss1, _ = criterion(output[1], bbox)
            loss2,_ = criterion(output[3], bbox)
            loss3,_ = criterion(output[5], bbox)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss = loss1 + loss2 + loss3
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss

            train_loss += loss.item()*data['image'].size(0)
            # save the model if validation loss has decreased
            if loss.item() < best_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss, loss.item()))
                torch.save(net.state_dict(), 'model.pt')
                best_loss = loss.item()
    
            # save the three feature maps with different name        
            #if i_batch%10 == 0:
            #    plt.imsave("Output/feature_map1_"+str(i_batch)+".jpg", output[0][0,0,:,:].detach().cpu().numpy())
            #    plt.imsave("Output/feature_map2_"+str(i_batch)+".jpg", output[1][0,0,:,:].detach().cpu().numpy())
            #    plt.imsave("Output/feature_map3_"+str(i_batch)+".jpg", output[2][0,0,:,:].detach().cpu().numpy())

            # print train loss in percentage, and remaining batches in epoch
            print('Epoch: {} \tBatch: {} \tLoss: {:.6f} \tRemaining: {}'.format(epoch+1, i_batch+1, loss.item(), len(train_loader)-i_batch-1))
            

        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

        # step the scheduler
        #scheduler.step()