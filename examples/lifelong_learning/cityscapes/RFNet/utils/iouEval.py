import torch


class iouEval:

    def __init__(self, nClasses, ignoreIndex=20):

        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()
        self.cdp_obstacle = torch.zeros(1).double()
        self.tp_obstacle = torch.zeros(1).double()
        self.idp_obstacle = torch.zeros(1).double()
        self.tp_nonobstacle = torch.zeros(1).double()
        # self.cdi = torch.zeros(1).double()

    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be "batch_size x nClasses x H x W"
        # cdi = 0

        # print ("X is cuda: ", x.is_cuda)
        # print ("Y is cuda: ", y.is_cuda)

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()  # dim index src 按照列用1替换0，索引为x
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1):
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)  # 加一维
            x_onehot = x_onehot[:, :self.ignoreIndex]  # ignoreIndex后的都不要
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0


        tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fpmult = x_onehot * (
                    1 - y_onehot - ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fnmult = (1 - x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

        cdp_obstacle = tpmult[:, 19].sum()  # obstacle index 19
        tp_obstacle = y_onehot[:, 19].sum()

        idp_obstacle = (x_onehot[:, 19] - tpmult[:, 19]).sum()
        tp_nonobstacle = (-1*y_onehot+1).sum()

        # for i in range(0, x.size(0)):
        #     if tpmult[i].sum()/(y_onehot[i].sum() + 1e-15) >= 0.5:
        #         cdi += 1


        self.cdp_obstacle += cdp_obstacle.double().cpu()
        self.tp_obstacle += tp_obstacle.double().cpu()
        self.idp_obstacle += idp_obstacle.double().cpu()
        self.tp_nonobstacle += tp_nonobstacle.double().cpu()
        # self.cdi += cdi.double().cpu()



    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        iou_not_zero = list(filter(lambda x: x != 0, iou))
        # print(len(iou_not_zero))
        iou_mean = sum(iou_not_zero) / len(iou_not_zero)
        tfp = self.tp + self.fp + 1e-15
        acc = num / tfp
        acc_not_zero = list(filter(lambda x: x != 0, acc))
        acc_mean = sum(acc_not_zero) / len(acc_not_zero)

        return iou_mean, iou, acc_mean, acc  # returns "iou mean", "iou per class"

    def getObstacleEval(self):

        pdr_obstacle = self.cdp_obstacle / (self.tp_obstacle+1e-15)
        pfp_obstacle = self.idp_obstacle / (self.tp_nonobstacle+1e-15)

        return pdr_obstacle, pfp_obstacle


# Class for colors
class colors:
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

