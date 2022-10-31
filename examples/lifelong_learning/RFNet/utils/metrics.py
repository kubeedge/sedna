import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    
    def Pixel_Accuracy_Class_Curb(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('-----------Acc of each classes-----------')
        print("road         : %.6f" % (Acc[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (Acc[1] * 100.0), "%\t")
        Acc = np.nanmean(Acc[:2])
        return Acc
        

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('-----------Acc of each classes-----------')
        print("road         : %.6f" % (Acc[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (Acc[1] * 100.0), "%\t")
        print("building     : %.6f" % (Acc[2] * 100.0), "%\t")
        print("wall         : %.6f" % (Acc[3] * 100.0), "%\t")
        print("fence        : %.6f" % (Acc[4] * 100.0), "%\t")
        print("pole         : %.6f" % (Acc[5] * 100.0), "%\t")
        print("traffic light: %.6f" % (Acc[6] * 100.0), "%\t")
        print("traffic sign : %.6f" % (Acc[7] * 100.0), "%\t")
        print("vegetation   : %.6f" % (Acc[8] * 100.0), "%\t")
        print("terrain      : %.6f" % (Acc[9] * 100.0), "%\t")
        print("sky          : %.6f" % (Acc[10] * 100.0), "%\t")
        print("person       : %.6f" % (Acc[11] * 100.0), "%\t")
        print("rider        : %.6f" % (Acc[12] * 100.0), "%\t")
        print("car          : %.6f" % (Acc[13] * 100.0), "%\t")
        print("truck        : %.6f" % (Acc[14] * 100.0), "%\t")
        print("bus          : %.6f" % (Acc[15] * 100.0), "%\t")
        print("train        : %.6f" % (Acc[16] * 100.0), "%\t")
        print("motorcycle   : %.6f" % (Acc[17] * 100.0), "%\t")
        print("bicycle      : %.6f" % (Acc[18] * 100.0), "%\t")
        print("dynamic      : %.6f" % (Acc[19] * 100.0), "%\t")
        print("stair        : %.6f" % (Acc[20] * 100.0), "%\t")
        if self.num_class == 20:
            print("small obstacles: %.6f" % (Acc[19] * 100.0), "%\t")
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        print('-----------IoU of each classes-----------')
        print("road         : %.6f" % (MIoU[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (MIoU[1] * 100.0), "%\t")
        print("building     : %.6f" % (MIoU[2] * 100.0), "%\t")
        print("wall         : %.6f" % (MIoU[3] * 100.0), "%\t")
        print("fence        : %.6f" % (MIoU[4] * 100.0), "%\t")
        print("pole         : %.6f" % (MIoU[5] * 100.0), "%\t")
        print("traffic light: %.6f" % (MIoU[6] * 100.0), "%\t")
        print("traffic sign : %.6f" % (MIoU[7] * 100.0), "%\t")
        print("vegetation   : %.6f" % (MIoU[8] * 100.0), "%\t")
        print("terrain      : %.6f" % (MIoU[9] * 100.0), "%\t")
        print("sky          : %.6f" % (MIoU[10] * 100.0), "%\t")
        print("person       : %.6f" % (MIoU[11] * 100.0), "%\t")
        print("rider        : %.6f" % (MIoU[12] * 100.0), "%\t")
        print("car          : %.6f" % (MIoU[13] * 100.0), "%\t")
        print("truck        : %.6f" % (MIoU[14] * 100.0), "%\t")
        print("bus          : %.6f" % (MIoU[15] * 100.0), "%\t")
        print("train        : %.6f" % (MIoU[16] * 100.0), "%\t")
        print("motorcycle   : %.6f" % (MIoU[17] * 100.0), "%\t")
        print("bicycle      : %.6f" % (MIoU[18] * 100.0), "%\t")
        print("dynamic      : %.6f" % (MIoU[19] * 100.0), "%\t")
        print("stair        : %.6f" % (MIoU[20] * 100.0), "%\t")
        if self.num_class == 20:
            print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Mean_Intersection_over_Union_Curb(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        print('-----------IoU of each classes-----------')
        print("road         : %.6f" % (MIoU[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (MIoU[1] * 100.0), "%\t")
        
        if self.num_class == 20:
            print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

        MIoU = np.nanmean(MIoU[:2])
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        CFWIoU = freq[freq > 0] * iu[freq > 0]
        print('-----------FWIoU of each classes-----------')
        print("road         : %.6f" % (CFWIoU[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (CFWIoU[1] * 100.0), "%\t")
       
        return FWIoU

    def Frequency_Weighted_Intersection_over_Union_Curb(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        CFWIoU = freq[freq > 0] * iu[freq > 0]
        print('-----------FWIoU of each classes-----------')
        print("road         : %.6f" % (CFWIoU[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (CFWIoU[1] * 100.0), "%\t")

        return np.nanmean(CFWIoU[:2])

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




