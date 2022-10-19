from basemodel import val_args
from utils.metrics import Evaluator
from tqdm import tqdm
from dataloaders import make_data_loader
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ('accuracy', 'robo_accuracy')

@ClassFactory.register(ClassType.GENERAL)
def accuracy(y_true, y_pred, **kwargs):
    args = val_args()
    _, _, test_loader = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(args.num_class)

    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, img_path) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
            if args.depth:
                depth = depth.cuda()

        target[target > evaluator.num_class-1] = 255
        target = target.cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    # Test during the training
    # Acc = evaluator.Pixel_Accuracy()
    CPA = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("CPA:{}, mIoU:{}, fwIoU: {}".format(CPA, mIoU, FWIoU))
    return CPA

@ClassFactory.register(ClassType.GENERAL)
def robo_accuracy(y_true, y_pred, **kwargs):
    y_pred = y_pred[0]
    args = val_args()
    _, _, test_loader = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(args.num_class)

    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, img_path) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
            if args.depth:
                depth = depth.cuda()

        target[target > evaluator.num_class-1] = 255
        target = target.cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    # Test during the training
    # Acc = evaluator.Pixel_Accuracy()
    CPA = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("CPA:{}, mIoU:{}, fwIoU: {}".format(CPA, mIoU, FWIoU))
    return CPA