from dataloaders.datasets import cityscapes
from torch.utils.data import DataLoader

def make_data_loader(args, train_data=None, valid_data=None, test_data=None, **kwargs):
    if train_data is not None:
        train_set = cityscapes.CityscapesSegmentation(args, data=train_data, split='train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        train_loader = None

    if valid_data is not None:
        val_set = cityscapes.CityscapesSegmentation(args, data=valid_data, split='val')
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
    else:
        val_loader = None

    if test_data is not None:
        test_set = cityscapes.CityscapesSegmentation(args, data=test_data, split='test')
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader

