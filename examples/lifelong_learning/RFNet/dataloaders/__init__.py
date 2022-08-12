from dataloaders.datasets import cityscapes, citylostfound, cityrand, target, xrlab, e1, mapillary
from torch.utils.data import DataLoader

def make_data_loader(args, train_data=None, valid_data=None, test_data=None, **kwargs):

    if args.dataset == 'cityscapes':
        if train_data is not None:
            train_set = cityscapes.CityscapesSegmentation(args, data=train_data, split='train')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            train_loader, num_class = None, cityscapes.CityscapesSegmentation.NUM_CLASSES

        if valid_data is not None:
            val_set = cityscapes.CityscapesSegmentation(args, data=valid_data, split='val')
            num_class = val_set.NUM_CLASSES
            val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        else:
            val_loader, num_class = None, cityscapes.CityscapesSegmentation.NUM_CLASSES

        if test_data is not None:
            test_set = cityscapes.CityscapesSegmentation(args, data=test_data, split='test')
            num_class = test_set.NUM_CLASSES
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        else:
            test_loader, num_class = None, cityscapes.CityscapesSegmentation.NUM_CLASSES

        # custom_set = cityscapes.CityscapesSegmentation(args, split='custom_resize')
        # custom_loader = DataLoader(custom_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        # return train_loader, val_loader, test_loader, custom_loader, num_class
        return train_loader, val_loader, test_loader, num_class

    if args.dataset == 'citylostfound':
        if args.depth:
            train_set = citylostfound.CitylostfoundSegmentation(args, split='train')
            val_set = citylostfound.CitylostfoundSegmentation(args, split='val')
            test_set = citylostfound.CitylostfoundSegmentation(args, split='test')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        else:
            train_set = citylostfound.CitylostfoundSegmentation_rgb(args, split='train')
            val_set = citylostfound.CitylostfoundSegmentation_rgb(args, split='val')
            test_set = citylostfound.CitylostfoundSegmentation_rgb(args, split='test')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    if args.dataset == 'cityrand':
        train_set = cityrand.CityscapesSegmentation(args, split='train')
        val_set = cityrand.CityscapesSegmentation(args, split='val')
        test_set = cityrand.CityscapesSegmentation(args, split='test')
        custom_set = cityrand.CityscapesSegmentation(args, split='custom_resize')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        custom_loader = DataLoader(custom_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, custom_loader, num_class
        
    if args.dataset == 'target':
        train_set = target.CityscapesSegmentation(args, split='train')
        val_set = target.CityscapesSegmentation(args, split='val')
        test_set = target.CityscapesSegmentation(args, split='test')
        custom_set = target.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        custom_loader = DataLoader(custom_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, custom_loader, num_class
        
    if args.dataset == 'xrlab':
        train_set = xrlab.CityscapesSegmentation(args, split='train')
        val_set = xrlab.CityscapesSegmentation(args, split='val')
        test_set = xrlab.CityscapesSegmentation(args, split='test')
        custom_set = xrlab.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        custom_loader = DataLoader(custom_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, custom_loader, num_class  
        
    if args.dataset == 'e1':
        train_set = e1.CityscapesSegmentation(args, split='train')
        val_set = e1.CityscapesSegmentation(args, split='val')
        test_set = e1.CityscapesSegmentation(args, split='test')
        custom_set = e1.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        custom_loader = DataLoader(custom_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, custom_loader, num_class  
        
    if args.dataset == 'mapillary':
        train_set = mapillary.CityscapesSegmentation(args, split='train')
        val_set = mapillary.CityscapesSegmentation(args, split='val')
        test_set = mapillary.CityscapesSegmentation(args, split='test')
        custom_set = mapillary.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        custom_loader = DataLoader(custom_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, custom_loader, num_class

    else:
        raise NotImplementedError

