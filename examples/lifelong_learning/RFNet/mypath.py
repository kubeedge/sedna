class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return '/home/robo/m0063/project/RFNet-master/Data/cityscapes/'      # folder that contains leftImg8bit/
        elif dataset == 'citylostfound':
            return '/home/robo/m0063/project/RFNet-master/Data/cityscapesandlostandfound/'  # folder that mixes Cityscapes and Lost and Found
        elif dataset == 'cityrand':
            return '/home/robo/m0063/project/RFNet-master/Data/cityrand/'
        elif dataset == 'target':
            return '/home/robo/m0063/project/RFNet-master/Data/target/'
        elif dataset == 'xrlab':
            return '/home/robo/m0063/project/RFNet-master/Data/xrlab/'
        elif dataset == 'e1':
            return '/home/robo/m0063/project/RFNet-master/Data/e1/'
        elif dataset == 'mapillary':
            return '/home/robo/m0063/project/RFNet-master/Data/mapillary/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
