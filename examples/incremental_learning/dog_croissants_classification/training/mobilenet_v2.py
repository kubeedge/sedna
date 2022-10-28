import mindspore as ms
from mindvision.classification.models import mobilenet_v2
from mindvision.dataset import DownLoad



class mobilenet_v2_fine_tune:
    # TODO: save model
    def __init__(self, base_model_url=None):
        models_download_url = "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt"
        dl=DownLoad()
        if base_model_url==None:
            dl.download_url(models_download_url)
        else:
            dl.download_url(models_download_url,filename=base_model_url )
        self.network = mobilenet_v2(num_classes=2, resize=224)
        #print("base_model_url == "+base_model_url)
        self.param_dict = ms.load_checkpoint(base_model_url)


    def get_train_network(self):
        self.filter_list = [x.name for x in self.network.head.classifier.get_parameters()]
        for key in list(self.param_dict.keys()):
            for name in self.filter_list:
                if name in key:
                    print("Delete parameter from checkpoint: ", key)
                    del self.param_dict[key]
                    break
        ms.load_param_into_net(self.network, self.param_dict)
        return self.network

    def get_eval_network(self):
        ms.load_param_into_net(self.network, self.param_dict)
        return self.network