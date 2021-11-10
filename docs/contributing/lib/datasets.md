# Dataset Development Guide

## Introduction

The Sedna provides interfaces and public methods related to data conversion and sampling in the Dataset class. The user data processing class can inherit from the Dataset class and use these public capabilities.


### 1. Example

The following describes how to use the Dataset by using a `txt-format contain sets of images` as an example. The procedure is as follows:

- 1.1. All dataset classes of Sedna are inherited from the base class `sedna.datasources.BaseDataSource`. The base class BaseDataSource defines the interfaces required by the dataset, provides attributes such as data_parse_func, save, and concat, and provides default implementation. The derived class can reload these default implementations as required.

```python
    
    class BaseDataSource:
        """
        An abstract class representing a :class:`BaseDataSource`.
    
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite parse`, supporting get train/eval/infer
        data by a function. Subclasses could also optionally overwrite `__len__`,
        which is expected to return the size of the dataset.overwrite `x` for the
        feature-embedding, `y` for the target label.
    
        Parameters
        ----------
        data_type : str
            define the datasource is train/eval/test
        func: function
            function use to parse an iter object batch by batch
        """
    
        def __init__(self, data_type="train", func=None):
            self.data_type = data_type  # sample type: train/eval/test
            self.process_func = None
            if callable(func):
                self.process_func = func
            elif func:
                self.process_func = ClassFactory.get_cls(
                    ClassType.CALLBACK, func)()
            self.x = None  # sample feature
            self.y = None  # sample label
            self.meta_attr = None  # special in lifelong learning
    
        def num_examples(self) -> int:
            return len(self.x)
    
        def __len__(self):
            return self.num_examples()
    
        def parse(self, *args, **kwargs):
            raise NotImplementedError
    
        @property
        def is_test_data(self):
            return self.data_type == "test"
    
        def save(self, output=""):
            return FileOps.dump(self, output)

    class TxtDataParse(BaseDataSource, ABC):
        """
        txt file which contain image list parser
        """
    
        def __init__(self, data_type, func=None):
            super(TxtDataParse, self).__init__(data_type=data_type, func=func)
    
        def parse(self, *args, **kwargs):
            pass
    
```

- 1.2. Defining Dataset parse function

```python
    def parse(self, *args, **kwargs):
        x_data = []
        y_data = []
        use_raw = kwargs.get("use_raw")
        for f in args:
            with open(f) as fin:
                if self.process_func:
                    res = list(map(self.process_func, [
                               line.strip() for line in fin.readlines()]))
                else:
                    res = [line.strip().split() for line in fin.readlines()]
            for tup in res:
                if not len(tup):
                    continue
                if use_raw:
                    x_data.append(tup)
                else:
                    x_data.append(tup[0])
                    if not self.is_test_data:
                        if len(tup) > 1:
                            y_data.append(tup[1])
                        else:
                            y_data.append(0)
        self.x = np.array(x_data)
        self.y = np.array(y_data)
```


### 2. Commissioning

The preceding implementation can be directly used in the PipeStep in Sedna or independently invoked. The code for independently invoking is as follows:

```python
import os
import unittest


def _load_txt_dataset(dataset_url):
    # use original dataset url,
    # see https://github.com/kubeedge/sedna/issues/35
    return os.path.abspath(dataset_url)


class TestDataset(unittest.TestCase):

    def test_txtdata(self):
        train_data = TxtDataParse(data_type="train", func=_load_txt_dataset)
        train_data.parse(train_dataset_url, use_raw=True)
        self.assertEqual(len(train_data), 1)


if __name__ == "__main__":
    unittest.main()
```
