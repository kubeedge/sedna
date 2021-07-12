Module sedna.datasources
========================
Data set format used as input in tasks of sedna

Classes
-------

`BaseDataSource(data_type='train', func=None)`
:   sedna dataset base class 
    
    :param data_type: train/eval/test
    :param func: process function for each sample in raw data

    ### Descendants

    * sedna.datasources.CSVDataParse
    * sedna.datasources.TxtDataParse

    ### Instance variables

    `is_test_data`
    :

    ### Methods

    `num_examples(self) ‑> int`
    :

    `parse(self, *args, **kwargs)`
    :

    `save(self, output='')`
    :

`CSVDataParse(data_type, func=None)`
:   csv file which contain Structured Data parser
    
    :param data_type: train/eval/test
    :param func: process function for each sample in raw data

    ### Ancestors (in MRO)

    * sedna.datasources.BaseDataSource
    * abc.ABC

    ### Static methods

    `parse_json(lines: dict, **kwargs) ‑> pandas.core.frame.DataFrame`
    :

    ### Methods

    `parse(self, *args, **kwargs)`
    :

`TxtDataParse(data_type, func=None)`
:   txt file which contain image list parser
    
    :param data_type: train/eval/test
    :param func: process function for each sample in raw data

    ### Ancestors (in MRO)

    * sedna.datasources.BaseDataSource
    * abc.ABC

    ### Methods

    `parse(self, *args, **kwargs)`
    :