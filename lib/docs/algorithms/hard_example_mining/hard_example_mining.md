Module sedna.algorithms.hard_example_mining.hard_example_mining
===============================================================
Hard Example Mining Algorithms

Classes
-------

`CrossEntropyFilter(threshold_cross_entropy=0.5, **kwargs)`
:   Implement the hard samples discovery methods named IBT
    (image-box-thresholds).
    
    :param threshold_cross_entropy: threshold_cross_entropy to filter img,
                        whose hard coefficient is less than
                        threshold_cross_entropy. And its default value is
                        threshold_cross_entropy=0.5

    ### Ancestors (in MRO)

    * sedna.algorithms.hard_example_mining.hard_example_mining.BaseFilter
    * abc.ABC

`IBTFilter(threshold_img=0.5, threshold_box=0.5, **kwargs)`
:   Implement the hard samples discovery methods named IBT
        (image-box-thresholds).
    
    :param threshold_img: threshold_img to filter img, whose hard coefficient
        is less than threshold_img.
    :param threshold_box: threshold_box to calculate hard coefficient, formula
        is hard coefficient = number(prediction_boxes less than
            threshold_box)/number(prediction_boxes)

    ### Ancestors (in MRO)

    * sedna.algorithms.hard_example_mining.hard_example_mining.BaseFilter
    * abc.ABC

`ThresholdFilter(threshold=0.5, **kwargs)`
:   The base class to define unified interface.

    ### Ancestors (in MRO)

    * sedna.algorithms.hard_example_mining.hard_example_mining.BaseFilter
    * abc.ABC