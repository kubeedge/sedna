import codecs
import logging
import os
import pickle
import shutil

LOG = logging.getLogger(__name__)


def clean_folder(folder):
    if not os.path.exists(folder):
        LOG.info(f"folder={folder} is not exist.")
    else:
        LOG.info(f"clean target dir, dir={folder}")
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))


def remove_path_prefix(org_str: str, prefix: str):
    """remove the prefix, for converting path in container to path in host."""
    p = prefix[:-1] if prefix.endswith('/') else prefix
    if org_str.startswith(p):
        out_str = org_str.replace(p, '', 1)
        return out_str
    else:
        LOG.info(f"remove prefix failed, original str={org_str}, "
                 f"prefix={prefix}")
        return org_str


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


def model_layer_flatten(weights):
    """like this:
    weights.shape=[(3, 3, 3, 64), (64,), (3, 3, 64, 32), (32,), (6272, 64),
        (64,), (64, 32), (32,), (32, 2), (2,)]
    flatten_weights=[(1728,), (64,), (18432,), (32,), (401408,), (64,),
        (2048,), (32,), (64,), (2,)]
    :param weights:
    :return:
    """
    flatten = [layer.reshape((-1)) for layer in weights]
    return flatten


def model_layer_reshape(flatten_weights, shapes):
    shaped_model = []
    for idx, flatten_layer in enumerate(flatten_weights):
        shaped_model.append(flatten_layer.reshape(shapes[idx]))
    return shaped_model
