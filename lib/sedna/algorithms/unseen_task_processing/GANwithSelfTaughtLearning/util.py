import yaml


def load_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        # print(data)
        return data


if __name__ == '__main__':
    load_yaml('config.yaml')
