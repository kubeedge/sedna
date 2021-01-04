from setuptools import setup

setup(
    name='neptune',
    version='0.0.1',
    description="The neptune package is designed to help developers \
                better use open source frameworks such as tensorflow \
                on Neptune project",
    packages=['neptune'],
    python_requires='>=3.6',
    install_requires=[
        'flask>=1.1.2',
        'keras>=2.4.3',
        'Pillow>=8.0.1',
        'opencv-python>=4.4.0.44',
        'websockets>=8.1'
        'requests>=2.24.0'
    ]
)
