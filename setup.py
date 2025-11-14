from setuptools import setup, find_packages

setup(
    name='nested-learning-continual',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project implementing a Nested Learning deep learning architecture for continual learning with image datasets.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'PyYAML',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'nested-learning=src.main:main',
        ],
    },
)