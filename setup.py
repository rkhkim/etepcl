from setuptools import setup

setup(
    name = 'etepcl',
    packages = ['etepcl'],
    entry_points = {
            "console_scripts": ['etepcl = etepcl.cli:main']
        },
    version = '1.3',
    description = 'Command line end to end processing',
    author = 'Anmol Warman',
    url = 'https://github.com/anmolwarman/etepcl',
    download_url = 'https://github.com/hugorut/etepcl/tarball/1.3', 
    keywords = ['machine learning', 'ai', 'neural network'],
    setup_requires=["numpy"],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pyparsing',
        'matplotlib',
        'click',        
        'tensorflow',
        'keras'
    ],
)
