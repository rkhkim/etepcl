from setuptools import setup

setup(
    name = 'etepcl',
    package = ['etepcl'],
#     entry_points = {
#             "console_scripts": ['etepcl = etepcl.cli']
#         },
    version = '1.00',
    description = 'Command line end to end processing',
    author = 'Anmol Warman',
    url = 'https://github.com/anmolwarman/etepcl',
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
