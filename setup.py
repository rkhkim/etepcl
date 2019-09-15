from setuptools import setup

setup(
    name = 'E2EP-CL',
    packages = ['E2EP-CL'],
    entry_points = {
            "console_scripts": ['E2EP-CL = E2EP-CL.cli:main']
        },
    version = '1.0',
    description = 'Command line end to end processing',
    author = 'Anmol Warman, Ryan Kim, Shawn Chao, Danny Guo',
    url = 'https://github.com/anmolwarman/E2EP-CL',
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
        'keras', 
        'os',
        'math'
        
    ],
)
