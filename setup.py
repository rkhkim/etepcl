from setuptools import setup

setup(
    name = 'e2ep',
    packages = ['e2ep'],
    entry_points = {
            "console_scripts": ['e2ep = e2ep.cli:main']
        },
    version = '1.0',
    description = 'Command line end to end processing',
    author = 'Ryan Kim, Danny Guo, Anmol Warman, Shawn Chao',
    url = 'https://github.com/anmolwarman/e2ep',
    keywords = ['machine learning', 'ai', 'neural nework'],
    setup_requires=["numpy"],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pyparsing',
        'matplotlib',
        'click'
    ],
)
