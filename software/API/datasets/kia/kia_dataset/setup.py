from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '20201125'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = ""
#with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'pip_requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='kia_dataset',
    version=__version__,
    description='Tools required to handle the kia dataset.',
    long_description=long_description,
    url='TODO',
    download_url='TODO/tarball/' + __version__,
    license='internal',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python',
    ],
    keywords='',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    author='Michael Fuerst',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='Michael.Fuerst@dfki.de',
    entry_points={
        'console_scripts': [
            'kia_fetch = kia_dataset.downloader:fetch',
            'kia_extract = kia_dataset.extractor:extract',
            'kia_fix_extract = kia_dataset.fixes.extract:extract'
            'kia_fix_names_tranche_1 = kia_dataset.fixes.names:tranche_1',
            'kia_fix_names_tranche_2 = kia_dataset.fixes.names:tranche_2',
            'kia_fix_box_2d = kia_dataset.fixes.box_2d:main',
            'kia_fix_box_3d = kia_dataset.fixes.box_3d:main',
            'kia_pp_fake_lidar = kia_dataset.post_processing.lidar_from_depth:main',
            'kia_test_filenames = kia_dataset.tests.filenames:main',
            'kia_test_attributenames = kia_dataset.tests.attributenames:main',
            'kia_test_vis = kia_dataset.tests.visualize_dataset:main',
        ]
    }
)
