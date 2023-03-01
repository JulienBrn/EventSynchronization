from distutils.core import setup


setup(
    name='event_synchronization',
    packages=['event_synchronization'],
    version='0.3',
    license='MIT',
    description = 'A synchronization API for two event channels not based on Dynamic Time Warping. ',
    description_file = "README.md",
    author="Julien Braine",
    author_email='julienbraine@yahoo.fr',
    url='https://github.com/JulienBrn/EventSynchronization',
    download_url = 'https://github.com/JulienBrn/EventSynchronization.git',
    package_dir={'': 'src'},
    keywords=['python',  'event', 'synchronization'],
    install_requires=['pandas'],
    python_requires='>=3.10'
)