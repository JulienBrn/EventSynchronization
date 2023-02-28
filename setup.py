from distutils.core import setup


setup(
    name='eventsynchronization',
    packages=['eventsynchronization'],
    version='0.1',
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
)