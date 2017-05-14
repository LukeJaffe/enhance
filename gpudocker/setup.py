from distutils.core import setup

setup(
    name='gpudocker',
    version='1.0',
    description='Docker Wrapper for GPU Utilization',
    author='Luke Jaffe',
    author_email='jaffe5@llnl.gov',
    packages=['gpudocker'],
    package_dir={'gpudocker': 'src'},
    scripts=['src/gpudocker']
)
