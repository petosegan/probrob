from distutils.core import setup, Command
from distutils.extension import Extension


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys, subprocess

        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(
    name='probrob',
    version='',
    packages=['probrob'],
    url='',
    license='',
    author='Richard W. Turner',
    author_email='rwturner@stanford.edu',
    description='Probabilistic Robotics Simulator',
    requires=['numpy', 'matplotlib', 'scipy', 'Cython'],
    cmdclass={'test': PyTest},
    ext_modules = [Extension("ray_trace", ["ray_trace.c"])]
)
