from setuptools import setup, find_packages

setup(name='silens',
      version='0.1',
      description='First version of Machine Learning Interpretation tool by SIngulariti',
      url='http://github.com/singularitiai/SiLens',
      author='Kanishka Sharma',
      author_email='kanishka.sharma@singulariti.ai',
      license='BSD',
      packages= find_packages(exclude=['js', 'node_modules', 'tests']),
      install_requires=[
          'lime',
          'numpy',
          'pandas',
          'scipy',
          'scikit-learn>=0.18',
          'scikit-image>=0.12'
      ],
      include_package_data=True,
      zip_safe=False)
