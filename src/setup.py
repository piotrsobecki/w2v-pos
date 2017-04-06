from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='w2vpos',
      version='0.1',
      description='Part of speech weighted word2vec text representation',
      long_description=readme,
      url='https://github.com/piotrsobecki/w2v-pos',
      author='Piotr Sobecki',
      author_email='ptrsbck@gmail.com',
      license=license,
      packages=find_packages(exclude=('tests', 'docs')),
      install_requires=['gensim','smart_open','opt==0.9'],
      dependency_links=['http://github.com/piotrsobecki/opt/tarball/master#egg=opt-0.9'],
      zip_safe=False)