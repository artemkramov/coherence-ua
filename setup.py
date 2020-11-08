import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


setuptools.setup(
    name='coherence-ua',
    version='0.0.4',
    packages=setuptools.find_packages(),
    url='https://github.com/artemkramov/coherence-ua',
    author='Artem Kramov',
    author_email='artemkramov@gmail.com',
    description='Transformer-based model to estimate the coherence of Ukrainian-language texts',
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[
          'ufal.udpipe', 'tensorflow', 'tensorflow-datasets', 'numpy'
    ],
    include_package_data=True
)
