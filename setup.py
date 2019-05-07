import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rocketbase',
    packages=setuptools.find_packages(),
    version='0.4.0',
    license='Proprietary',
    description='Making Machine Learning available to everyone',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mirage Technologies AG',
    author_email='hello@mirage.id',
    url='https://rocketbase.ai',
    keywords=['automatic', 'ML', 'deployment', 'pytorch', 'pretrained', 'API'],
    install_requires=[
        'tqdm',
        'requests',
        'google-cloud-storage',
        'numpy>=1.16.3',
        'opencv-python>=4.1.0.25',
        'Pillow>=6.0.0',
        'scikit-image>=0.15.0',
        'scipy>=1.2.1',
        'torch>=1.0.1.post2',
        'torchvision>=0.2.2.post3'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
