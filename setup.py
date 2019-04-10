import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rockethub',
    packages=setuptools.find_packages(),
    version='0.1.1',
    license='Proprietary',
    description='Accelerating Machine Learning Research',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mirage Technologies AG',
    author_email='hello@mirage.id',
    url='https://rockethub.ai',
    keywords=['Automating', 'ML', 'Research'],
    install_requires=[
        'tqdm',
        'requests',
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
