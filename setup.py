import sys
from setuptools import find_packages, setup


setup(
    name="mwcat",
    version="0.1",
    url="https://github.com/tarekziade/mwcat",
    packages=find_packages(),
    description="Classifies per Wikipedia main categories.",
    author="Tarek Ziade",
    author_email="tziade@mozilla.com",
    include_package_data=True,
    zip_safe=False,
    entry_points="""
      [console_scripts]
      mwcat-create-dataset = mwcat.create_dataset:main
      mwcat-train = mwcat.train:main
      mwcat-evaluate = mwcat.evaluate:main
      mwcat-validate = mwcat.manual_validation:main
      mwcat-distil = mwca.distil:main
      """,
)
