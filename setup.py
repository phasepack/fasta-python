from setuptools import setup

install_requires = ["numpy", "scipy", "matplotlib"]

setup(name="fasta",
      version="0.1",
      description="A solver for convex optimization problems",
      url="https://github.com/phasepack/fasta-python",
      author="Noah Singer",
      author_email="",
      license="GNU GPLv3",
      packages=["fasta"],
      zip_safe=False)
