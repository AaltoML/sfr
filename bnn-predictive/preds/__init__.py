import os

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = "/".join(PACKAGE_DIR.split("/")[:-1])
DATA_DIR = ROOT + "/data"

# PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
print("PACKAGE_DIR {}".format(PACKAGE_DIR))
# ROOT = "/".join(PACKAGE_DIR.split("/")[:-1])
print("ROOT {}".format(ROOT))
# DATA_DIR = os.path.join(ROOT, "bnn-predictive/data")
print("DATA_DIR {}".format(DATA_DIR))
