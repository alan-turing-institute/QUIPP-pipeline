from generate import main as generate_main
from deidentify import main as deidentify_main
import csv
import os


def test_generator():

    # Test postcode data was generated with command:
    # $ awk 'NR % 50000 == 1' London\ postcodes.csv > London\ postcodes\ test.csv
    generate_main(20, os.path.join(os.path.dirname(os.path.realpath(__file__)), "test-data"),
                  "test_generated",
                  23414,
                  os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "London postcodes test.csv"))

    with open(os.path.join(os.getcwd(), "test-data", "test_reference.csv")) as r, \
         open(os.path.join(os.getcwd(), "test-data", "test_generated.csv")) as f:
        reader_r = csv.reader(r)
        reader_f = csv.reader(f)
        
        for line_r, line_f in zip(reader_r, reader_f):
            assert line_r == line_f


def test_deidentify():
    deidentify_main(os.path.join(os.path.dirname(os.path.realpath(__file__)), "test-data", "test_reference"),
                    "test_deidentify",
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "test-data"),
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "London postcodes test.csv"))

    with open(os.path.join(os.getcwd(), "test-data", "test_deidentify_reference.csv")) as r, \
         open(os.path.join(os.getcwd(), "test-data", "test_deidentify.csv")) as f:

        reader_r = csv.reader(r)
        reader_f = csv.reader(f)

        for line_r, line_f in zip(reader_r, reader_f):
            assert line_r == line_f
