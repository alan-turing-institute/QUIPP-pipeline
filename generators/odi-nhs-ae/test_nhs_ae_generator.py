from generate import main as generate_main
from deidentify import main as deidentify_main
import csv
import os


def test_generator():

    # Test postcode data was generated with command:
    # $ awk 'NR % 50000 == 1' London\ postcodes.csv > London\ postcodes\ test.csv
    generate_main(20, "test_generated.csv", 23414, os.path.join(os.getcwd(), "test-data"), "London postcodes test.csv")

    with open(os.path.join("test-data", "test_reference.csv")) as r, \
         open(os.path.join("test-data", "test_generated.csv")) as f:
        reader_r = csv.reader(r)
        reader_f = csv.reader(f)
        
        for line_r, line_f in zip(reader_r, reader_f):
            assert line_r == line_f


def test_deidentify():
    deidentify_main("test_reference.csv", "test_deidentify.csv", postcode_file="London postcodes test.csv")

    data_path = os.path.join(os.getcwd(), "..", "..", "datasets", "generated", "odi_nhs_ae")
    with open(os.path.join(data_path, "test_deidentify_reference.csv")) as r, \
         open(os.path.join(data_path, "test_deidentify.csv")) as f:

        reader_r = csv.reader(r)
        reader_f = csv.reader(f)

        for line_r, line_f in zip(reader_r, reader_f):
            assert line_r == line_f
