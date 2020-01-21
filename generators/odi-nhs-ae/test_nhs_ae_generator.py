from generate import main
import csv
import os


def test_generator():


    os.system('curl -o "./data/London postcodes.csv" https://www.doogal.co.uk/UKPostcodesCSV.ashx?region=E12000007')

    main(20, 'test_generated.csv', 23414)

    with open(os.path.join(os.getcwd(), "..", "..", "datasets", "generated", "odi_nhs_ae", "test_reference.csv")) as r,\
    open(os.path.join(os.getcwd(), "..", "..", "datasets", "generated", "odi_nhs_ae", "test_generated.csv")) as f:
        reader_r = csv.reader(r)
        reader_f = csv.reader(f)
        
        for line_r, line_f in zip(reader_r, reader_f):
            assert line_r == line_f
