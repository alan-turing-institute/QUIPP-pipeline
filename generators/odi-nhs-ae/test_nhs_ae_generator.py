from generate import main
import csv
import os


def test_generator():

    main(10)

    with open(os.path.join(os.getcwd(), "..", "..", "datasets", "generated", "odi_nhs_ae", "hospital_ae_data.csv")) as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):

            if index == 1:
                expected_row = ["039-379-1657", "48", "69", "Epsom General Hospital", "2019-04-07 04:37:28", "Defibrillation/pacing", "Male", "HA2 8HZ"]
                for e, r in zip(expected_row, row):
                    assert e == r

            if index == 7:
                expected_row = ["051-938-0592", "32", "54", "Northwick Park & St Marks Hospital", "2019-04-06 06:12:57", "Other (consider alternatives)", "Male", "EC50 0US"]
                for e, r in zip(expected_row, row):
                    assert e == r
