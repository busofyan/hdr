import os
import numpy as np


def read_dir(dirname):
    file_list = os.listdir(dirname)
    exposures = []
    filenames = []

    # Get Filenames and dom + nom for calculating the exposure times
    for m in file_list:
        filenames.append(dirname + m)
        head, sep, tail = m.partition('.')
        dom_nom = [float(s) for s in head.split('_') if s.isdigit()]
        exposures.append(dom_nom[0] / dom_nom[1])

    # Sort two Arrays by Indices
    exposures, filenames = zip(*sorted(zip(exposures, filenames)))

    # Revers Array to get Descending Order
    desc_exposures = exposures[::-1]
    desc_filenames = filenames[::-1]

    return [np.array(desc_filenames), np.array(desc_exposures), len(desc_exposures)]