import os
import numpy as N


def read_dir(dirname):
    file_list = os.listdir(dirname)
    exposures = []
    filenames = []

    # Get Filenames and dom + nom for calculating the exposure times
    for m in file_list:
        filenames.append(m)
        head, sep, tail = m.partition('.')
        dom_nom = [int(s) for s in head.split('_') if s.isdigit()]
        exposures.append(float(dom_nom[0]) / float(dom_nom[1]))

    # Sort two Arrays by Indices
    exposures, filenames = zip(*sorted(zip(exposures, filenames)))

    # Revers Array to get Descending Order
    desc_exposures = exposures[::-1]
    desc_filenames = filenames[::-1]

    return [N.array(desc_filenames), N.array(desc_exposures), len(desc_exposures)]
