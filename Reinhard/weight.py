def weight(z, zmin, zmax):
    if z <= 0.5 * (zmin + zmax):
        # never let the weights be zero because that would influence the equation system !!!
        w = ((z - zmin) + 1)
    else:
        w = ((zmax - z) + 1)
    return w
