import numpy
import plotly.express as px
import pickle
from os import path

## PARAMETRIC DEFINITION FOR
## E-LASER SETUP in the CDR ECAL-P

# ECAL DIMENSIONAL PARAMETERS
DX = 5  # mm
DY = 5  # mm
Z_WSi = 4258.38 + 0.16 - 4254.5  # mm
DZ = 4.502  # mm
X0 = 39.13 + DX / 2  # mm

XX = 110
YY = 11
ZZ = 21

# GEOMETRIC PARAMETERS
INTERVAL2 = 4254.5 - 3814  # mm
DIPOLE_DEVIATION = 547.4  # mm/(E/GeV)
DIPOLE_TANGENT = 0.31060  # /(E/GeV)

def ShiftingX(x_phys, incz, z_interval=INTERVAL2, f1=DIPOLE_DEVIATION, f2=DIPOLE_TANGENT):
    x_shifted = x_phys * (f2 + f1 * z_interval) / (f2 + f1 * (z_interval + incz))
    return x_shifted

def ShiftingXArray(x0=X0, dimension=3, pads=(ZZ, XX, YY), pad_sizes=(DZ, DX, DY), wsi_gap=Z_WSi,
                   z_interval=INTERVAL2, f1=DIPOLE_DEVIATION, f2=DIPOLE_TANGENT):
    x_phys = numpy.array(
        [[[x0 + pad_sizes[1] * xx for yy in range(pads[2])] for xx in range(pads[1])] for zz in range(pads[0])])
    x_shifted = numpy.array([[[ShiftingX(x_phys[zz, xx, yy], wsi_gap + pad_sizes[0] * zz) \
                               for yy in range(pads[2])] for xx in range(pads[1])] for zz in range(pads[0])])
    if dimension == 3:
        return x_shifted
    elif dimension == 2:
        return x_shifted[:, :, 5]
    elif dimension == 1:
        return x_shifted[0, :, 5]
    else:
        raise Exception()

SHIFT_2D_PATH = r'C:\Users\dor00\PycharmProjects\Weight-Shape-Decomposition\pipelines\shift_array_2d'
with open(SHIFT_2D_PATH, 'rb') as f:
    shift_2d_matrix = pickle.load(f)

if __name__ == '__main__':
    test = ShiftingXArray(dimension=2)
    with open(SHIFT_2D_PATH, 'wb+') as f:
        pickle.dump(test, f)
    fig = px.imshow(test)
    fig.show()
