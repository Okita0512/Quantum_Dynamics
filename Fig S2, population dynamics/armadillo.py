import numpy as np 

"""
  'IVEC': 'ARMA_MAT_TXT_IS004',
  'DVEC': 'ARMA_MAT_TXT_FN008',
  'CVEC': 'ARMA_MAT_TXT_FC016',
  'IMAT': 'ARMA_MAT_TXT_IS004',
  'DMAT': 'ARMA_MAT_TXT_FN008',
  'CMAT': 'ARMA_MAT_TXT_FC016',
  'ICUB': 'ARMA_CUB_TXT_IS004',
  'DCUB': 'ARMA_CUB_TXT_FN008',
  'CCUB': 'ARMA_CUB_TXT_FC016',
"""

def arma_write(ndarray, filename):

    shape = ndarray.shape
    dimen = len(shape)

    with open(filename, 'w') as f:
        
        if dimen == 1:
            if issubclass(type(ndarray[0]), np.int_):
                print('ARMA_MAT_TXT_IS004\n%d %d' % (shape[0], 1), file=f)
                for row in ndarray:
                    print('%d' % row, file=f)
            elif issubclass(type(ndarray[0]), float):
                print('ARMA_MAT_TXT_FN008\n%d %d' % (shape[0], 1), file=f)
                for row in ndarray:
                    print('%.8e' % row, file=f)
            elif issubclass(type(ndarray[0]), complex):
                print('ARMA_MAT_TXT_FC016\n%d %d' % (shape[0], 1), file=f)
                for row in ndarray:
                    print('(%.8e,%-.8e)' % (row.real, row.imag), file=f)

        elif dimen == 2:

            if issubclass(type(ndarray[0, 0]), np.int_):
                print('ARMA_MAT_TXT_IS004\n%d %d' % (shape[0], shape[1]),
                      file=f)
                for row in ndarray:
                    print(' '.join('%d' % x for x in row), file=f)
            elif issubclass(type(ndarray[0, 0]), float):
                print('ARMA_MAT_TXT_FN008\n%d %d' % (shape[0], shape[1]),
                      file=f)
                for row in ndarray:
                    print(' '.join('%.8e' % x for x in row), file=f)
            elif issubclass(type(ndarray[0, 0]), complex):
                print('ARMA_MAT_TXT_FC016\n%d %d' % (shape[0], shape[1]),
                      file=f)
                for row in ndarray:
                    print(' '.join('(%.8e,%-.8e)' % (x.real, x.imag)
                                   for x in row),
                          file=f)

        elif dimen == 3:

            if issubclass(type(ndarray[0, 0, 0]), np.int_):
                print('ARMA_CUB_TXT_IS004\n%d %d %d' %
                      (shape[1], shape[2], shape[0]),
                      file=f)
                for slc in ndarray:
                    for row in slc:
                        print(' '.join('%d' % x for x in row))
            elif issubclass(type(ndarray[0, 0, 0]), float):
                print('ARMA_CUB_TXT_FN008\n%d %d %d' %
                      (shape[1], shape[2], shape[0]),
                      file=f)
                for slc in ndarray:
                    for row in slc:
                        print(' '.join('%-.8e' % x for x in row), file=f)
            elif issubclass(type(ndarray[0, 0, 0]), complex):
                print('ARMA_CUB_TXT_FC016\n%d %d %d' %
                      (shape[1], shape[2], shape[0]),
                      file=f)
                for slc in ndarray:
                    for row in slc:
                        print(' '.join('(%.8e,%-.8e)' % (x.real, x.imag)
                                       for x in row),
                              file=f)