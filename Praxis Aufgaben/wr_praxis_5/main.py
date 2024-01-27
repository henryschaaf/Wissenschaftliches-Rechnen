import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # TODO: create principal term for DFT matrix

    omega = np.exp(2j * np.pi / n,dtype=np.complex128)

    print("omega: ", omega)

    # TODO: fill matrix with values

    for i in range(n):
        for j in range(n):
            F[i,j] = np.complex128(omega ** (j * i))

    # TODO: normalize dft matrix

    
    F = np.dot(np.complex128(1/np.sqrt(n)), F)

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = False
    # TODO: check that F is unitary, if not return false

    conjugated = np.conjugate(matrix).T

    matrix_result = np.dot(conjugated, matrix)

    if np.allclose(matrix_result, np.eye(matrix.shape[0])) and matrix.shape[0] == matrix.shape[1]:
        unitary = True
    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []

    for i in range(n):
        sigs.append(np.zeros(n))
        sigs[i][i] = 1
    
    print("sigs: ", sigs)
    print("\n")

    # Fourier-transformed signals
    fsigs = []

    for i in  range(n):
        fsigs.append(np.dot(dft_matrix(n), sigs[i]))

    print("fsigs: ", fsigs)
    print("\n")

    

    # TODO: create signals and extract harmonics out of DFT matrix

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    datacopy = np.copy(data)

    # TODO: implement shuffling by reversing index bits
    
    länge = len(bin(data.size)[2:]) - 1

    for i in range(datacopy.size):
        newnumber = 0
        

        number = bin(int(i))[2:].zfill(länge)
        for j in range(länge - 1,0 - 1, -1 ):
                newnumber = newnumber * 10
                if int(number[j]) == 0:
                    newnumber = newnumber + 0
                else:
                    newnumber = newnumber + 1
                
        number = int(str(newnumber), 2)


        data[i] = datacopy[number]
      

    
    

    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # TODO: first step of FFT: shuffle data

    fdata = shuffle_bit_reversed_order(fdata)

    # TODO: second step, recursively merge transforms
    for i in range(int(np.log2(n))):
        for l in range(int(n / 2 ** (i + 1))):
            for k in range(2 ** i):
               
                p = np.dot(np.exp(2j * np.pi * k / 2 ** (i + 1),dtype=np.complex128),fdata[2 ** (i + 1) * l + k + 2 ** i])

                fdata[2 ** (i + 1) * l + k + 2 ** i] = fdata[2 ** (i + 1) * l + k] - p

                fdata[2 ** (i + 1) * l + k] += p

                
    



    # TODO: normalize fft signal
                
    fdata = np.dot(np.complex128(1/np.sqrt(n)), fdata)

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0
    #not neccessary beacause the length is one

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency
    
    for j in range(num_samples):
        data[j] = np.sin(2 *  np.pi * j / (num_samples - 1) * f)



    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 500, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # TODO: compute Fourier transform of input data

    adata = np.fft.fft(adata)
    

    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    print("bandlimit_index",bandlimit_index)
    print("adata",adata[0:8000])
    print("\n")
    print("range",int((adata.size/2)) + 1)
    print("\n")
    print("size",adata.size)
    
    for i in range(1,adata.size):
        if i > bandlimit_index and i < adata.size - bandlimit_index:
            adata[i] = 0
            

    # TODO: compute inverse transform and extract real component
   
    adata_filtered = np.fft.ifft(adata).real

    #print("adata_filtered 2",adata_filtered[0:1000])


    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
