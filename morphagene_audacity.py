# -*- coding: utf-8 -*-
"""
USAGE: 
morphagene_audacity.py -w <inputwavfile> -l <inputlabels> -o <outputfile>'

Used to convert Audacity labels in .txt form on .WAV files into
    single 32-bit float .WAV with CUE markers within the file, directly
    compatible with the Make Noise Morphagene.
    
Does not require input file to be 48000Hz, only that the Audacity label matches
    the .WAV file that generated it, and that the input .WAV is stereo.
     
See the Morphagene manual for naming conventions of output files:    
    http://www.makenoisemusic.com/content/manuals/morphagene-manual.pdf
    
# see http://stackoverflow.com/questions/15576798/create-32bit-float-wav-file-in-python
# see... http://blog.theroyweb.com/extracting-wav-file-header-information-using-a-python-script
# marker code from Joseph Basquin [https://gist.github.com/josephernest/3f22c5ed5dabf1815f16efa8fa53d476]

Requires wavfile.py (by X-Raym)
"""

import sys, getopt
import numpy as np
from scipy import interpolate
from wavfile import read, write

def test_normalized(array):
    '''
    Determine if an array is entirely -1 < array[i,j] < 1, to see if array is
        normalized
    '''
    return (array > -1).all() and (array < 1).all()

def norm_to_32float(array):
    '''
    Convert a variety of audio types to float32 while normalizing if needed
    '''
    if array.dtype == 'int16': 
        bits=16
        normfactor = 2 ** (bits-1)
        data = np.float32(array) * 1.0 / normfactor
        
    if array.dtype == 'int32': 
        bits=32
        normfactor = 2 ** (bits-1)
        data = np.float32(array) * 1.0 / normfactor
        
    if array.dtype == 'float32': 
        if test_normalized(array):
            data = np.float32(array) # nothing needed
        else:
            bits=32
            normfactor = 2 ** (bits-1)
            data = np.float32(array) * 1.0 / normfactor

    if array.dtype == 'float64': 
        bits=64
        normfactor = 2 ** (bits-1)
        data = np.float32(array) * 1.0 / normfactor
        
    elif array.dtype == 'uint8':
        if isinstance(data[0], (int, np.uint8)):
            bits=8
            # handle uint8 data by shifting to center at 0
            normfactor = 2 ** (bits-1)
            data = (np.float32(array) * 1.0 / normfactor) -\
                            ((normfactor)/(normfactor-1))
    return data

def load_audacity_labels(label_file):
    '''
    Load Audacity labels, ignoring the additional frequency range info lines,
        if labels were exported from a spectrogram.
    '''
    fi = open(label_file, 'r')
    labs = [line.strip().split()[0] for line in fi if not line.startswith('\\')]
    fi.close()
    return np.array(labs).astype('float')

def change_samplerate_interp(old_audio,old_rate,new_rate):
    '''
    Change sample rate to new sample rate by simple interpolation.
    If old_rate > new_rate, there may be aliasing / data loss.
    Input should be in column format, as the interpolation will be completed
        on each channel this way.
    Modified from:
    https://stackoverflow.com/questions/33682490/how-to-read-a-wav-file-using-scipy-at-a-different-sampling-rate
    '''    
    if old_rate != new_rate:
        # duration of audio
        duration = old_audio.shape[0] / old_rate
        
        # length of old and new audio
        time_old  = np.linspace(0, duration, old_audio.shape[0])
        time_new  = np.linspace(0, duration, int(old_audio.shape[0] * new_rate / old_rate))
        
        # fit old_audio into new_audio length by interpolation
        interpolator = interpolate.interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        print('Conversion not needed, old and new rates match')
        return old_audio # conversion not needed

def main(argv):
    inputwavefile = ''
    inputlabelfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hw:l:o:",["wavfile=","labelfile=","outputfile="])
    except getopt.GetoptError:
        print('Error in usage, correct format:\n'+\
            'morphagene_audacity.py -w <inputwavfile> -l <inputlabels> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('morphagene_audacity.py -w <inputwavfile> -l <inputlabels> -o <outputfile>')
            sys.exit()
        elif opt in ("-w", "--wavfile"):
            inputwavefile = arg
        elif opt in ("-l", "--labelfile"):
            inputlabelfile = arg
        elif opt in ("-o", "--outputfile"):
            outputfile = arg

    print('Input wave file: %s'%inputwavefile)
    print('Input label file: %s'%inputlabelfile)
    print('Output Morphagene reel: %s'%outputfile)
    
    ###########################################################################
    '''
    Write single file, edited in Audacity with labels, to Morphagene 32bit
        WAV file at 48000hz sample rate.
    '''
    ###########################################################################
    morph_srate = 48000 # required samplerate for Morphagene
     
    # read labels from stereo Audacity label file, ignore text, and use one channel
    audac_labs = load_audacity_labels(inputlabelfile)
     
    # read pertinent info from audio file, convert, exit if input wave file is broken
    try:
        sample_rate, array, _, _, _ = read(inputwavefile)
        array = norm_to_32float(array)
    except: 
        print('Input file %s.wav is poorly formatted, exiting'%inputwavefile)
        sys.exit()
    
    # check if input wav has a different rate than desired Morphagene rate,
    #   and correct by interpolation
    if sample_rate != morph_srate:
        print("Correcting input sample rate %iHz to Morphagene rate %iHz"%(sample_rate,morph_srate))
        # perform interpolation on each channel, then transpose back
        array = change_samplerate_interp(array.T,float(sample_rate),float(morph_srate)).T
        # convert labels in seconds to labels in frames, adjusting for change
        #   in rate
        sc = float(morph_srate) / float(sample_rate)
        frame_labs = (audac_labs * sample_rate * sc).astype(int)
    else:
        frame_labs = (audac_labs * sample_rate).astype(int)
    frame_dict = [{'position': l, 'label': b'marker%i'%(i+1)} for i,l in enumerate(frame_labs)]
    
    # write wav file with additional cue markers from labels
    # no need to transpose again for data from Audacity
    write(outputfile,morph_srate,array.astype('float32'),
          markers=frame_dict,
          normalized=True)
    print(f'Saved Morphagene reel {outputfile} with {len(frame_labs)} splices')

if __name__ == "__main__":
   main(sys.argv[1:])