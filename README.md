# TabMEI_transcriber
`transcriber.py` is a Python script for transcribing TabMEI files into CMN. The transcription can be diplomatic or polyphonic. 

# Instructions
Clone or download the repository and `cd` into it:

    $ cd <path/to/TabMEI_transcriber/>

The files to be transcribed are placed in the `user/in/` folder; the output files will be stored in the `user/out/` folder. The script takes as arguments a set of optional arguments, which take their default values when not specified. 

In its minimal form -- i.e., with all the options set to their default values -- it is called with

    $ python3 transcriber.py

from a Linux command line, and with

    $ python transcriber.py

from a Windows command line (note the use of `python` instead of `python3`.

To see all options and their possible values, call the Help function with

    $ python3 transcriber.py -h

or

    $ python3 transcriber.py --help

## Example
# Diplomatic transcription
With the following command, a single input file is transcribed diplomatically (default behaviour) onto a single staff (`-s s`), with a B flat major key signature, i.e., two flats (`-k -2`). This staff is placed above the tablature 'staff', which is kept (default behaviour). The tablature is shown as Italian lute tablature (`-tp ILT`), and the A tuning is assumed for the input file (`-tn A`).

    $ python3 transcriber.py user/in/<filename>.xml -s s -k -2 -tp ILT -tn A

where on a Windows command line, a backslash (`\`) is used instead of a forward slash to separate directories.

The resulting MEI file is stored in the `user/out/` folder.

# Polyphonic transcription
Similarly, with the following command, all input files in the input folder are transcribed polyphonically (`-tr p`) onto a double staff (default behaviour), with a B flat major key signature, i.e., two flats (`-k -2`). This double staff replaces the tablature 'staff' (`-tb n`). The tunings as given in the inputs files are assumed for the input files (default behaviour).

    $ python3 transcriber.py -tr p -k -2 -tb n

# Requirements
* Python 3.12.0
* Java installation
