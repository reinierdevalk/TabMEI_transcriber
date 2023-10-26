# TabMEI_transcriber
`transcriber.py` is a Python script for transcribing TabMEI files into CMN.

# Instructions
Clone or download the repository and `cd` into it:

    $ cd <path/to/TabMEI_transcriber/>

The file to be transcribed is placed in the `in/` folder; the output file will be stored in the `out/` folder. The script takes as argument one positional argument (the name of the file to be transcribed, possibly preceded by 'in/'), and a set of optional arguments, which take their default values when not specified. In its minimal form (i.e., with all the options set to their default values), it is called with

    $ python3 transcriber.py in/<filename>

from a Linux command line, and with

    $ python transcriber.py in\<filename>

from a Windows command line (note the use of `python` instead of `python3`, and a backslash (`\`) instead of a forward slash to separate directories).
    
To see all options and their possible values, use the Help function:

    $ python3 transcriber.py -h

or

    $ python3 transcriber.py --help

## Example
With the following command, the input file is transcribed onto a double staff (`-s d`) with a B flat major key signature, i.e., two flats (`-k -2`). This double staff is placed above the tablature 'staff', which is kept. The tablature is shown as Italian lute tablature (`-tp ILT`), and the A tuning is assumed for the tablature (`-tn A`).

    $ python3 transcriber.py in/<filename> -s d -k -2 -tp ILT -tn A

The resulting MEI file is stored in the `out/` folder.

# Requirements
* Python 3.12.0
* Java installation
