"""
This script must be called from the folder that holds it, and that furthermore 
contains the following subfolders:
- in/	contains the input MEI file
- out/	where the output MEI file is stored
- java/	contains the Java code required for the pitch spelling:
        - utils/lib/commons-lang3-3.8.1.jar
        - utils/bin/tools/music/PitchKeyTools.class 
        - utils/bin/tools/text/StringTools.class

NB: Updated from Python 3.6.0 to 3.12.0 for this script.

Relevant Python documentation
- https://docs.python.org/3/library/argparse.html

TODO
- have the choices rendered automatically in the parser.add:argument()s' help='...' 
  (or remove metavar='')
- how do I make a rest invisible?
- diplomat.py
  - @head.fill on <note> is not rendered in Verovio
  - show flags option: do not show flags above notehead notation (/tab) if tab + nh

"""

import argparse
import os
from diplomat import transcribe_dipl
from polyphonist import transcribe_poly

parser = argparse.ArgumentParser(prog=		 'transcriber',
								 description='Creates a transcription in CMN from all tablature MEI\
								 			  files in the input folder (\'user/in/\'). Depending on the\
								 			  value of -tr (--trans), these transcription are either\
								 			  (1) diplomatic transcriptions in notehead notation, or\
								 			  (2) polyphonic transcriptions output by a machine\
								 			  learning model for voice separation trained on tablature.\
								 			  Depending on the value of -tb (--tab), the transcriptions\
								 			  are either placed above the tablature, or replace it.',
								 epilog=	 'Stores new MEI files in the output folder (\'user/out/\').')
parser.add_argument('-f', '--file',
					metavar='',
					help='the input file, possibly preceded by the input folder name (\'user/in/\') --\
						  to be provided only when a single file from the input folder is to be\
						  transcribed')
parser.add_argument('-tn', '--tuning', 
					choices=['F', 'F-', 'G', 'G-', 'A', 'A-'], 
					metavar='', 
					help='the tuning -- to be provided only when the tuning given in the input file\
						  is to be overruled); options are [F, F-, G, G-, A, A-]')
parser.add_argument('-k', '--key', 
					choices=[str(i) for i in list(range(-5, 6, 1))], 
					default='0', 
					metavar='',
					help='the key signature for the diplomatic transcription, expressed as its\
						  number of accidentals (where a negative number indicates flats);\
						  options are [-5, ..., 5], default is 0')
parser.add_argument('-m', '--mode', 
					choices=['0', '1'], 
					default='0',
					metavar='', 
					help='the key signature\'s \'mode\': major (0) or minor (1);\
						  options are [0, 1], default is 0')
parser.add_argument('-s', '--staff', 
					choices=['s', 'd'], 
					default='d',
					metavar='', 
					help='the staff type: single or double; options are [s, d], default is d')
parser.add_argument('-tb', '--tab', 
					choices=['y', 'n'], 
					default='y',
					metavar='',
					help='whether or not to retain the tab in the resulting transcription;\
						  options are [y, n], default is y')
parser.add_argument('-tp', '--type', 
					choices=['FLT', 'ILT', 'SLT', 'GLT'], 
					default='FLT',
					metavar='',
					help='the tablature type;\
						  options are [FLT, ILT, SLT, GLT], default is FLT')
parser.add_argument('-tr', '--trans', 
					choices=['d', 'p'], 
					default='d',
					metavar='',
					help='the transcription style: diplomatic or polyphonic;\
						  options are [d, p], default is d')
args = parser.parse_args()


if __name__ == "__main__":
	scriptpath = os.getcwd() # full path to script
	paths = {
			 'inpath': os.path.join(scriptpath, 'user/in/'), # full path to input file
			 'outpath': os.path.join(scriptpath, 'user/out/') # full path to output file
			}
	if not os.path.exists(paths['outpath']):
		os.makedirs(paths['outpath'])

	infiles = []
	if args.file != None:
		print(os.path.split(args.file)[-1])
		infiles.append(os.path.split(args.file)[-1]) # input file (without user/in/)
	else:
		for item in os.listdir(paths['inpath']):
			if args.trans == 'd' and (item.endswith('.xml') or item.endswith('.mei')):
				infiles.append(item)
			elif args.trans == 'p' and item.endswith('.tc'):
				infiles.append(item)	

	if args.trans == 'd':
		transcribe_dipl(infiles, paths, args)
	else:
		transcribe_poly(infiles, paths, args)