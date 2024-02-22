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
								 			  files (.mei, .xml) in the input folder (\'user/in/\').\
								 			  Depending on the value of -tr (--trans), these transcriptions\
								 			  are either (1) diplomatic transcriptions in notehead notation,\
								 			  or (2) polyphonic transcriptions output by a machine\
								 			  learning model for voice separation trained on tablature.\
								 			  Depending on the value of -tb (--tab), the transcriptions\
								 			  are either placed above the tablature, or replace it.\
								 			  NB: polyphonic transcription to be implemented.',
								 epilog=	 'Stores new MEI files in the output folder (\'user/out/\').'
								)
parser.add_argument('-f', '--file',
					metavar='',
					help='input file;\
						  form options are <input_folder>/<filename>.<ext> or\
						  <filename>.<ext> (where the former enables autocompletion).\
						  To be provided only when a single file in the input folder\
						  must be transcribed -- if not provided, all .mei and .xml files\
						  in the input folder are transcribed'
					)
parser.add_argument('-tn', '--tuning', 
					choices=['F', 'F-', 'G', 'G-', 'A', 'A-'], 
					metavar='', 
					help='tuning;\
						  options are [F, F-, G, G-, A, A-] (where minus signs indicate Abzug).\
						  To be provided only when the tuning given in the input file must be\
						  overruled'
					)
parser.add_argument('-k', '--key', 
					choices=[str(i) for i in list(range(-5, 6, 1))], 
					default='0', 
					metavar='',
					help='key signature, i.e.,\
						  number of flats (negative) or sharps (positive);\
						  options are [-5, ..., 5], default is 0'
					)
parser.add_argument('-m', '--mode', 
					choices=['0', '1'], 
					default='0',
					metavar='', 
					help='key signature mode, i.e.,\
						  major (0) or minor (1);\
						  options are [0, 1], default is 0'
					)
parser.add_argument('-s', '--staff', 
					choices=['s', 'd'], 
					default='d',
					metavar='', 
					help='staff type, i.e.,\
						  single or double;\
						  options are [s, d], default is d'
					)
parser.add_argument('-tb', '--tab', 
					choices=['y', 'n'], 
					default='y',
					metavar='',
					help='tablature retention;\
						  options are [y, n], default is y'
					)
parser.add_argument('-tp', '--type', 
					choices=['FLT', 'ILT', 'SLT', 'GLT'], 
					default='FLT',
					metavar='',
					help='tablature type;\
						  options are [FLT, ILT, SLT, GLT], default is FLT'
					)
parser.add_argument('-tr', '--trans', 
					choices=['d', 'p'], 
					default='d',
					metavar='',
					help='transcription style:\
						  diplomatic or polyphonic;\
						  options are [d, p], default is d'
					)
args = parser.parse_args()

sl = '/' if os.name == 'posix' else '\\'


if __name__ == '__main__':
	scriptpath = os.getcwd() # full path to script
	paths = {
			 'inpath': os.path.join(scriptpath, sl.join(['user', 'in'])), # full path to input file
			 'outpath': os.path.join(scriptpath, sl.join(['user', 'out'])) # full path to output file
			}

	# Collect files to transcribe
	infiles = []
	# a. Single file processing
	if args.file != None:
		infiles.append(os.path.split(args.file)[-1]) # input file (without user/in/)
	# b. Batch processing
	else:
		for item in os.listdir(paths['inpath']):
			if args.trans == 'd' and (item.endswith('.xml') or item.endswith('.mei')):
				infiles.append(item)
			elif args.trans == 'p' and item.endswith('.tbp'):
				infiles.append(item)

	# Transcribe files
	if args.trans == 'd':
		transcribe_dipl(infiles, paths, args)
	else:
		print('Not implemented.')
#		transcribe_poly(infiles, paths, args)