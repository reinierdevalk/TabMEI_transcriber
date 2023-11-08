import argparse
import os

from utils import call_java, cp


def transcribe_poly(infiles: str, arg_paths: dict, args: argparse.Namespace): # -> None
	inpath = arg_paths['inpath']
	outpath = arg_paths['outpath']
#	filename, ext = os.path.splitext(os.path.basename(infile)) # input file name, extension
#	outfile = filename + '-dipl' + ext # output file

	print(inpath)
	print(outpath)
#	print(filename)
#	print(ext)
#	print(outfile)
#	print(args.key)
#	print(vars(args))

	# Convert all .tc files in infiles to .tbp
	for infile in infiles:
#	for item in os.listdir(inpath):
#		if item.endswith('.tc'):
		f, e = os.path.splitext(os.path.basename(infile))
		cmd = ['java', '-cp', cp, 'imports.TabImport', inpath, inpath, f]	
		call_java(cmd, False)

	# Transcribe all .tbp files in infiles
	model_ID = 'N-bwd-thesis-int-4vv'
	args_dict = vars(args)
	trans_params = '-k=' + args.key + '|' +\
				   '-m=' + args.mode + '|' +\
				   '-s=' + args.staff + '|' +\
				   '-tb=' + args.tab + '|' +\
				   '-tp=' + args.type
#	print(trans_params)
	for f in infiles:
		print(f)

	cmd = ['java', '-cp', cp, 'ui.UI', model_ID, '.', 'false', '', 
		   infiles[0] if len(infiles) == 1 else '', trans_params]
	call_java(cmd, False)
