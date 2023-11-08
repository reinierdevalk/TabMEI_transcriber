import argparse
import os

from utils import call_java, cp

parser = argparse.ArgumentParser(prog=		 'mapper',
								 description='Maps a tablature onto its vocal model.',
								 epilog=	 'Stores files in the output folder (\'user/out/\').')

parser.add_argument('-p', '--position',  
					choices=['t', 'b'],
					default='t',
					metavar='',
					help='the position of the tab: at the top (t) or bottom (b) of the transcription;\
						  (default is t)')
parser.add_argument('-s', '--style',
					choices=['s', 'd'],
					default='d',  
					metavar='',
					help='the style of the transcription: score (s) or double staff (d);\
						  (default is d)')
parser.add_argument('-o', '--ornament',
					choices=['y', 'n'],
					default='y',  
					metavar='',
					help='whether or not (y/n) to retain ornamental notes (default is y)')
parser.add_argument('-d', '--duration',
					choices=['y', 'n'],
					default='n',  
					metavar='',
					help='whether or not (y/n) to add duration (default is n)')
args = parser.parse_args()

options = ''
if args.position != 't':
	options = options + args.position
if args.style != 'd':
	options = options + args.style
if args.ornament != 'y':
	options = options + 'o'
if args.duration != 'n':
	options = options + 'd'
if options != '':
	options = '-' + options

cmd = ['java', '-cp', cp, 'tabmapper.TabMapper', options]
call_java(cmd, False)