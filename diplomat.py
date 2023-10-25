"""
Relevant Python documentation
- https://docs.python.org/3/library/subprocess.html
- https://docs.python.org/3/library/xml.etree.elementtree.html

Useful links
- running Java code from CLI
  - https://stackoverflow.com/questions/16137713/how-do-i-run-a-java-program-from-the-command-line-on-windows
- using subprocess
  - https://www.datacamp.com/tutorial/python-subprocess
  - https://stackoverflow.com/questions/59214417/winerror-2-file-not-found-with-subprocess-run
  - https://stackoverflow.com/questions/21406887/subprocess-changing-directory
  - https://stackoverflow.com/questions/77239936/how-to-call-subprocess-efficiently-and-avoid-calling-it-in-a-loop
  - https://stackoverflow.com/questions/9322796/keep-a-subprocess-alive-and-keep-giving-it-commands-python
- calling Java from Python
  - https://www.askpython.com/python/examples/call-java-using-python
  - https://www.tutorialspoint.com/how-can-we-read-from-standard-input-in-java
- other
  - https://stackoverflow.com/questions/1953761/accessing-xmlns-attribute-with-python-elementree
  - https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file
  - https://www.geeksforgeeks.org/xml-parsing-python/
  - https://w3c.github.io/smufl/latest/tables/renaissance-lute-tablature.html

ElementTree tips
- use find() to find first direct child
- use findall() with XPath to find first recursive child. See 
  https://docs.python.org/3/library/xml.etree.elementtree.html#elementtree-xpath

"""

import argparse
import json
import os.path
import subprocess
import xml.etree.ElementTree as ET
from subprocess import Popen, PIPE, run

notationtypes = {'FLT': 'tab.lute.french',
				 'ILT': 'tab.lute.italian',
				 'SLT': 'tab.lute.spanish',
				 'GLT': 'tab.lute.german'
				}
tunings = {'F' : [('f', 4), ('c', 4), ('g', 3), ('eb', 3), ('bb', 2), ('f', 2)],
		   'F-': [('f', 4), ('c', 4), ('g', 3), ('eb', 3), ('bb', 2), ('eb', 2)],
		   'G' : [('g', 4), ('d', 4), ('a', 3), ('f', 3), ('c', 3), ('g', 2)], 
		   'G-': [('g', 4), ('d', 4), ('a', 3), ('f', 3), ('c', 3), ('f', 2)], 
		   'A' : [('a', 4), ('e', 4), ('b', 3), ('g', 3), ('d', 3), ('a', 2)], 
		   'A-': [('a', 4), ('e', 4), ('b', 3), ('g', 3), ('d', 3), ('g', 2)]
		  }
shift_intervals = {'F': -2, 'G': 0, 'A': 2}
smufl_lute_durs = {1: 'luteDurationDoubleWhole',
				   2: 'luteDurationWhole',
				   4: 'luteDurationHalf', 
				   8: 'luteDurationQuarter',
				   16: 'luteDuration8th',
				   32: 'luteDuration16th',
				   '.': 'augmentationDot'
				  }
cp_dirs = ['java/utils/lib/*',
		   'java/utils/bin/']
cp = (':' if os.name == 'posix' else ';').join(cp_dirs)
java_path = 'tools.music.PitchKeyTools' # <package>.<package>.<file>
verbose = False


def _handle_namespaces(path: str): # -> Tuple
	# There is only one namespace, whose key is an empty string -- replace the  
	# key with something meaningful ('mei'). See
	# https://stackoverflow.com/questions/42320779/get-the-namespaces-from-xml-with-python-elementtree/42372404#42372404
	# To avoid an 'ns0' prefix before each tag, register the namespace as an empty string. See
	# https://stackoverflow.com/questions/8983041/saving-xml-files-using-elementtree
	ns = dict([node for _, node in ET.iterparse(path, events=['start-ns'])])
	ns['mei'] = ns.pop('')
	ET.register_namespace('', ns['mei'])
	uri = '{' + ns['mei'] + '}'

	return (ns, uri)


def _parse_tree(path: str, ns: dict): # -> Tuple
	"""
	Basic structure of <mei>:
	
	<mei> 
	  <meiHead/>
	  <music>
	    ...
	    <score>
	      <scoreDef/>
	      <section/>
	    </score>
	  </music>
	</mei>   
	"""
	tree = ET.parse(path)
	mei = tree.getroot()
	meiHead = mei.find('mei:meiHead', ns)
	music = mei.find('mei:music', ns)

	return (tree, meiHead, music)


def _handle_scoreDef(scoreDef: ET.Element, ns: dict, uri: str, args: argparse.Namespace): # -> None
	"""
	Basic structure of <scoreDef>:

	<scoreDef>
	  <staffGrp>
	    <staffGrp>
	      <staffDef/>
	     (<staffDef/>)
	    </staffGrp>  
	    <staffDef/>
	  </staffGrp>
	</scoreDef>

	The nested <staffGrp> is for the notehead notation and contains one <staffDef> in case of a single 
	staff, otherwise two; the lower <staffDef> is for the tablature. 
	"""
	staffGrp = scoreDef.find('mei:staffGrp', ns)
	
	# 1. Tablature <staffDef>: adapt or remove  
	tab_staffDef = staffGrp.find('mei:staffDef', ns)
	tab_meterSig = tab_staffDef.find('mei:meterSig', ns)
	tab_mensur = tab_staffDef.find('mei:mensur', ns)
	# Adapt
	if args.tab == 'y':
		n = tab_staffDef.get('n')
		lines = tab_staffDef.get('lines')
		not_type = tab_staffDef.get('notationtype')
		tuning = tab_staffDef.find('mei:tuning', ns)
		# TODO: this is a placeholder -- remove when GLT is ready to use
		if not_type == notationtypes['GLT']:
			not_type = notationtypes['FLT']
			tab_staffDef.attrib.pop('lines.visible', None)
			tab_staffDef.attrib.pop('notationsubtype', None)
			tab_staffDef.attrib.pop('valign', None)
		# Reset <staffDef> attributes
		tab_staffDef.set('n', str(int(n) + (1 if args.staff == 's' else 2)))
		if not_type != notationtypes['GLT']:
			tab_staffDef.set('lines', '5' if lines == '5' and args.type == 'FLT' else '6')
			tab_staffDef.set('notationtype', notationtypes[args.type])
		# Reset <tuning>	
		tuning.clear()
		for i, (pitch, octv) in enumerate(tunings[args.tuning]):
			course = ET.SubElement(tuning, uri + 'course',
								   n=str(i+1),
								   pname=pitch[0],
								   oct=str(octv),
								   accid='' if len(pitch) == 1 else ('f' if pitch[1] == 'b' else 's')
								  )
	# Remove
	else:
		staffGrp.remove(tab_staffDef)

	# 2. Notehead <staffGrp>: create and set as first element in <staffGrp>
	nh_staffGrp = ET.Element(uri + 'staffGrp')
	if args.staff == 'd':
		nh_staffGrp.set('symbol', 'bracket')
		nh_staffGrp.set('bar.thru', 'true')
	staffGrp.insert(0, nh_staffGrp)
	# Add <staffDef>(s)
	for i in [1] if args.staff == 's' else [1, 2]:
		nh_staffDef = ET.SubElement(nh_staffGrp, uri + 'staffDef',
									n=str(i),
									lines='5'
								   )
		if i == 1:
			nh_staffDef.set('dir.dist', '4')
		# Add <clef>
		if args.staff == 's':
			clef = _create_element(uri + 'clef', parent=nh_staffDef, atts=
								   [('shape', 'G'), 
									('line', '2'),
									('dis', '8'), 
									('dis.place', 'below')]
								  )
		else:
			clef = ET.SubElement(nh_staffDef, uri + 'clef', 
								 shape='G' if i==1 else 'F',
								 line='2' if i==1 else '4'
								)
		# Add <keySig>
		keySig = ET.SubElement(nh_staffDef, uri + 'keySig',
							   sig=_get_MEI_keysig(int(args.key)),
							   mode='minor' if args.mode == '1' else 'major'
							  )
		# Add <meterSig> or <mensur>
		if tab_meterSig is not None:
			nh_staffDef.append(tab_meterSig)
		elif tab_mensur is not None:
			nh_staffDef.append(tab_mensur)


def _get_MEI_keysig(key: int): # -> str:
	return str(key) + 's' if key > 0 else str(abs(key)) + 'f'


def _handle_section(section: ET.Element, ns: dict, uri: str, args: argparse.Namespace): # -> None
	"""
	Basic structure of <section>:

	<section>
	  <measure>
	    <staff>
	      <layer>
	        <chord/>
	        ...
	      </layer>   
	    </staff>
	   (<staff/>)
	    </dir>
	    <staff>
	      <layer> 
	        <tabGrp/>
	        ...
	      </layer>
	    </staff>
	  </measure>
	  ...
	</section>

	The upper <staff> is for the notehead notation; the lower for the tablature. The <dir>s 
	contain the flags for the notehead notation. In case of a double staff for the notehead
	notation, there is also a middle staff. 
	"""

	grids_dict = _call_java(['java', '-cp', cp, java_path, args.key, args.mode])
	mpcGrid = grids_dict['mpcGrid'] # list
	mpcGridStr = str(mpcGrid)
	altGrid = grids_dict['altGrid'] # list
	altGridStr = str(altGrid)
	pcGrid = grids_dict['pcGrid'] # list
	pcGridStr = str(pcGrid)

	event_count = 1
	for measure in section.iter(uri + 'measure'):
		accidsInEffect = [[], [], [], [], []]

		# 1. Tablature <staff>: adapt or remove
		# Adapt
		tab_staff = measure.find('mei:staff', ns)
		tab_staff.set('n', str(int(tab_staff.attrib['n']) + (1 if args.staff == 's' else 2)))
		tab_layer = tab_staff.find('mei:layer', ns)
		# Remove
		if args.tab == 'n':
			measure.remove(tab_staff)

		# 2. Notehead <staff>(s): create and set as first element(s) in <measure>
		# NB: in the single staff case, nh_staff_2 and its subelements are not used
		nh_staff_1 = ET.Element(uri + 'staff', n='1')
		nh_staff_2 = ET.Element(uri + 'staff', n='2')
		if args.staff == 'd':
			measure.insert(0, nh_staff_2)
		measure.insert(0, nh_staff_1)
		# Add <layer>(s)
		nh_layer_1 = ET.SubElement(nh_staff_1, uri + 'layer', n='1')
		nh_layer_2 = ET.SubElement(nh_staff_2, uri + 'layer', n='1')
		for tabGrp in tab_layer.iter(uri + 'tabGrp'):
			dur = tabGrp.get('dur')
			dots = tabGrp.get('dots')
			flag = tabGrp.find('mei:tabDurSym', ns)
			xml_id = 'e' + str(event_count)
			# Add <rest>s
			if len(tabGrp) == 1 and flag != None:
				_create_element(uri + 'rest', parent=nh_layer_1, atts=
								[('dur', dur),
								 ('xml:id', xml_id)]
							   )
				_create_element(uri + 'rest', parent=nh_layer_2, atts=
								[('dur', dur),
								 ('xml:id', xml_id + '_lwr')]
							   )
				# Add <dir>
				measure.insert(len(measure)-1, _make_dir(uri, xml_id, dur, dots))	
			# Add <chord>s	
			else:
				# If args.staff == 'd', chords are split over the two staffs, where there are
				# three possibilities: (1) both the upper and the lower staff have a chord;
				# (2) only the upper staff has a chord; (3) only the lower staff has a chord.
				# In the latter two cases, the other staff gets a <space> to fill the gap.
				# <chord>s can therefore not be SubElements, added to the parent <layer>
				# upon creation, but must be Elements appended at the end of this 'else'.
				chord_1 = _create_element(uri + 'chord', atts=
										  [('dur', dur), 
										   ('stem.visible', 'false'),
										   ('xml:id', xml_id)]
										  )
				chord_2 = _create_element(uri + 'chord', atts=
										  [('dur', dur), 
										   ('stem.visible', 'false'),
										   ('xml:id', xml_id + '_lwr')]
										  )
				# Add <note>s
				for element in tabGrp:
					if element != flag:
						midi_pitch = _get_midi_pitch(int(element.get('tab.course')), 
													 int(element.get('tab.fret')), 
													 args.tuning)
						midi_pitch_class = midi_pitch % 12
						# a. The note is in key and there are no accidentals to correct  
						if midi_pitch_class in mpcGrid and not any(accidsInEffect):
							pname = pcGrid[mpcGrid.index(midi_pitch_class)]
							accid = ''
						# b. The note is in key and there are accidentals to correct / the note 
						#    is not in key
						else:
							cmd = ['java', '-cp', cp, java_path, str(midi_pitch), args.key, 
					  			   mpcGridStr, altGridStr, pcGridStr, str(accidsInEffect)]
							spell_dict = _call_java(cmd)
							pname = spell_dict['pname'] # str
							accid = spell_dict['accid'] # str
							accidsInEffect = spell_dict['accidsInEffect'] # list
						nh_note = _create_element(uri + 'note', 
												  parent=chord_1 if args.staff == 's' else\
												         (chord_1 if midi_pitch >= 60 else chord_2), 
												  atts=[('pname', pname),
												        ('oct', str(_get_octave(midi_pitch))),
												   ('head.fill', 'solid')]
												 )
						if accid != '':
							nh_note.set('accid', accid)
				# Add <dir>
				_dir = None
				if flag != None:
					_dir = _make_dir(uri, xml_id, dur, dots)
					measure.insert(len(measure)-1, _dir)

				# Append <chord> or <space> to <layer>
				if args.staff == 's':
					nh_layer_1.append(chord_1)
				else:
					if len(chord_1) > 0 and len(chord_2) > 0:
						nh_layer_1.append(chord_1)
						nh_layer_2.append(chord_2)						
					else:
						space = _create_element(uri + 'space', atts=
												[('dur', dur),
												 ('xml:id', xml_id + '_spc')]
											   )
						nh_layer_1.append(chord_1 if len(chord_1) > 0 else space)
						nh_layer_2.append(chord_2 if len(chord_2) > 0 else space)
						# Adapt <dir>'s @startid if the upper staff has the space
						if len(chord_1) == 0 and _dir != None:
							_dir.set('startid', xml_id + '_spc')
			event_count += 1

		if verbose:
			for elem in measure:
				print(elem.tag, elem.attrib)
				for e in elem:
					print(e.tag, e.attrib)
					for ee in e:
						print(ee.tag, ee.attrib)
						for eee in ee:
							print(eee.tag, eee.attrib)


def _call_java(cmd: list, use_Popen: bool=False): # -> dict:
	# For debugging
	if use_Popen:
		process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=False)
		output, errors = process.communicate()
		outp = output.decode('utf-8') # str
		print(errors)
		print(outp)
	# For normal use
	else:
		process = run(cmd, capture_output=True, shell=False)
		outp = process.stdout # bytes
#	print(outp)
 
	return json.loads(outp)


def _make_dir(uri: str, xml_id: str, dur: int, dots: int): # -> 'ET.Element'
	d = ET.Element(uri + 'dir',
				   place='above', 
				   startid='#' + xml_id
				  )
	_create_element(uri + 'symbol', parent=d, atts=
					[('glyph.auth', 'smufl'), 
					 ('glyph.name', smufl_lute_durs[int(dur)])]
				   )
	if dots != None:
		_create_element(uri + 'symbol', parent=d, atts=
						[('glyph.auth', 'smufl'), 
						 ('glyph.name', smufl_lute_durs['.'])]
					   )

	return d


def _get_midi_pitch(course: int, fret: int, tuning: str): # -> int:
	# Determine the MIDI pitches for the open courses
	abzug = 0 if not '-' in tuning else 2
	open_courses = [67, 62, 57, 53, 48, (43 - abzug)]
	if tuning[0] != 'G':
		shift_interv = shift_intervals[tuning[0]]
		open_courses = list(map(lambda x: x+shift_interv, open_courses))
	return open_courses[course-1] + fret


def _get_octave(midi_pitch: int): # -> int:
	c = midi_pitch - (midi_pitch % 12)
	return int((c / 12) - 1)


def _create_element(name: str, parent: ET.Element=None, atts: list=[]): # -> ET.Element:
	"""
	Convenience method for creating an ET.Element or ET.SubElement object with a one-liner. 
	Useful because, in the conventional way, any attributes that contain a dot in their 
	name must be set separately with set():

	e = ET.Element(name, att_1='<val_1>', att_2='<val_2>', ..., att_n='<val_n>')
	e.set('<att_with_dot>', '<val>')

	or 

	se = ET.SubElement(parent, name, att_1='<val_1>', att_2='<val_2>', ..., att_n='<val_n>')
	se.set('<att_with_dot>', '<val>')
	"""
	o = ET.Element(name) if parent == None else ET.SubElement(parent, name)
	for a in atts:
		o.set(a[0], a[1])

	return o


def transcribe(infile: str, arg_paths: dict, args: argparse.Namespace): # -> None
	inpath = arg_paths['inpath']
	outpath = arg_paths['outpath']
	filename, ext = os.path.splitext(os.path.basename(infile)) # input file name, extension
	outfile = filename + '-dipl' + ext # output file

	# Handle namespaces
	ns, uri = _handle_namespaces(os.path.join(inpath, infile))

	# Get the main MEI elements
	tree, meiHead, music = _parse_tree(os.path.join(inpath, infile), ns)

	# Handle <scoreDef>
	score = music.findall('.//' + uri + 'score')[0]
	scoreDef = score.find('mei:scoreDef', ns)
	_handle_scoreDef(scoreDef, ns, uri, args)

	# Handle <section>
	section = score.find('mei:section', ns)
	_handle_section(section, ns, uri, args)

	# Fix indentation and write to file
	ET.indent(tree, space="\t", level=0)
	tree.write(os.path.join(outpath, outfile))	