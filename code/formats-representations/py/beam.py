import music21 as music21
from sys import argv

script, file_ = argv

lb = '\n'

# https://web.mit.edu/music21/doc/moduleReference/moduleStreamMakeNotation.html#music21.stream.makeNotation.makeBeams
# https://web.mit.edu/music21/doc/moduleReference/moduleBeam.html

#time_sig = '4/4'
#bars_array = [ [bar_1_note_1_length, bar_1_note_2_length, ... ], [bar_2_note_1_length, ...] ...]
#BWV 846, altus
#bars_per_voice = [[(0.5, 'r', ''), (0.5, 'n', 'c'), (0.5, 'n', 'd'), (0.5, 'n', 'e'), 
#                   (0.75, 'n', 'f'), (0.125, 'n', 'g'), (0.125, 'n', 'f'), (0.5, 'n', 'e'), (0.5, 'n', 'a')], 
#                  [(0.5, 'n', 'd'), (0.75, 'n', 'g'), (0.25, 'n', 'a'), (0.25, 'n', 'g'), (0.25, 'n', 'f'),
#                   (0.25, 'n', 'e'), (0.25, 'n', 'f'), (0.25, 'n', 'e'), (0.25, 'n', 'd'), 
#                   (0.25, 'n', 'c'), (0.25, 'n', 'd'), (0.25, 'n', 'c'), (0.25, 'n', 'b')]]

with open(file_) as file:
	data = file.read().split('\n')

#with open('notes.txt') as file:  
#    data = file.read().split('\n')


# Remove last (empty) line caused by final line break on notes.txt
data = data[:-1]


# Split per voice
data_per_voice = []
curr_voice = []
for i in range(1, len(data)): # skip first line ("voice=0")
	line = data[i]
	if not 'voice' in line: 
		curr_voice.append(line)
	if 'voice' in line or i == len(data)-1:
		data_per_voice.append(curr_voice)
		curr_voice = []

#print(len(data_per_voice))
#print(data_per_voice[0][-1])
#print(data_per_voice[1][-1])
#print(data_per_voice[2][-1])
#print(data_per_voice[3][-1])


# Call compute_beams() on each element of all_voices

# each element of all_voices is a voice, i.e., a list of bars 
# each element of a bar is a list of strings, containing 
# at element [0]  : the meter  
# at elements [1:]: the notes and rest MEI elements 
all_voices = []
for data in data_per_voice:
	bars_curr_voice = []
	new_bar_ind = [i for i in range(len(data)) if 'meter=' in data[i]]
	for i in range(len(new_bar_ind)):
		if i == len(new_bar_ind)-1:
			bars_curr_voice.append(data[new_bar_ind[i]:])
		else:
			bars_curr_voice.append(data[new_bar_ind[i]:new_bar_ind[i+1]])
	all_voices.append(bars_curr_voice)


def compute_beams(bars_curr_voice):
	beams_output = []
	# For each bar
	for i in range(len(bars_curr_voice)):
		mei_bar = bars_curr_voice[i]
		# Determine time_sig and remove from mei_bar
		time_sig = mei_bar[0][mei_bar[0].index('\''):mei_bar[0].rfind('\'')+1]
		mei_bar = mei_bar[1:]
		# For each item in mei_bar: add to new music21 measure   
		m21_bar = music21.stream.Measure()
		m21_bar.timeSignature = music21.meter.TimeSignature(time_sig)
		for j in range(len(mei_bar)):
			xml_note = mei_bar[j]
			if not 'chord' in xml_note and not 'tuplet' in xml_note:
				if not 'mRest' in xml_note:
					begin = xml_note.index('dur=\'') + len('dur=\'')
					dur_str = xml_note[begin:xml_note.index('\'', begin)]
					if dur_str != 'breve' and dur_str != 'long':
						dur = int(dur_str)
					else:
						if dur_str == 'breve':
							dur = 0.5
						else:
							dur = 0.25
					# Durations in music21 are counted in quarter notes
					note_length = 4/float(dur)
					if 'note' in xml_note:
						m21_bar.append(music21.note.Note('C', quarterLength=note_length))                    
					elif 'rest' in xml_note:
						m21_bar.append(music21.note.Rest(quarterLength=note_length))
#						m21_bar.append(music21.note.Rest('rest', quarterLength=note_length)) # key 'rest' no longer valid after music21 upgrade
				else:
					split = time_sig[1:-1].split('/')
					note_length = int(int(split[0]) / int(split[1]))
					m21_bar.append(music21.note.Rest(quarterLength=4))
#					m21_bar.append(music21.note.Rest('rest', quarterLength=4)) # key 'rest' no longer valid after music21 upgrade

		# Make beams
		m21_bar.makeBeams(inPlace=True)
		# Remove time_signature from m21_bar
		m21_bar = [item for item in m21_bar if type(item) is music21.note.Note or type(item) is music21.note.Rest]    
		beamed_mei = ''
		# mei_bar and m21_bar have corresponding elements and can be indexed concurrently
		for j in range(len(m21_bar)):
			m21_note = m21_bar[j]
			mei_note = mei_bar[j]
			mei_note = mei_note.strip()
			# In case of bar rest
			if len(m21_bar) == 1 and (type(m21_bar[0]) is music21.note.Rest):
				beamed_mei = beamed_mei + mei_note + lb 
			# In case of note or rest
			else: # beams = m21_note.beams.beamsList 
				# Rest: add 
				if m21_note.isRest: # if len(beams) == 0: # if type(m21_bar[j]) is music21.note.Rest:
					beamed_mei = beamed_mei + mei_note + lb
				# Note: only the beginning and end of the topmost beam (beams[0]) are relevant
				else:
					beams = m21_note.beams.beamsList
					if len(beams) == 0:                        
						beamed_mei = beamed_mei + mei_note + lb
					if len(beams) > 0:
						# Start of beaming group
						if '1/start' in str(beams[0]):
#							beamed_mei = beamed_mei + '<beam>' + lb
#							beamed_mei = beamed_mei + '    ' + mei_note + lb
							beamed_mei = beamed_mei + '<beam>' + mei_note + lb
						# End of beaming group beam 
						elif '1/stop' in str(beams[0]):
#							beamed_mei = beamed_mei + '    ' + mei_note + lb
#							beamed_mei = beamed_mei + '</beam>' + lb
							beamed_mei = beamed_mei + mei_note + '</beam>' + lb
						# Middle beam of beaming group
						else:
#							beamed_mei = beamed_mei + '    ' + mei_note + lb
							beamed_mei = beamed_mei + mei_note + lb

		beams_output.append(beamed_mei)

	return beams_output

as_string = ''
for i in range(0, len(all_voices)):
	bars_curr_voice = all_voices[i]
#	as_string += 'voice ' + str(i) + lb
	for item in compute_beams(bars_curr_voice):
		as_string += item + 'end of bar' + lb
	as_string += 'end of voice' 
	if i < len(all_voices)-1:
		as_string += lb  

print(as_string)

write_file = False
if write_file:
	with open('notes-beamed.txt', 'w') as f: 
		f.write(as_string) 
