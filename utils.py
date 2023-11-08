import json
import os
import subprocess
from subprocess import Popen, PIPE, run

cp_dirs = ['code/formats-representations/lib/*',
		   'code/formats-representations/bin/',
		   'code/machine_learning/lib/*',
		   'code/machine_learning/bin/',
		   'code/melody_models/lib/*',
		   'code/melody_models/bin/',
		   'code/path/lib/*',
		   'code/path/bin/',
		   'code/utils/lib/*',
		   'code/utils/bin/',
		   'code/voice_separation/lib/*',
		   'code/voice_separation/bin/',
		   'code/tabmapper/lib/*',
		   'code/tabmapper/bin/'
		  ]
cp = (':' if os.name == 'posix' else ';').join(cp_dirs)



def call_java(cmd: list, return_outp, use_Popen: bool=False): # -> dict:
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
 
	if return_outp:
		return json.loads(outp)