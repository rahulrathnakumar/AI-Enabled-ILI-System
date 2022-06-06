import urllib.request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", help = 'Raspberry Pi Addr')
args = parser.parse_args()

ip = args.ip
link = 'http://' + ip + ':8080'
try:
	while True:
		f = urllib.request.urlopen(link)
		myfile = f.read()
		print("Orig: ", myfile, "Convert:", float(myfile))
		myfile.split()

except:
	exit()
