import os
import subprocess


os.chdir("../build")

cmd = "cmake -DN_DISCR=128 -DSOLVER=ETDRK4 -DUSE_CUDA=off .."
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process.wait()

cmd = "make"
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process.wait()

cmd = "perf stat ./project"
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time, out = process.communicate()

time = float(time.decode("UTF-8").split("\n")[0])
freq = float("".join(out.decode("UTF-8").split("\n")[7].split("#")[1].split("GHz")[0].split(",")).strip())

print(freq)
print(time*freq/2e3)
