import re 

def isfloat(str):
    return str.replace('.','').isdecimal()

def process_time_string(line):
    compile_time = re.findall(r'\d+\.\d+', line)
    print(compile_time[0], end=", ")
    mem_usage = re.findall(r'\d+ KB', line) 
    mem = mem_usage[0].replace(' KB', '')
    print(mem, end=", ")

def process_exe_size_string(line):
    file_size = re.findall(r'\d+K', line)
    if (len(file_size) > 0):
        file_size = file_size[0].replace('K', '')        
        print(file_size, end=', ')
    else:
        file_size = re.findall(r'\d+M', line)
        file_size = file_size[0].replace('M', '')          
        file_size = int(file_size[0])
        file_size = file_size * 1000
        print(file_size, end=", ")

def process_runtime(dirname, exe_name):
    log_file = dirname + "/" + exe_name + ".log"
    with open(log_file, 'r') as fp:
        for index, line in enumerate(fp):
            if ('***' in line):
                # remove comma separating values in output
                line = line.replace(',', '')
                times = [t for t in line.split() if isfloat(t)]
                print(times[0].strip(), ', ', times[1].strip(), sep='')

file = open('build.log', 'r')
lines = file.readlines()

print("benchmark, compile time (sec.), memory usage (KB), executable size (bytes), min runtime, max runtime")

with open('build.log', 'r') as fp:
    executable = ''
    for index, line in enumerate(fp):
        if ".x86_64" in line:
            executable = line.strip()
            dirname = (executable.split("-"))[0]
            print(executable, end=", ")      
        elif ".aarch64" in line:
            executable = line.strip()
            dirname = (executable.split("-"))[0]
            print(executable, end=", ")      
        elif "compile time" in line:
            process_time_string(line.strip())
        elif "executable size" in line:
            process_exe_size_string(line.strip())
            process_runtime(dirname, executable)

file.close()

