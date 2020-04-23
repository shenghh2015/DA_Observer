import os

job_folder = './job_folder'
job_list_file = './GPU_jobs.txt'

job_list = []
with open(job_list_file, 'rb') as f:
	lines = f.readlines()
	for line in lines:
		splits = line.strip().split(' ')
		if len(splits) > 0:
			if splits[0] == 'JOB:':
				print(line.replace('JOB: ', ''))
				job_list.append(line.replace('JOB: ', ''))

for i in range(len(job_list)):
	job_file = os.path.join(job_folder, 'job_{}.sh'.format(i))
	if os.path.exists(job_file):
		os.system('rm -f {}'.format(job_file))
	with open(job_file, 'w+') as f:
		f.write('{}\n'.format(job_list[i]))


