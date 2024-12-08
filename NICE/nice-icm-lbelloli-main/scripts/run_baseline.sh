#!/bin/zsh
#
b=()
for s in /media/data2/old_patients/subjects/*;
do
		b+=(${s:t})
		# python run_baseline.py --runid=baseline --subject=${s:t}
done

parallel -j4 --joblog job-pipeline.log --resume --resume-failed --tag --delay 30 python run_baseline.py --runid=baseline --subject={} ::: $b
