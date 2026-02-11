
# python main.py --config configs/config_MedQA.yaml --method CP-unconditional --shift LinearShift 
# python main.py --config configs/config_MedQA.yaml --method CP-conditional --shift LinearShift
# python main.py --config configs/config_MedQA.yaml --method Online --shift LinearShift 

for shift in LinearShift SquareShift SineShift BernoulliShift
do
	for dataset in MedQA Wiki
	do
		for method in CP-unconditional CP-conditional Online
		do
			python main.py --config configs/config_${dataset}.yaml --method ${method} --shift ${shift}
		done
	done
done

for dataset in WildChat
do
	for method in CP-unconditional CP-conditional Online
	do
		python main_wildchat.py --config configs/config_${dataset}.yaml --method ${method} 
	done
done