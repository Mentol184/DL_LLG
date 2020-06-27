currents='1e9 2e9 3e9 4e9 5e9 6e9 7e9 8e9 9e9 10e9 15e9 20e9 25e9 30e9 35e9 40e9 45e9 50e9 55e9 60e9 65e9 70e9 75e9 80e9 85e9 90e9 95e9 100e9 110e9 120e9 130e9 140e9 150e9 160e9 170e9 180e9 190e9 200e9 220e9 240e9 260e9 280e9 300e9 350e9 400e9'


	for current in $currents
		do

		sed -i '21i\J = vector(0, 0, '$current')' scriptOPP.txt 
		
		    mumax3 scriptOPP.txt
		    mv scriptOPP.out/table.txt scriptOPP.out/table_$current.txt

		sed -i '/J = vector(0, 0, '$current')/d' ./scriptOPP.txt
		
		
	done


# enter in output directory
cd scriptOPP.out
# run script
./script_cat_and_delete.sh 
