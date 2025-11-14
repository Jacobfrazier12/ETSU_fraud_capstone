.PHONY: init, run

init:
		
		pip3 install -r "requirements.txt"
		-(rm -rf Sparkov_Data_Generation)
		-(git clone https://github.com/namebrandon/Sparkov_Data_Generation.git)
		-(cd  ./Sparkov_Data_Generation && rm -rf .git)
		-tar -xzvf data2.tar.gz data2.csv
		
run:
		time python3 etl.py && time python3 app.py