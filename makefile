.PHONY: init, run

init:
		
		pip3 install -r "requirements.txt"
		-(rm -rf Sparkov_Data_Generation)
		-(git clone https://github.com/namebrandon/Sparkov_Data_Generation.git)
		-gdown  1RytEvBvyOUkvXpZMjFHvy78tSxd6fK2P
		-tar -xzvf data2.tar.gz data2.csv
		-rm data2.tar.gz
run:
		time python3 etl.py && time python3 app.py "y" "n" && python3 app.py "n" "n"