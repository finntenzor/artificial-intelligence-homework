cpp_src=$(wildcard ./src/*.cpp)
cu_src=$(wildcard ./src/*.cu)
cpp_obj=$(patsubst ./src/%.cpp, ./obj/%.obj, $(cpp_src))
cu_obj=$(patsubst ./src/%.cu, ./obj/%.obj, $(cu_src))
obj=$(cpp_obj) $(cu_obj)
inc_path=./include/

CC=nvcc
target=main.exe
INCLUDE_ARG=-I $(inc_path)
FLAGS=-rdc=true $(INCLUDE_ARG)

default:build
$(target):$(obj)
	$(CC) $(obj) -o $(target)
$(cpp_obj):./obj/%.obj:./src/%.cpp
	$(CC) -c $< -o $@ $(FLAGS)
$(cu_obj):./obj/%.obj:./src/%.cu
	$(CC) -c $< -o $@ $(FLAGS)
build:$(target)
train:
	./main.exe --train --config=mnist.config
predict:
	./main.exe --predict --config=mnist.config
run:
	./main.exe --train --predict --config=mnist.config
clean:
	-rm *.exp *.lib *.exe *.log ./obj/*.obj
count:
	bash count.sh
before:
	python dev.py prod
after:
	python dev.py dev
.PHONY: default train predict run clean count before after
