TARGET=gravitation-box
FILES=main.cu particle_system_cpu.cu

${TARGET}:
	nvcc -Icuda-samples-common --std=c++14 -lGL -lGLU -lglut -o ${TARGET} ${FILES}

clean:
	-rm -rf ${TARGET}

.PHONY: clean
