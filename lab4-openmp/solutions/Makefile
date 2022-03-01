EXE=hello_world \
	pi_seq \
	pi_omp
CFLAGS=-Wall -fopenmp -g

all: $(EXE)

% : %.c
	gcc $(CFLAGS) -o $@ $<

clean:
	rm -f $(EXE)
