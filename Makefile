# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++11 -Xcompiler -Wall

# Target executable
TARGET = vector_add

# Source and object files
MAIN_SRC = main.cu
MAIN_OBJ = main.o

# Default rule to build the project
all: $(TARGET)

# Linking the executable
$(TARGET): $(MAIN_OBJ)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(MAIN_OBJ)

# Compiling the main source file
$(MAIN_OBJ): $(MAIN_SRC)
	$(NVCC) $(CFLAGS) -c $(MAIN_SRC) -o $(MAIN_OBJ)

# Clean up build files
clean:
	rm -f $(MAIN_OBJ) $(TARGET)

# Phony targets
.PHONY: all clean
