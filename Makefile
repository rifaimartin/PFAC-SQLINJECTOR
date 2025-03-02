# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -I./include -I./src

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/sequential/aho_corasick.c \
       $(SRC_DIR)/patterns.c

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Executable name
TARGET = $(BIN_DIR)/sql_detector

# Create necessary directories
$(shell mkdir $(OBJ_DIR) 2>NUL)
$(shell mkdir $(BIN_DIR) 2>NUL)
$(shell mkdir $(OBJ_DIR)\sequential 2>NUL)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/sequential/%.o: $(SRC_DIR)/sequential/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /Q /F /S $(OBJ_DIR)\* $(BIN_DIR)\*

.PHONY: all clean