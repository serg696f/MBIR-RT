# =============================================================================
# Makefile — plain-C MBIRJAX library
#
# Targets:
#   all          build libmbirjax.a (default)
#   demo         build the standalone denoising demo
#   clean        remove build artefacts
#   install      copy headers + library to PREFIX (default /usr/local)
# =============================================================================

CC       ?= gcc
CFLAGS   := -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -Iinclude
LDFLAGS  := -lm
PREFIX   ?= /usr/local

SRCS     := $(wildcard src/*.c)
OBJS     := $(patsubst src/%.c,build/%.o,$(SRCS))
LIB      := build/libmbirjax.a

.PHONY: all demo clean install

all: build $(LIB)

build:
	mkdir -p build

$(LIB): $(OBJS)
	ar rcs $@ $^
	@echo "Built $@"

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

demo: all
	$(CC) $(CFLAGS) -DDENOISING_DEMO -o build/denoising_demo \
	    src/denoising.c src/vcd_utils.c $(LDFLAGS)
	@echo "Built build/denoising_demo"

clean:
	rm -rf build

install: all
	install -d $(PREFIX)/include/mbirjax
	install -m 644 include/*.h $(PREFIX)/include/mbirjax/
	install -d $(PREFIX)/lib
	install -m 644 $(LIB) $(PREFIX)/lib/
	@echo "Installed to $(PREFIX)"
