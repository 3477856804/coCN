CC = gcc
CFLAGS = -O2 -Wall

co: co.c
	$(CC) $(CFLAGS) -o co co.c -lm

.PHONY: test clean

test: co
	./co 例子/张量演示.co
	./co 例子/线性回归.co
	./co 例子/异或网络.co
	./co 例子/分类演示.co

clean:
	rm -f co
