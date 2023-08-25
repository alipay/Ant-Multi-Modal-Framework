A C program to build PHOC descriptors of the query strings. The cphoc library must be compiled as follows:

```
gcc -c -fPIC `python-config --cflags` cphoc.c

gcc -shared -o cphoc.so cphoc.o `python-config --ldflags`
```

