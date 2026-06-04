#ifndef PGM_IO_H
#define PGM_IO_H

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* read_pgm(const char* filename, int* width, int* height);
int write_pgm(const char* filename, const unsigned char* data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif