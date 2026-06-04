#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgm_io.h"

static void skip_comments(FILE *fp){
    int ch;
    while((ch=fgetc(fp)) == '#'){
        while((ch=fgetc(fp))!='\n' && ch!=EOF){}
    }
    ungetc(ch,fp);  //退回一个输入
}

unsigned char* read_pgm(const char* filename, int* width, int* height){
    FILE* fp=fopen(filename,"rb");
    if(fp==NULL){
        printf("File %s open error.\n",filename);
        return NULL;
    }
    char magic[3];
    int max_value;
    if (fscanf(fp,"%2s",magic) != 1 ||
        fscanf(fp,"%d",width) != 1 ||
        fscanf(fp,"%d",height) != 1 ||
        fscanf(fp,"%d",&max_value) != 1) {
        printf("PGM parse error: %s\n", filename);
        fclose(fp);
        return NULL;
    }
    if (fgetc(fp) == EOF) {
        printf("PGM parse error: %s\n", filename);
        fclose(fp);
        return NULL;
    }

    int size = (*width) * (*height);
    unsigned char* data = malloc(size);
    if (data == NULL) {
        printf("Memory allocation failed for %s\n", filename);
        fclose(fp);
        return NULL;
    }
    if (fread(data, 1, size, fp) != (size_t)size) {
        printf("PGM read error: %s\n", filename);
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    return data;
}

int write_pgm(const char* filename, const unsigned char* data, int width, int height){
    FILE* fp=fopen(filename,"wb");
    if(fp==NULL){
        printf("File %s open error.",filename);
        return 0;
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width*height,fp);
    fclose(fp);
    return 1;
}