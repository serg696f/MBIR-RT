#define _POSIX_C_SOURCE 200809L
#include "utils.h"
#include "tomography_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

int mj_makedirs(const char *path)
{
    char tmp[MJ_MAX_PATH];
    strncpy(tmp, path, sizeof(tmp)-1);
    char *last=strrchr(tmp,'/');
    if (last && strchr(last,'.')) *last='\0';
    if (!tmp[0]) return 0;
    for (char *p=tmp+1;*p;p++) {
        if(*p=='/'){*p='\0'; mkdir(tmp,0755); *p='/';}
    }
    mkdir(tmp,0755);
    return 0;
}

int mj_save_raw(const char *path, const f32 *data, const i32 *dims, i32 ndims)
{
    FILE *f=fopen(path,"wb"); if(!f){perror(path);return -1;}
    fwrite(&ndims,sizeof(i32),1,f); fwrite(dims,sizeof(i32),ndims,f);
    i32 total=1; for(i32 d=0;d<ndims;d++) total*=dims[d];
    fwrite(data,sizeof(f32),total,f);
    fclose(f); return 0;
}

int mj_load_raw(const char *path, f32 **data, i32 **dims, i32 *ndims)
{
    FILE *f=fopen(path,"rb"); if(!f){perror(path);return -1;}
    if (fread(ndims,sizeof(i32),1,f)!=1) { fclose(f); return -1; }
    *dims=(i32*)malloc(*ndims*sizeof(i32));
    if (fread(*dims,sizeof(i32),*ndims,f)!=(size_t)*ndims) { fclose(f); free(*dims); return -1; }
    i32 total=1; for(i32 d=0;d<*ndims;d++) total*=(*dims)[d];
    *data=(f32*)malloc(total*sizeof(f32));
    if (fread(*data,sizeof(f32),total,f)!=(size_t)total) { fclose(f); free(*dims); free(*data); return -1; }
    fclose(f); return 0;
}

void mj_gen_full_indices(i32 rows, i32 cols, int use_ror, i32 *out, i32 *n_out)
{
    if (use_ror) { mj_gen_ror_mask(rows,cols,out,n_out); return; }
    i32 n=rows*cols;
    for(i32 i=0;i<n;i++) out[i]=i;
    if(n_out) *n_out=n;
}

void mj_gen_partition_sequence(i32 n_gran, i32 max_iter, i32 *seq_out)
{
    for(i32 i=0;i<max_iter;i++) seq_out[i]=i%n_gran;
}

f32 mj_nmae(const f32 *a, const f32 *b, i32 n)
{
    f32 num=0.f,den=0.f;
    for(i32 i=0;i<n;i++){num+=mj_absf(a[i]-b[i]);den+=mj_absf(b[i]);}
    return (den>0.f)?num/den:0.f;
}

void mj_lerp(const f32 *a, const f32 *b, f32 t, f32 *out, i32 n)
{
    f32 tm=1.f-t;
    for(i32 i=0;i<n;i++) out[i]=a[i]*tm+b[i]*t;
}

long mj_get_memory_stats(int print_results)
{
    long avail=0;
#ifdef __linux__
    FILE *f=fopen("/proc/meminfo","r");
    if(f){char line[128];
        while(fgets(line,sizeof(line),f))
            if(strncmp(line,"MemAvailable:",13)==0){long kb;sscanf(line+13,"%ld",&kb);avail=kb*1024;break;}
        fclose(f);}
#endif
    if(print_results) printf("[mbirjax] Available memory: %.2f GB\n",avail/1e9);
    return avail;
}
