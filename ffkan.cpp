#include <math.h>
#include <iostream>

//The function increment the value of out
//out should be initialized to zero beforehand
void ffkan( float* x, float* coeff, float* bias, int bs, int inputdim, int outputdim, int gridsize, float* out )
{
const int s_bs_out = outputdim;
const int s_bs_x = inputdim;

//Coeff shape (2,inputdim,outputdim,gridsize)
const int s_d_coeff= inputdim*outputdim*gridsize;
const int s_i_coeff = outputdim*gridsize;
const int s_o_coeff = gridsize;

for( int i = 0 ; i < bs ; i++)
for( int j = 0 ; j < inputdim ; j++)
{
 float xx =  x[i*s_bs_x+j];
 float c0 = cosf(xx); 
 float s0 = sinf(xx);
for( int l = 0 ; l < outputdim ; l++)
{
 float ckm = 1.0f;
 float skm = 0.0f;
for( int k = 1 ; k < gridsize+1 ; k++)
{
 //float xx =  x[i*s_bs_x+j];
 //For better performance We use trig formula to compute ck,sk from ck-1, sk-1, cos(xx),sin(xx)
 //But this form is better to obtain the bacwkard pass
 //float c = cos(k*xx); 
 //float s = sin(k*xx);
 float c = ckm*c0-skm*s0;
 float s = skm*c0+ckm*s0;
 ckm = c;
 skm = s;

 out[i*s_bs_out+l] += coeff[s_d_coeff*0 + s_i_coeff*j + s_o_coeff*l + k-1] * c;
 out[i*s_bs_out+l] += coeff[s_d_coeff*1 + s_i_coeff*j + s_o_coeff*l + k-1] * s;
}

}
}

for( int i = 0 ; i < bs ; i++)
for( int l = 0 ; l < outputdim ; l++)
    out[i*s_bs_out+l] += bias[l];

}


//The function should not use the value of out and doesn't
void ffkan_b(float *x, float *xb, float *coeff, float *coeffb, float *bias, 
        float *biasb, int bs, int inputdim, int outputdim, int gridsize, float
        *out, float *outb) {
    const int s_bs_out = outputdim;
    const int s_bs_x = inputdim;
    const int s_d_coeff = inputdim*outputdim*gridsize;
    const int s_i_coeff = outputdim*gridsize;
    const int s_o_coeff = gridsize;

    //These loop should be iterated backward according to autodiff, but if we assume (even if not true) commutativity and associativity of floating point addition 
    //we can iterate them in the nicer looking normal order
    for( int i = 0 ; i < bs ; i++)
        for( int l = 0 ; l < outputdim ; l++)
            biasb[l] = biasb[l] + outb[i*s_bs_out + l];

    //These loop should be iterated backward according to autodiff, but if we assume (even if not true) commutativity and associativity of floating point addition 
    //we can iterate them in the nicer looking normal order
    for( int i = 0 ; i < bs ; i++)
    for( int j = 0 ; j < inputdim ; j++)
    {
        float xx =  x[i*s_bs_x+j];
        float c0 = cosf(xx); 
        float s0 = sinf(xx);
    for( int l = 0 ; l < outputdim ; l++)
    {
        float ckm = 1.0f;
        float skm = 0.0f;
        for( int k = 1 ; k < gridsize+1 ; k++)
                   {
                    float xxb = 0.0;
                    //For better performance We use trig formula to compute ck,sk from ck-1, sk-1, cos(xx),sin(xx)
                    //But this form is better to obtain the bacwkard pass
                    //float c = cos(k*xx);
                    //float s = sin(k*xx);
                    float c = ckm*c0-skm*s0;
                    float s = skm*c0+ckm*s0;
                    ckm = c;
                    skm = s;
                    float cb;
                    float sb;
                    coeffb[s_d_coeff*1 + s_i_coeff*j + s_o_coeff*l + k-1] += s*outb[i*s_bs_out+l];
                    sb = coeff[s_d_coeff*1+s_i_coeff*j+s_o_coeff*l+k-1]*outb[i*s_bs_out+l];
                    coeffb[s_d_coeff*0 + s_i_coeff*j + s_o_coeff*l + k-1] += c*outb[i*s_bs_out+l];
                    cb = coeff[s_d_coeff*0+s_i_coeff*j+s_o_coeff*l+k-1]*outb[i*s_bs_out+l];
                    xxb = k*c*sb - k*s*cb;
                    xb[i*s_bs_x + j] += xxb;
                   }
    }
    }
}




#ifdef DEMO

int main()
{
int bs = 3;
int inputdim = 4;
int outputdim = 5;
int gridsize = 6;

float *x = new float[bs*inputdim];
float *coeff = new float[2*inputdim*outputdim*gridsize];
float *bias = new float[outputdim];
float *out = new float[bs*outputdim];

float *xb = new float[bs*inputdim];
float *coeffb = new float[2*inputdim*outputdim*gridsize];
float *biasb = new float[outputdim];
float *outb = new float[bs*outputdim];


ffkan(x,coeff,bias,bs,inputdim,outputdim,gridsize,out);
ffkan_b(x,xb,coeff,coeffb,bias,biasb,bs,inputdim,outputdim,gridsize,out,outb);

return 0;
}


#endif